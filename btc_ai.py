# btc_ai.py
from __future__ import annotations

import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# optional: OpenAI (사용 가능하면 리포트/설명 생성)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class MarketState:
    ts: float
    symbol: str
    price: float
    base: float
    trigger: float
    drop_from_base_pct: float

    vol: float  # log-return std (너의 _estimate_vol 값)
    # cap snapshot
    free_usdt: float
    total_usdt: float
    base_total: float
    trade_cap: float
    used: float
    remaining_cap: float
    spendable_free: float

    # positions
    pool_btc: float
    pool_avg: float
    pool_cost: float

    all_btc: float
    all_avg: float
    all_cost: float

    # order/health (간단히)
    tp_order_id: Optional[int]


@dataclass
class GateDecision:
    regime: str  # "ATTACK" | "NEUTRAL" | "DEFENSE" | "HALT"
    allow_buy: bool
    buy_mult: float
    allow_tp_refresh: bool
    tp_refresh_mult: float
    notes: str


@dataclass
class TuneDecision:
    # “적용 가능한 범위” 안에서만 변경
    grid_step_pct: float
    take_profit_pct: float
    tp_refresh_sec: int
    recenter_threshold_pct: float

    notes: str


class BTCAI:
    """
    A) 리포트/모니터링: LLM(선택)
    B) 파라미터 자동 튜닝: 규칙 기반 + 강한 가드레일
    C) 리스크 게이트/레짐: 규칙 기반 + 히스테리시스(쿨다운)
    """

    def __init__(self, cfg, db, *, client=None):
        self.cfg = cfg
        self.db = db
        self.client = client

        # 런타임 캐시
        self._last_report_ts = 0.0
        self._last_tune_ts = 0.0
        self._last_gate_ts = 0.0

        self._last_regime = "NEUTRAL"
        self._last_regime_change_ts = 0.0

        # LLM
        self._llm = None
        if getattr(cfg, "ai_enable_report", True) and OpenAI is not None:
            api_key = getattr(cfg, "openai_api_key", "") or ""
            if api_key:
                try:
                    self._llm = OpenAI(api_key=api_key)
                except Exception:
                    self._llm = None

    # -------------------------
    # hard guardrails
    # -------------------------
    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def _smooth(self, old: float, new: float, alpha: float = 0.25) -> float:
        # alpha=0.25면 25%만 따라감 (튐 방지)
        return float(old) * (1.0 - alpha) + float(new) * alpha

    # -------------------------
    # Gate / Regime
    # -------------------------
    def decide_gate(self, s: MarketState) -> GateDecision:
        """
        레짐 규칙 예시(안전형):
        - 변동성/급락이 심하면 DEFENSE/HALT로 가서 매수 축소 또는 중단
        - 횡보/저변동이면 NEUTRAL/ATTACK
        히스테리시스(쿨다운) 적용: 자주 바뀌지 않게
        """
        now = s.ts
        cooldown = float(getattr(self.cfg, "ai_regime_cooldown_sec", 180))  # 3분 기본
        if now - self._last_gate_ts < float(getattr(self.cfg, "ai_gate_interval_sec", 10)):
            # 너무 자주 계산 안 해도 됨
            return self._gate_from_regime(self._last_regime, s)

        self._last_gate_ts = now

        # 핵심 지표
        drop = s.drop_from_base_pct  # base 대비 하락률(양수면 하락)
        vol = s.vol

        # 사용자 설정(기본값은 “보수적”)
        vol_hi = float(getattr(self.cfg, "ai_vol_high", 0.012))
        vol_extreme = float(getattr(self.cfg, "ai_vol_extreme", 0.020))
        drop_def = float(getattr(self.cfg, "ai_drop_defense_pct", 0.030))      # 3% 급락 -> 방어
        drop_halt = float(getattr(self.cfg, "ai_drop_halt_pct", 0.060))         # 6% 급락 -> 일시 중단
        min_free = float(getattr(self.cfg, "ai_min_free_usdt", 10.0))

        # 레짐 판정
        if s.free_usdt < min_free and s.pool_btc <= 0:
            target = "HALT"
            notes = "free USDT too low"
        elif drop >= drop_halt or vol >= vol_extreme:
            target = "HALT"
            notes = "extreme drop/vol"
        elif drop >= drop_def or vol >= vol_hi:
            target = "DEFENSE"
            notes = "high drop/vol"
        else:
            # 완만한 시장에서는 NEUTRAL, 저변동 + 충분한 여력 있으면 ATTACK
            if vol > 0 and vol < float(getattr(self.cfg, "ai_vol_low", 0.005)) and s.remaining_cap > 0:
                target = "ATTACK"
                notes = "low vol / room to trade"
            else:
                target = "NEUTRAL"
                notes = "normal"

        # 히스테리시스: 너무 자주 바뀌지 않게
        desired = target
        if desired != self._last_regime:
            if now - self._last_regime_change_ts < cooldown:
                # ✅ "원래 바꾸고 싶었던 레짐"을 notes에 남김
                notes = f"hold regime (cooldown). want={desired} keep={self._last_regime}"
                target = self._last_regime
            else:
                self._last_regime = desired
                self._last_regime_change_ts = now
                target = desired

        return self._gate_from_regime(self._last_regime, s, notes_override=notes)

    def _gate_from_regime(self, regime: str, s: MarketState, notes_override: str = "") -> GateDecision:
        if regime == "ATTACK":
            return GateDecision(regime, True, 1.10, True, 1.0, notes_override or "attack")
        if regime == "NEUTRAL":
            return GateDecision(regime, True, 1.00, True, 1.0, notes_override or "neutral")
        if regime == "DEFENSE":
            # 매수 축소 + TP 갱신 덜 자주(주문 취소/갱신 스트레스 감소)
            return GateDecision(regime, True, 0.50, True, 1.5, notes_override or "defense")
        # HALT
        return GateDecision(regime, False, 0.0, True, 2.0, notes_override or "halt buys")

    # -------------------------
    # Parameter tuning (bounded)
    # -------------------------
    def decide_tuning(self, s: MarketState, current: Dict[str, Any]) -> TuneDecision:
        """
        “안전한 자동 튜닝”:
        - grid_step_pct: 변동성↑/급락↑ -> 더 넓게(덜 자주 물타기)
        - take_profit_pct: 변동성↑ -> 살짝 ↑(스캘핑 과열 방지), 변동성↓ -> 살짝 ↓(체결 빈도↑)
        - tp_refresh_sec: 변동성↑ -> ↑ (갱신 덜)
        - recenter_threshold_pct: 급락/급등 -> ↓ (base를 더 빨리 따라가게)
        """
        now = s.ts
        interval = int(getattr(self.cfg, "ai_tune_interval_sec", 600))  # 10분 기본
        if now - self._last_tune_ts < interval:
            # 쿨다운 중이면 현재값 유지
            return TuneDecision(
                grid_step_pct=float(current["grid_step_pct"]),
                take_profit_pct=float(current["take_profit_pct"]),
                tp_refresh_sec=int(current["tp_refresh_sec"]),
                recenter_threshold_pct=float(current["recenter_threshold_pct"]),
                notes="tune cooldown",
            )

        self._last_tune_ts = now

        # 현재값
        g0 = float(current["grid_step_pct"])
        tp0 = float(current["take_profit_pct"])
        r0 = int(current["tp_refresh_sec"])
        rc0 = float(current["recenter_threshold_pct"])

        vol = s.vol
        drop = s.drop_from_base_pct

        # 목표값(규칙)
        # grid_step_pct
        g_target = g0
        if vol >= 0.012 or drop >= 0.03:
            g_target = g0 * 1.20
        elif vol > 0 and vol < 0.005 and drop < 0.01:
            g_target = g0 * 0.90

        # take_profit_pct
        tp_target = tp0
        if vol >= 0.012:
            tp_target = tp0 * 1.10
        elif vol > 0 and vol < 0.005:
            tp_target = tp0 * 0.95

        # tp_refresh_sec
        r_target = r0
        if vol >= 0.012 or drop >= 0.03:
            r_target = int(r0 * 1.5)
        elif vol > 0 and vol < 0.005:
            r_target = int(max(10, r0 * 0.9))

        # recenter_threshold_pct
        rc_target = rc0
        if drop >= 0.03:
            rc_target = rc0 * 0.85
        elif drop < 0.01 and vol < 0.006:
            rc_target = rc0 * 1.05

        # 하드 가드레일(네 전략 기본값 주변 “안전 범위”)
        g_lo, g_hi = 0.003, 0.030
        tp_lo, tp_hi = 0.003, 0.050
        r_lo, r_hi = 10, 300
        rc_lo, rc_hi = 0.005, 0.060

        # 스무딩(튀는 거 방지)
        g_new = self._smooth(g0, self._clamp(g_target, g_lo, g_hi), alpha=0.25)
        tp_new = self._smooth(tp0, self._clamp(tp_target, tp_lo, tp_hi), alpha=0.25)
        r_new = int(self._clamp(r_target, r_lo, r_hi))
        rc_new = self._smooth(rc0, self._clamp(rc_target, rc_lo, rc_hi), alpha=0.25)

        notes = f"tuned by vol={vol:.5f}, drop={drop*100:.2f}%"
        return TuneDecision(g_new, tp_new, r_new, rc_new, notes)

    # -------------------------
    # Reporting (LLM optional)
    # -------------------------
    def maybe_make_report(self, s: MarketState, gate: GateDecision, tune: TuneDecision) -> Optional[str]:
        """
        리포트는 비용/지연 때문에 자주 안 함. 기본 1시간.
        """
        if not getattr(self.cfg, "ai_enable_report", True):
            return None

        now = s.ts
        interval = int(getattr(self.cfg, "ai_report_interval_sec", 3600))
        if now - self._last_report_ts < interval:
            return None
        self._last_report_ts = now

        # LLM 없으면 간단 텍스트 리포트라도 남김
        if self._llm is None:
            return self._fallback_report(s, gate, tune)

        model = getattr(self.cfg, "ai_report_model", "gpt-4.1-mini")
        prompt = self._build_report_prompt(s, gate, tune)
        try:
            resp = self._llm.responses.create(
                model=model,
                input=prompt,
                temperature=0.2,
            )
            text = resp.output[0].content[0].text.strip()
            return text
        except Exception as e:
            return self._fallback_report(s, gate, tune) + f"\n(LLM error: {e})"

    def _fallback_report(self, s: MarketState, gate: GateDecision, tune: TuneDecision) -> str:
        return (
            f"[BTC AI REPORT]\n"
            f"- price={s.price:.2f}, base={s.base:.2f}, trigger={s.trigger:.2f}, drop={s.drop_from_base_pct*100:.2f}%\n"
            f"- vol={s.vol:.5f}, regime={gate.regime} (buy_mult={gate.buy_mult}, allow_buy={gate.allow_buy})\n"
            f"- cap: free={s.free_usdt:.2f}, cap30={s.trade_cap:.2f}, used={s.used:.2f}, remain={s.remaining_cap:.2f}\n"
            f"- pool: btc={s.pool_btc:.8f}, avg={s.pool_avg:.2f}, cost={s.pool_cost:.2f}\n"
            f"- tune: grid={tune.grid_step_pct:.4f}, tp={tune.take_profit_pct:.4f}, refresh={tune.tp_refresh_sec}s, recenter={tune.recenter_threshold_pct:.4f}\n"
        )

    def _build_report_prompt(self, s: MarketState, gate: GateDecision, tune: TuneDecision) -> str:
        payload = {
            "market": asdict(s),
            "gate": asdict(gate),
            "tune": asdict(tune),
        }
        return (
            "You are an on-call monitoring assistant for a BTC stacking/grid bot.\n"
            "Write a concise Korean report with:\n"
            "1) 한 줄 요약(상태/레짐/리스크)\n"
            "2) 지금 위험 신호(있으면) + 원인\n"
            "3) 다음 1시간 운영 제안(파라미터/주의사항). Do NOT tell to buy/sell; focus on bot ops.\n"
            "Use numbers from the JSON.\n\n"
            f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
