# btc_trader.py
from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Optional, List
from btc_ai import BTCAI, MarketState
from btc_client import BTCClient
from btc_config import BTCConfig
from btc_db_manager import BTCDB, PositionSnapshot


class BTCStackingTrader:
    """
    v2.1 핵심:
    1) TP를 시장가가 아니라 "리밋 주문"으로 유지/갱신 (avg_entry 변화/qty 변화 반영)
    2) buy_quote_usdt를 노출(exposure) + 단기 변동성(vol)에 따라 동적으로 조절
    + ✅ 추가: USDT 70/30 룰 (총액의 70%는 보유, 30% 한도 내에서만 매수/매도 반복)
    """

    def __init__(self, cfg: BTCConfig, client: BTCClient | None = None, db: BTCDB | None = None):
        self.cfg = cfg

        # ✅ 외부에서 주입되면 그대로 사용 (btc_run.py 호환)
        if db is not None:
            self.db = db
        else:
            self.db = BTCDB(db_url=cfg.build_db_url(), verbose=cfg.verbose)

        if client is not None:
            self.client = client
        else:
            self.client = BTCClient(cfg)

        self._last_order_ts = 0.0
        self._last_recenter_ts = 0.0
        self._last_tp_refresh_ts = 0.0
        self._last_cap_log_ts = 0.0
        self._price_hist: List[float] = []

        # base_price 초기화
        if self.db.get_setting("base_price") is None:
            p = self.client.get_price(cfg.symbol)
            self.db.set_setting("base_price", p)
            self.db.log("INFO", f"base_price init -> {p:.2f}")

        # ✅ 고정 70/30을 위한 "레퍼런스 USDT 총액" 저장 (한 번만)
        self._init_usdt_reference()
        self.ai = BTCAI(cfg, self.db, client=self.client)

        # ✅ AI 튜닝 적용값(런타임 오버라이드)
        self._rt_params = {
            "grid_step_pct": float(self.cfg.grid_step_pct),
            "take_profit_pct": float(self.cfg.take_profit_pct),
            "tp_refresh_sec": int(self.cfg.tp_refresh_sec),
            "recenter_threshold_pct": float(self.cfg.recenter_threshold_pct),
        }

    # -----------------------------
    # USDT 70/30 helpers
    # -----------------------------
    def _now(self) -> float:
        # 백테스트면 캔들 ts를 시간으로 사용
        if getattr(self.client, "is_backtest", False):
            return float(getattr(self.client, "_cur_ts", time.time()))
        return time.time()
    
    def _cfg(self, key: str, default):
        return getattr(self.cfg, key, default)

    def _init_usdt_reference(self):
        """
        고정 70/30 모드(use_fixed_usdt_reference=True)일 때,
        시작 시점 USDT 총액(total=free+locked)을 DB에 한 번 저장.
        """
        if not self._cfg("use_fixed_usdt_reference", True):
            return

        if self.db.get_setting("usdt_ref_total") is not None:
            return

        try:
            quote_asset = self._cfg("quote_asset", "USDT")
            bal = self.client.get_balance(quote_asset)  # {"free","locked","total"}
            ref_total = float(bal.get("total", 0.0) or 0.0)
            self.db.set_setting("usdt_ref_total", ref_total)
            self.db.log("INFO", f"usdt_ref_total init -> {ref_total:.2f} {quote_asset}")
        except Exception as e:
            # 레퍼런스 저장 실패해도 트레이더가 죽지 않도록
            self.db.log("WARN", f"usdt_ref_total init failed: {e}")
            
    def _p(self, key: str, default=None):
        if key in self._rt_params:
            return self._rt_params[key]
        return getattr(self.cfg, key, default)

    def _apply_tune(self, tune):
        # tune은 btc_ai.TuneDecision
        self._rt_params["grid_step_pct"] = float(tune.grid_step_pct)
        self._rt_params["take_profit_pct"] = float(tune.take_profit_pct)
        self._rt_params["tp_refresh_sec"] = int(tune.tp_refresh_sec)
        self._rt_params["recenter_threshold_pct"] = float(tune.recenter_threshold_pct)

    def _get_trade_cap_snapshot(self, pos: PositionSnapshot) -> dict:
        """
        70/30 룰 기준으로 현재 cap 상태 계산.
        - cap 기준 총액은 (고정) usdt_ref_total 또는 (동적) 현재 total_usdt
        - cap(30%) 내에서만 추가 매수 가능
        """
        quote_asset = self._cfg("quote_asset", "USDT")
        trade_cap_ratio = float(self._cfg("trade_cap_ratio", 0.30))
        reserve_buffer = float(self._cfg("usdt_reserve_buffer", 2.0))

        bal = self.client.get_balance(quote_asset)
        free_usdt = float(bal.get("free", 0.0) or 0.0)
        total_usdt_now = float(bal.get("total", 0.0) or 0.0)

        # 기준 총액: 고정/동적
        if self._cfg("use_fixed_usdt_reference", True):
            ref = self.db.get_setting("usdt_ref_total", None)
            try:
                ref_f = float(ref) if ref is not None and str(ref) != "" else 0.0
            except Exception:
                ref_f = 0.0

            # ✅ ref가 0이면(또는 비정상) 현재 total로 대체 + DB도 자동 복구
            if ref_f <= 0:
                base_total = total_usdt_now
                if base_total > 0:
                    try:
                        self.db.set_setting("usdt_ref_total", base_total)
                        self.db.log("INFO", f"usdt_ref_total auto-fix -> {base_total:.2f} {quote_asset}")
                    except Exception:
                        pass
            else:
                base_total = ref_f

            # ✅✅ 여기 추가: total이 base_total보다 “충분히” 커졌으면(=입금/외부유입 가능성)
            # usdt_ref_total을 올려서 30% 풀도 같이 커지게 함
            inc_threshold = float(self._cfg("usdt_ref_increase_threshold", 10.0))  # 기본 10 USDT
            if total_usdt_now > base_total + inc_threshold:
                try:
                    old = base_total
                    base_total = total_usdt_now
                    self.db.set_setting("usdt_ref_total", base_total)
                    self.db.log("INFO", f"usdt_ref_total increased -> {old:.2f} -> {base_total:.2f} {quote_asset}")
                except Exception as e:
                    self.db.log("WARN", f"usdt_ref_total increase failed: {e}")

        else:
            base_total = total_usdt_now

        trade_cap = max(0.0, base_total * trade_cap_ratio)
        used = max(0.0, float(pos.cost_basis_usdt))
        remaining_cap = max(0.0, trade_cap - used)
        spendable_free = max(0.0, free_usdt - reserve_buffer)

        return {
            "quote_asset": quote_asset,
            "free_usdt": free_usdt,
            "total_usdt_now": total_usdt_now,
            "base_total": base_total,
            "trade_cap": trade_cap,
            "used": used,
            "remaining_cap": remaining_cap,
            "spendable_free": spendable_free,
        }

    def log_cap_status(self, pos_pool: PositionSnapshot, pos_all: Optional[PositionSnapshot] = None):
        # 너무 자주 찍히지 않게(기본 10초) 제한
        interval = float(self._cfg("cap_log_interval_sec", 10))
        now = self._now()
        if now - self._last_cap_log_ts < interval:
            return
        self._last_cap_log_ts = now

        s = self._get_trade_cap_snapshot(pos_pool)
        extra = ""
        if pos_all is not None:
            extra = (
                f" | ALL btc={pos_all.btc_qty:.8f} avg={pos_all.avg_entry:.2f} "
                f"cost={pos_all.cost_basis_usdt:.2f}"
            )

        self.db.log(
            "INFO",
            f"[CAP] free={s['free_usdt']:.2f} total={s['total_usdt_now']:.2f} base_total={s['base_total']:.2f} "
            f"cap(30%)={s['trade_cap']:.2f} used={s['used']:.2f} remain={s['remaining_cap']:.2f} "
            f"| POOL btc={pos_pool.btc_qty:.8f} avg={pos_pool.avg_entry:.2f} cost={pos_pool.cost_basis_usdt:.2f}"
            f"{extra}",
        )

    def maybe_initial_market_entry(self):
        """
        실행 직후 1회성으로 (기본 70%)를 시장가 매수.
        - 중복 실행 방지: btc_settings.init_entry_done 사용
        - pool(30% 운용분)과 분리하고 싶으면 clientOrderId prefix를 BTCSTACK_가 아닌 값으로 둠
        """
        if not self._cfg("initial_buy_on_start", False):
            return

        # 이미 한 번 했으면 스킵
        if self.db.get_setting("init_entry_done", None) == "1":
            return

        quote_asset = self._cfg("quote_asset", "USDT")
        ratio = float(self._cfg("initial_buy_ratio", 0.70))
        reserve_buffer = float(self._cfg("usdt_reserve_buffer", 2.0))

        # 기준 총액: 고정 ref가 있으면 ref를, 없으면 현재 total
        bal = self.client.get_balance(quote_asset)
        free_usdt = float(bal.get("free", 0.0) or 0.0)
        total_usdt_now = float(bal.get("total", 0.0) or 0.0)

        base_total = total_usdt_now
        if self._cfg("use_fixed_usdt_reference", True):
            ref = self.db.get_setting("usdt_ref_total", None)
            try:
                ref_f = float(ref) if ref is not None and str(ref) != "" else 0.0
            except Exception:
                ref_f = 0.0
            if ref_f > 0:
                base_total = ref_f

        target_buy = max(0.0, base_total * ratio)

        # 실제로는 free에서만 지출 가능 + 버퍼 제외
        spendable = max(0.0, free_usdt - reserve_buffer)
        buy_usdt = min(target_buy, spendable)

        # 최소 주문금액 체크
        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, self.cfg.min_trade_usdt)
        if buy_usdt < min_notional:
            self.db.log("WARN", f"INIT BUY skipped: buy_usdt({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
            return

        # 너무 자주 주문 방지(실수 방지)
        if not self._cooldown_ok():
            return

        try:
            # ✅ pool과 분리: clientOrderId prefix를 BTCSTACK_가 아닌 값으로 둠
            # pool에 포함시키고 싶으면 client_oid="BTCSTACK_INIT_..." 로 바꿔도 됨
            client_oid = f"BTCINIT_{int(self._now()*1000)}"
            o = self.client.place_market_buy_by_quote(
                quote_usdt=buy_usdt,
                symbol=self.cfg.symbol,
                client_oid=client_oid,
            )
            self.db.upsert_order(o)
            self._touch_order()

            self.db.set_setting("init_entry_done", "1")
            self.db.log("INFO", f"INIT BUY done: quote={buy_usdt:.2f} ({ratio*100:.0f}% of base_total={base_total:.2f})")
        except Exception as e:
            self.db.log("ERROR", f"INIT BUY failed: {e}")

    # -----------------------------
    # Sync: orders + fills + position
    # -----------------------------
    def sync_orders_and_fills(self):
        open_ids = self.db.get_recent_open_orders(limit=50)

        try:
            ex_open = self.client.get_open_orders(self.cfg.symbol)
            for o in ex_open:
                self.db.upsert_order(o)
                oid = int(o["orderId"])
                if oid not in open_ids:
                    open_ids.append(oid)
        except Exception as e:
            self.db.log("WARN", f"get_open_orders failed: {e}")

        for oid in open_ids[:50]:
            try:
                o = self.client.get_order(order_id=oid, symbol=self.cfg.symbol)
                self.db.upsert_order(o)
            except Exception as e:
                self.db.log("WARN", f"get_order failed: order_id={oid} err={e}")

        try:
            trades = self.client.get_my_trades(self.cfg.symbol)
            seen_oids = set()

            for t in trades:
                oid = int(t.get("orderId", 0))
                if oid <= 0:
                    continue

                # ✅ order 먼저 확보 (clientOrderId를 btc_orders에 저장해야 pool join이 됨)
                if oid not in seen_oids:
                    seen_oids.add(oid)
                    try:
                        o = self.client.get_order(order_id=oid, symbol=self.cfg.symbol)
                        self.db.upsert_order(o)
                    except Exception as e:
                        self.db.log("WARN", f"get_order(for trade) failed: order_id={oid} err={e}")

                self.db.insert_fill(order_id=oid, t=t)
        except Exception as e:
            self.db.log("WARN", f"get_my_trades failed: {e}")

        snap = self.db.recompute_position_from_fills(symbol=self.cfg.symbol)
        if self.cfg.verbose:
            self.db.log("INFO", f"position -> {asdict(snap)}")

        self.db.recompute_pool_position_from_fills(symbol=self.cfg.symbol, client_prefix="BTCSTACK_")

    # -----------------------------
    # base_price
    # -----------------------------
    def get_base_price(self) -> float:
        v = self.db.get_setting("base_price", "0")
        try:
            return float(v)
        except Exception:
            return 0.0

    def set_base_price(self, price: float):
        self.db.set_setting("base_price", float(price))

    def maybe_recenter_base_price(self, cur_price: float):
        base = self.get_base_price()
        if base <= 0:
            self.set_base_price(cur_price)
            return

        now = self._now()
        if now - self._last_recenter_ts < self.cfg.recenter_cooldown_sec:
            return

        diff = abs(cur_price - base) / base
        if diff >= float(self._p("recenter_threshold_pct", self.cfg.recenter_threshold_pct)):
            self.set_base_price(cur_price)
            self._last_recenter_ts = now
            self.db.log("INFO", f"recenter base_price: {base:.2f} -> {cur_price:.2f} (diff={diff*100:.2f}%)")

    # -----------------------------
    # Cooldown
    # -----------------------------
    def _cooldown_ok(self) -> bool:
        return (self._now() - self._last_order_ts) >= self.cfg.order_cooldown_sec

    def _touch_order(self):
        self._last_order_ts = self._now()

    # -----------------------------
    # Dynamic buy sizing (v2.1)
    # -----------------------------
    def _update_price_history(self, cur_price: float):
        self._price_hist.append(float(cur_price))
        w = int(self.cfg.price_vol_window)
        if len(self._price_hist) > w:
            self._price_hist = self._price_hist[-w:]

    def _estimate_vol(self) -> float:
        """
        최근 price_vol_window 구간의 단순 로그수익률 표준편차(틱 기준).
        루프가 10초면 30틱=5분 정도.
        """
        if len(self._price_hist) < 10:
            return 0.0
        rets = []
        for i in range(1, len(self._price_hist)):
            p0 = self._price_hist[i - 1]
            p1 = self._price_hist[i]
            if p0 <= 0 or p1 <= 0:
                continue
            rets.append(math.log(p1 / p0))
        if len(rets) < 5:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, (len(rets) - 1))
        return math.sqrt(var)

    def compute_dynamic_buy_usdt(self, pos: PositionSnapshot) -> float:
        """
        buy_quote_usdt를:
          - 노출 비율(exposure_ratio)에 따라 감소
          - 단기 변동성(vol)에 따라 약간 증/감
        (⚠️ 여기서는 '계산만' 하고, 70/30 캡은 maybe_buy_on_drop에서 적용)
        """
        base_buy = float(self.cfg.buy_quote_usdt)

        # 1) 노출 기반 축소
        exposure = max(0.0, float(pos.cost_basis_usdt))
        cap = float(self.cfg.max_quote_exposure_usdt)
        exposure_ratio = min(1.0, exposure / cap) if cap > 0 else 0.0
        exposure_factor = (1.0 - exposure_ratio) ** float(self.cfg.exposure_power)

        # 2) 변동성 기반 보정
        vol = self._estimate_vol()
        if vol <= 0:
            vol_factor = 1.0
        elif vol < self.cfg.vol_low:
            vol_factor = float(self.cfg.vol_boost_max)
        elif vol > self.cfg.vol_high:
            vol_factor = float(self.cfg.vol_cut_min)
        else:
            t = (vol - self.cfg.vol_low) / (self.cfg.vol_high - self.cfg.vol_low + 1e-12)
            vol_factor = float(self.cfg.vol_boost_max) + (float(self.cfg.vol_cut_min) - float(self.cfg.vol_boost_max)) * t

        buy_usdt = base_buy * exposure_factor * vol_factor
        buy_usdt = max(float(self.cfg.buy_min_usdt), min(float(self.cfg.buy_max_usdt), buy_usdt))
        return float(buy_usdt)

    # -----------------------------
    # TP LIMIT manager (v2.1)
    # -----------------------------
    def _get_tp_order_id(self) -> Optional[int]:
        v = self.db.get_setting("tp_order_id", None)
        if not v:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _set_tp_order_id(self, oid: Optional[int]):
        if oid is None:
            self.db.set_setting("tp_order_id", "")
        else:
            self.db.set_setting("tp_order_id", int(oid))

    def _tp_target(self, pos: PositionSnapshot) -> tuple[float, float]:
        """
        리밋 TP 목표:
        - price = avg_entry * (1 + take_profit_pct) (tick 보정 + bump)
        - qty   = btc_qty * sell_fraction_on_tp (step 보정)
        """
        tp_price = float(pos.avg_entry) * (1.0 + float(self._p("take_profit_pct", self.cfg.take_profit_pct)))
        tp_price = self.client.adjust_price(tp_price, self.cfg.symbol)

        # tick 단위로 살짝 올리고 싶으면
        if self.cfg.tp_price_bump_ticks > 0:
            f = self.client.get_symbol_filters(self.cfg.symbol)
            tp_price = tp_price + float(self.cfg.tp_price_bump_ticks) * f.tick_size
            tp_price = self.client.adjust_price(tp_price, self.cfg.symbol)

        qty = float(pos.btc_qty) * float(self.cfg.sell_fraction_on_tp)
        qty = self.client.adjust_qty(qty, self.cfg.symbol)
        return tp_price, qty

    def ensure_tp_limit_order(self, cur_price: float, pos: PositionSnapshot, gate=None):
        """
        TP 리밋 주문을 "항상 1개" 유지하는 방식.
        - 포지션이 없거나 avg_entry가 없으면 TP 주문 제거
        - 목표 TP 가격/수량이 바뀌면 기존 TP 주문 취소 후 새로 발행
        - 너무 자주 갱신되지 않게 tp_refresh_sec 적용
        """
        if not self.cfg.use_tp_limit_orders:
            return

        # ✅ gate가 TP refresh 자체를 막으면 중단
        if gate is not None and getattr(gate, "allow_tp_refresh", True) is False:
            return

        now = self._now()

        # ✅ refresh_sec에 gate.tp_refresh_mult 반영
        refresh_sec = int(self._p("tp_refresh_sec", self.cfg.tp_refresh_sec))
        if gate is not None:
            try:
                refresh_sec = int(refresh_sec * float(getattr(gate, "tp_refresh_mult", 1.0)))
            except Exception:
                pass

        if now - self._last_tp_refresh_ts < refresh_sec:
            return

        self._last_tp_refresh_ts = now

        tp_oid = self._get_tp_order_id()

        # 포지션 없으면 TP 주문도 없애기
        if pos.btc_qty <= 1e-12 or pos.avg_entry <= 0:
            if tp_oid is not None:
                try:
                    self.client.cancel_order(tp_oid, self.cfg.symbol)
                except Exception:
                    pass
                self._set_tp_order_id(None)
            return

        tp_price, tp_qty = self._tp_target(pos)
        if tp_qty <= 0:
            return

        # 최소 notional 확인
        notional = tp_qty * tp_price
        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, self.cfg.min_trade_usdt)
        if notional < min_notional:
            if tp_oid is not None:
                try:
                    self.client.cancel_order(tp_oid, self.cfg.symbol)
                except Exception:
                    pass
                self._set_tp_order_id(None)
            return

        # 기존 TP 주문 확인
        if tp_oid is not None:
            try:
                o = self.client.get_order(tp_oid, self.cfg.symbol)
                self.db.upsert_order(o)
                status = o.get("status")
                if status in ("NEW", "PARTIALLY_FILLED"):
                    old_price = float(o.get("price") or 0.0)
                    old_qty = float(o.get("origQty") or 0.0)

                    f = self.client.get_symbol_filters(self.cfg.symbol)
                    same_price = abs(old_price - tp_price) <= (f.tick_size * 0.5)
                    same_qty = abs(old_qty - tp_qty) <= (f.step_size * 0.5)
                    if same_price and same_qty:
                        return

                    try:
                        self.client.cancel_order(tp_oid, self.cfg.symbol)
                    except Exception:
                        return
                    self._set_tp_order_id(None)
                else:
                    self._set_tp_order_id(None)
            except Exception:
                return

        # TP 주문 새로 발행
        if not self._cooldown_ok():
            return

        try:
            o = self.client.place_limit_sell(qty_base=tp_qty, price=tp_price, symbol=self.cfg.symbol)
            self.db.upsert_order(o)
            self._set_tp_order_id(int(o["orderId"]))
            self._touch_order()
            self.db.log("INFO", f"TP LIMIT set: price={tp_price:.2f} qty={tp_qty:.8f} (avg={pos.avg_entry:.2f})")
        except Exception as e:
            self.db.log("ERROR", f"TP LIMIT place failed: {e}")

    # -----------------------------
    # Buy on drop (v2.1 + 70/30 cap)
    # -----------------------------
    def maybe_buy_on_drop(self, cur_price: float, pos: PositionSnapshot, gate: Optional[object] = None):
        if not self._cooldown_ok():
            return

        base = self.get_base_price()
        trigger = base * (1.0 - float(self._p("grid_step_pct", self.cfg.grid_step_pct)))

        # (기존 안전상한) 이 값이 30% cap보다 작으면 이게 먼저 막음
        if pos.cost_basis_usdt >= self.cfg.max_quote_exposure_usdt:
            return

        if cur_price <= trigger:
            filters = self.client.get_symbol_filters(self.cfg.symbol)
            min_notional = max(filters.min_notional, self.cfg.min_trade_usdt)

            # 1) 기본 buy_usdt 계산 (동적/고정)
            buy_usdt = self.compute_dynamic_buy_usdt(pos) if self.cfg.dynamic_buy else float(self.cfg.buy_quote_usdt)

            # 2) ✅ 70/30 룰 캡 적용: 30% 한도 + free 한도
            s = None
            try:
                s = self._get_trade_cap_snapshot(pos)
                buy_usdt = min(float(buy_usdt), float(s["remaining_cap"]), float(s["spendable_free"]))

                if buy_usdt <= 0:
                    self.db.log(
                        "INFO",
                        f"BUY skipped (cap). free={s['free_usdt']:.2f} total={s['total_usdt_now']:.2f} "
                        f"cap={s['trade_cap']:.2f} used={s['used']:.2f} remain={s['remaining_cap']:.2f}"
                    )
                    return
            except Exception as e:
                self.db.log("WARN", f"cap calc failed -> BUY blocked for safety: {e}")
                return  # ✅ cap 계산 못하면 매수 금지(30% 풀 보호)


            # ✅ gate는 step()에서 1회 계산된 값을 주입받는 방식
            if gate is not None:
                if not getattr(gate, "allow_buy", True):
                    self.db.log("INFO", f"BUY blocked by AI gate: regime={gate.regime} notes={gate.notes}")
                    return

                try:
                    buy_usdt = float(buy_usdt) * float(getattr(gate, "buy_mult", 1.0))
                except Exception:
                    pass

                # ✅ 다시 cap 리밋 (30% 풀 유지)
                buy_usdt = min(float(buy_usdt), float(s["remaining_cap"]), float(s["spendable_free"]))
                if buy_usdt <= 0:
                    self.db.log(
                        "INFO",
                        f"BUY skipped (cap after gate). remain={s['remaining_cap']:.2f} free={s['free_usdt']:.2f}"
                    )
                    return


            # 3) 최소 주문금액 체크
            if buy_usdt < min_notional:
                self.db.log("WARN", f"buy_usdt too small ({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
                return

            # 4) 시장가 매수
            try:
                o = self.client.place_market_buy_by_quote(quote_usdt=buy_usdt, symbol=self.cfg.symbol)
                self.db.upsert_order(o)
                self._touch_order()
                self.db.log("INFO", f"BUY market quote={buy_usdt:.2f} | base={base:.2f} trig={trigger:.2f} cur={cur_price:.2f}")
                # 매수 후 base_price를 현재가로 내려서 바닥을 따라가게
                self.set_base_price(cur_price)
            except Exception as e:
                self.db.log("ERROR", f"BUY failed: {e}")

    # -----------------------------
    # Main loop
    # -----------------------------
    def step(self):
        self.sync_orders_and_fills()

        cur_price = self.client.get_price(self.cfg.symbol)
        self._update_price_history(cur_price)

        pos_all = self.db.get_position(self.cfg.symbol)                 # 전체(참고용)
        pos_pool = self.db.get_pool_position(self.cfg.symbol)           # ✅ 30% 풀(핵심)

        # cap snapshot(POOL 기준으로)
        cap = self._get_trade_cap_snapshot(pos_pool)

        base = self.get_base_price()
        trigger = base * (1.0 - float(self._p("grid_step_pct", self.cfg.grid_step_pct)))
        drop_pct = 0.0
        if base > 0:
            drop_pct = max(0.0, (base - cur_price) / base)

        vol = self._estimate_vol()

        tp_oid = self._get_tp_order_id()

        state = MarketState(
            ts=self._now(),
            symbol=self.cfg.symbol,
            price=float(cur_price),
            base=float(base),
            trigger=float(trigger),
            drop_from_base_pct=float(drop_pct),
            vol=float(vol),
            free_usdt=float(cap["free_usdt"]),
            total_usdt=float(cap["total_usdt_now"]),
            base_total=float(cap["base_total"]),
            trade_cap=float(cap["trade_cap"]),
            used=float(cap["used"]),
            remaining_cap=float(cap["remaining_cap"]),
            spendable_free=float(cap["spendable_free"]),
            pool_btc=float(pos_pool.btc_qty),
            pool_avg=float(pos_pool.avg_entry),
            pool_cost=float(pos_pool.cost_basis_usdt),
            all_btc=float(pos_all.btc_qty),
            all_avg=float(pos_all.avg_entry),
            all_cost=float(pos_all.cost_basis_usdt),
            tp_order_id=int(tp_oid) if tp_oid is not None else None,
        )

        gate = self.ai.decide_gate(state)
        self.db.insert_ai_event("GATE", self.cfg.symbol, asdict(gate))

        cur_params = dict(self._rt_params)
        tune = self.ai.decide_tuning(state, cur_params)
        self._apply_tune(tune)
        self.db.insert_ai_event("TUNE", self.cfg.symbol, asdict(tune))

        report = self.ai.maybe_make_report(state, gate, tune)
        if report:
            self.db.insert_ai_event("REPORT", self.cfg.symbol, {"text": report})
            self.db.log("INFO", report)

        self.log_cap_status(pos_pool)

        self.maybe_recenter_base_price(cur_price)

        # ✅ gate 전달
        self.ensure_tp_limit_order(cur_price, pos_pool, gate=gate)

        # ✅ gate 전달
        self.maybe_buy_on_drop(cur_price, pos_pool, gate=gate)

    def run_forever(self):
        self.db.log("INFO", "BTCStackingTrader start")

        self.maybe_initial_market_entry()
        
        while True:
            try:
                self.step()
                
            except Exception as e:
                self.db.log("ERROR", f"loop error: {e}")
            time.sleep(self.cfg.loop_interval_sec)


if __name__ == "__main__":
    cfg = BTCConfig()
    trader = BTCStackingTrader(cfg)
    trader.run_forever()
