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
    1) TP를 "리밋 주문"으로 유지/갱신 (avg_entry 변화/qty 변화 반영)
    2) buy_quote_usdt를 노출(exposure) + 단기 변동성(vol)에 따라 동적으로 조절

    ✅ 이번 수정:
    - (백테스트/실매매 공통) 초기 1회 매수에서 "금액(initial_buy_usdt)" 우선 지원
    - (백테스트/실매매 공통) TP 주문 수량이 전체 BTC의 일정 비율 아래로 내려가지 않도록 캡(기본 70% 보유)
    - pool prefix(BTCSTACK_) 하드코딩 제거 (cfg.client_order_prefix 반영)
    - clientOrderId / client_oid 파라미터 호환(실거래 클라이언트와 백테스트 클라이언트 차이 방지)
    """

    def __init__(self, cfg: BTCConfig, client: BTCClient | None = None, db: BTCDB | None = None):
        self.cfg = cfg

        # 외부 주입
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
        self._crash_active = False
        self._price_hist: List[float] = []

        # base_price 초기화
        if self.db.get_setting("base_price") is None:
            p = self.client.get_price(cfg.symbol)
            self.db.set_setting("base_price", p)
            self.db.log("INFO", f"base_price init -> {p:.2f}")

        # 고정 레퍼런스
        self._init_usdt_reference()
        self.ai = BTCAI(cfg, self.db, client=self.client)

        # 런타임 오버라이드(튜닝)
        self._rt_params = {
            "grid_step_pct": float(self.cfg.grid_step_pct),
            "take_profit_pct": float(self.cfg.take_profit_pct),
            "tp_refresh_sec": int(self.cfg.tp_refresh_sec),
            "recenter_threshold_pct": float(self.cfg.recenter_threshold_pct),
        }

    # -----------------------------
    # helpers
    # -----------------------------
    def _now(self) -> float:
        if getattr(self.client, "is_backtest", False):
            return float(getattr(self.client, "_cur_ts", time.time()))
        return time.time()

    def _cfg(self, key: str, default):
        return getattr(self.cfg, key, default)

    def _p(self, key: str, default=None):
        if key in self._rt_params:
            return self._rt_params[key]
        return getattr(self.cfg, key, default)

    def _apply_tune(self, tune):
        self._rt_params["grid_step_pct"] = float(tune.grid_step_pct)
        self._rt_params["take_profit_pct"] = float(tune.take_profit_pct)
        self._rt_params["tp_refresh_sec"] = int(tune.tp_refresh_sec)
        self._rt_params["recenter_threshold_pct"] = float(tune.recenter_threshold_pct)

    def _crash_adjustments(self, base: float, cur_price: float, vol: float) -> float:
        if base <= 0:
            return 1.0

        drop_pct = max(0.0, (base - cur_price) / base)
        drop_thr = float(self._cfg("crash_drop_pct", 0.0))
        vol_thr = float(self._cfg("crash_vol_threshold", 0.0))

        crash_active = drop_pct >= drop_thr and vol >= vol_thr
        if crash_active and not self._crash_active:
            self._crash_active = True
            self.db.log(
                "INFO",
                (
                    "CRASH mode ON: widen grid "
                    f"(drop={drop_pct*100:.2f}% >= {drop_thr*100:.2f}%, "
                    f"vol={vol:.4f} >= {vol_thr:.4f})"
                ),
            )
        elif not crash_active and self._crash_active:
            self._crash_active = False
            self.db.log(
                "INFO",
                (
                    "CRASH mode OFF: grid back to normal "
                    f"(drop={drop_pct*100:.2f}%, vol={vol:.4f})"
                ),
            )

        if crash_active:
            grid_mult = float(self._cfg("crash_grid_mult", 1.0))
            return max(1.0, grid_mult)

        return 1.0

    # -----------------------------
    # USDT reference (optional)
    # -----------------------------
    def _init_usdt_reference(self):
        if not self._cfg("use_fixed_usdt_reference", True):
            return
        if self.db.get_setting("usdt_ref_total") is not None:
            return

        try:
            quote_asset = self._cfg("quote_asset", "USDT")
            bal = self.client.get_balance(quote_asset)
            ref_total = float(bal.get("total", 0.0) or 0.0)
            self.db.set_setting("usdt_ref_total", ref_total)
            self.db.log("INFO", f"usdt_ref_total init -> {ref_total:.2f} {quote_asset}")
        except Exception as e:
            self.db.log("WARN", f"usdt_ref_total init failed: {e}")

    # -----------------------------
    # cap snapshot (used in buy sizing)
    # -----------------------------
    def _get_trade_cap_snapshot(self, pos: PositionSnapshot) -> dict:
        quote_asset = self._cfg("quote_asset", "USDT")
        trade_cap_ratio = float(self._cfg("trade_cap_ratio", 0.30))
        reserve_buffer = float(self._cfg("usdt_reserve_buffer", 2.0))

        bal = self.client.get_balance(quote_asset)
        free_usdt = float(bal.get("free", 0.0) or 0.0)
        total_usdt_now = float(bal.get("total", 0.0) or 0.0)

        # 기준 총액
        if self._cfg("use_fixed_usdt_reference", True):
            ref = self.db.get_setting("usdt_ref_total", None)
            try:
                ref_f = float(ref) if ref is not None and str(ref) != "" else 0.0
            except Exception:
                ref_f = 0.0

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

    # -----------------------------
    # initial buy
    # -----------------------------
    def maybe_initial_market_entry(self):
        """
        백테스트/실매매 공통:
        - 초기 1회 매수 실행
        - (백테스트 의도) initial_buy_usdt/initial_entry_usdt가 있으면 '고정 금액'으로 매수
        - 초기 매수 체결 수량의 initial_reserve_ratio 만큼은 reserve_btc_qty로 저장 (영구 홀딩)
        """
        if not self._cfg("initial_buy_on_start", False):
            return

        if self.db.get_setting("init_entry_done", None) == "1":
            return

        quote_asset = self._cfg("quote_asset", "USDT")
        reserve_buffer = float(self._cfg("usdt_reserve_buffer", 0.0))

        # ✅ 1) 초기 매수 금액 우선(백테스트 의도)
        fixed_usdt = getattr(self.cfg, "initial_buy_usdt", None)
        if fixed_usdt is None:
            fixed_usdt = getattr(self.cfg, "initial_entry_usdt", None)

        bal = self.client.get_balance(quote_asset)
        free_usdt = float(bal.get("free", 0.0) or 0.0)

        if fixed_usdt is not None:
            buy_usdt = float(fixed_usdt)
        else:
            # (기존 방식) ratio 기반
            ratio = float(self._cfg("initial_buy_ratio", 0.70))
            total_usdt_now = float(bal.get("total", 0.0) or 0.0)
            buy_usdt = max(0.0, total_usdt_now * ratio)

        # free 기반(버퍼 제외)
        spendable = max(0.0, free_usdt - reserve_buffer)
        buy_usdt = min(buy_usdt, spendable)

        # 최소 주문금액 체크
        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(float(filters.min_notional), float(self._cfg("min_trade_usdt", 0.0)))
        if buy_usdt < min_notional:
            self.db.log("WARN", f"INIT BUY skipped: buy_usdt({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
            return

        if not self._cooldown_ok():
            return

        try:
            # ✅ 초기 매수도 POOL에 포함시키기 위해 prefix 강제
            prefix = str(getattr(self.cfg, "client_order_prefix", "BTCSTACK_"))
            client_oid = f"{prefix}INIT_{int(self._now()*1000)}"

            o = self.client.place_market_buy_by_quote(
                quote_usdt=buy_usdt,
                symbol=self.cfg.symbol,
                clientOrderId=client_oid,
            )
            self.db.upsert_order(o)
            self._touch_order()

            # ✅ reserve 저장(처음 1회만)
            if self.db.get_setting(self._reserve_key(), None) in (None, "", 0, 0.0):
                reserve_ratio = float(getattr(self.cfg, "initial_reserve_ratio",
                                              getattr(self.cfg, "keep_btc_ratio_on_tp", 0.70)))
                try:
                    bought_qty = float(o.get("executedQty") or o.get("origQty") or 0.0)
                except Exception:
                    bought_qty = 0.0

                reserve_qty = max(0.0, bought_qty * reserve_ratio)
                reserve_qty = self.client.adjust_qty(reserve_qty, self.cfg.symbol)
                self._set_reserve_btc_qty(reserve_qty)

                self.db.log(
                    "INFO",
                    f"RESERVE set: reserve_btc_qty={reserve_qty:.8f} "
                    f"(ratio={reserve_ratio:.2f} of initial buy qty={bought_qty:.8f})"
                )

            self.db.set_setting("init_entry_done", "1")
            self.db.log("INFO", f"INIT BUY done: quote={buy_usdt:.2f} {quote_asset}")

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

        # ✅ pool prefix는 cfg에서
        pool_prefix = str(self._cfg("client_order_prefix", "BTCSTACK_"))
        self.db.recompute_pool_position_from_fills(symbol=self.cfg.symbol, client_prefix=pool_prefix)

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
        if now - self._last_recenter_ts < int(self._cfg("recenter_cooldown_sec", 60)):
            return

        diff = abs(cur_price - base) / base
        if diff >= float(self._p("recenter_threshold_pct", float(self._cfg("recenter_threshold_pct", 0.020)))):
            self.set_base_price(cur_price)
            self._last_recenter_ts = now
            self.db.log("INFO", f"recenter base_price: {base:.2f} -> {cur_price:.2f} (diff={diff*100:.2f}%)")

    # -----------------------------
    # Cooldown
    # -----------------------------
    def _cooldown_ok(self) -> bool:
        return (self._now() - self._last_order_ts) >= int(self._cfg("order_cooldown_sec", 7))

    def _touch_order(self):
        self._last_order_ts = self._now()

    # -----------------------------
    # Dynamic buy sizing
    # -----------------------------
    def _update_price_history(self, cur_price: float):
        self._price_hist.append(float(cur_price))
        w = int(self._cfg("price_vol_window", 30))
        if len(self._price_hist) > w:
            self._price_hist = self._price_hist[-w:]

    def _estimate_vol(self) -> float:
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
        base_buy = float(self._cfg("buy_quote_usdt", 25.0))

        exposure = max(0.0, float(pos.cost_basis_usdt))
        cap = float(self._cfg("max_quote_exposure_usdt", 500.0))
        exposure_ratio = min(1.0, exposure / cap) if cap > 0 else 0.0
        exposure_factor = (1.0 - exposure_ratio) ** float(self._cfg("exposure_power", 1.6))

        vol = self._estimate_vol()
        vol_low = float(self._cfg("vol_low", 0.005))
        vol_high = float(self._cfg("vol_high", 0.012))
        vol_boost_max = float(self._cfg("vol_boost_max", 1.25))
        vol_cut_min = float(self._cfg("vol_cut_min", 0.70))

        if vol <= 0:
            vol_factor = 1.0
        elif vol < vol_low:
            vol_factor = vol_boost_max
        elif vol > vol_high:
            vol_factor = vol_cut_min
        else:
            t = (vol - vol_low) / (vol_high - vol_low + 1e-12)
            vol_factor = vol_boost_max + (vol_cut_min - vol_boost_max) * t

        buy_usdt = base_buy * exposure_factor * vol_factor
        buy_min = float(self._cfg("buy_min_usdt", 10.0))
        buy_max = float(self._cfg("buy_max_usdt", 60.0))
        buy_usdt = max(buy_min, min(buy_max, buy_usdt))
        return float(buy_usdt)

    # -----------------------------
    # TP LIMIT manager
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

    def _tp_target(self, pos: PositionSnapshot, pos_all: Optional[PositionSnapshot] = None) -> tuple[float, float]:
        """
        리밋 TP 목표:
        - price = avg_entry * (1 + take_profit_pct)
        - qty   = (sellable_btc) * sell_fraction_on_tp
          sellable_btc = max(0, pos.btc_qty - reserve_btc_qty)
        """
        tp_price = float(pos.avg_entry) * (1.0 + float(self._p("take_profit_pct", self.cfg.take_profit_pct)))
        tp_price = self.client.adjust_price(tp_price, self.cfg.symbol)

        # tick bump
        if getattr(self.cfg, "tp_price_bump_ticks", 0) > 0:
            f = self.client.get_symbol_filters(self.cfg.symbol)
            tp_price = tp_price + float(self.cfg.tp_price_bump_ticks) * f.tick_size
            tp_price = self.client.adjust_price(tp_price, self.cfg.symbol)

        reserve_qty = self._get_reserve_btc_qty()
        sellable = max(0.0, float(pos.btc_qty) - float(reserve_qty))

        qty = sellable * float(self.cfg.sell_fraction_on_tp)
        qty = self.client.adjust_qty(qty, self.cfg.symbol)
        return tp_price, qty

    def ensure_tp_limit_order(self, cur_price: float, pos_pool: PositionSnapshot, pos_all: Optional[PositionSnapshot] = None, gate=None):
        if not bool(self._cfg("use_tp_limit_orders", True)):
            return

        if gate is not None and getattr(gate, "allow_tp_refresh", True) is False:
            return

        now = self._now()
        refresh_sec = int(self._p("tp_refresh_sec", int(self._cfg("tp_refresh_sec", 30))))
        if gate is not None:
            try:
                refresh_sec = int(refresh_sec * float(getattr(gate, "tp_refresh_mult", 1.0)))
            except Exception:
                pass

        if now - self._last_tp_refresh_ts < refresh_sec:
            return
        self._last_tp_refresh_ts = now

        tp_oid = self._get_tp_order_id()

        if pos_pool.btc_qty <= 1e-12 or pos_pool.avg_entry <= 0:
            if tp_oid is not None:
                try:
                    self.client.cancel_order(tp_oid, self.cfg.symbol)
                except Exception:
                    pass
                self._set_tp_order_id(None)
            return

        tp_price, tp_qty = self._tp_target(pos_pool, pos_all=pos_all)
        if tp_qty <= 0:
            # ✅ 팔 수 있는 물량이 없으면(TP 가능한 sellable=0) TP 주문 제거
            tp_oid = self._get_tp_order_id()
            if tp_oid is not None:
                try:
                    self.client.cancel_order(tp_oid, self.cfg.symbol)
                except Exception:
                    pass
                self._set_tp_order_id(None)
            return

        notional = tp_qty * tp_price
        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, float(self._cfg("min_trade_usdt", 0.0)))
        if notional < min_notional:
            if tp_oid is not None:
                try:
                    self.client.cancel_order(tp_oid, self.cfg.symbol)
                except Exception:
                    pass
                self._set_tp_order_id(None)
            return

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

        if not self._cooldown_ok():
            return

        try:
            o = self.client.place_limit_sell(qty_base=tp_qty, price=tp_price, symbol=self.cfg.symbol)
            self.db.upsert_order(o)
            self._set_tp_order_id(int(o["orderId"]))
            self._touch_order()
            self.db.log("INFO", f"TP LIMIT set: price={tp_price:.2f} qty={tp_qty:.8f} (avg={pos_pool.avg_entry:.2f})")
        except Exception as e:
            self.db.log("ERROR", f"TP LIMIT place failed: {e}")

    # -----------------------------
    # Buy on drop
    # -----------------------------
    def maybe_buy_on_drop(self, cur_price: float, pos_pool: PositionSnapshot, gate: Optional[object] = None):
        if not self._cooldown_ok():
            return

        base = self.get_base_price()
        vol = self._estimate_vol()
        grid_mult = self._crash_adjustments(base, cur_price, vol)
        trigger = base * (1.0 - (float(self._p("grid_step_pct", float(self._cfg("grid_step_pct", 0.008)))) * grid_mult))

        if cur_price > trigger:
            return

        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, float(self._cfg("min_trade_usdt", 0.0)))

        # ✅ AI gate
        if gate is not None and not getattr(gate, "allow_buy", True):
            self.db.log("INFO", f"BUY blocked by AI gate: regime={gate.regime} notes={gate.notes}")
            return

        # =========================
        # ✅ 고정매수 모드 (25USDT 고정, 쪼개지 않음)
        # =========================
        fixed_grid = bool(getattr(self.cfg, "fixed_grid_buy", False))
        if fixed_grid:
            buy_usdt = float(getattr(self.cfg, "grid_buy_usdt", 25.0))

            if buy_usdt < min_notional:
                self.db.log("WARN", f"fixed buy_usdt({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
                return

            quote_asset = getattr(self.cfg, "quote_asset", "USDT")
            bal = self.client.get_balance(quote_asset)
            free_usdt = float(bal.get("free", 0.0) or 0.0)

            reserve_buffer = float(getattr(self.cfg, "usdt_reserve_buffer", 0.0))
            spendable = max(0.0, free_usdt - reserve_buffer)

            # ✅ free 부족하면 "쪼개지 말고" 그냥 스킵
            if spendable + 1e-12 < buy_usdt:
                return

            try:
                o = self.client.place_market_buy_by_quote(quote_usdt=buy_usdt, symbol=self.cfg.symbol)
                self.db.upsert_order(o)
                self._touch_order()
                self.db.log("INFO", f"BUY(FIXED) quote={buy_usdt:.2f} | base={base:.2f} trig={trigger:.2f} cur={cur_price:.2f}")
                self.set_base_price(cur_price)
            except Exception as e:
                self.db.log("ERROR", f"BUY failed: {e}")
            return

        # =========================
        # (고정매수 아닌 경우) 기존 로직 유지
        # =========================
        dynamic_buy = bool(self._cfg("dynamic_buy", True))
        buy_usdt = self.compute_dynamic_buy_usdt(pos_pool) if dynamic_buy else float(self._cfg("buy_quote_usdt", 25.0))

        if buy_usdt < min_notional:
            self.db.log("WARN", f"buy_usdt too small ({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
            return

        try:
            o = self.client.place_market_buy_by_quote(quote_usdt=buy_usdt, symbol=self.cfg.symbol)
            self.db.upsert_order(o)
            self._touch_order()
            self.db.log("INFO", f"BUY quote={buy_usdt:.2f} | base={base:.2f} trig={trigger:.2f} cur={cur_price:.2f}")
            self.set_base_price(cur_price)
        except Exception as e:
            self.db.log("ERROR", f"BUY failed: {e}")

    # -----------------------------
    # Reserve (initial 70% hold) helpers
    # -----------------------------
    def _reserve_key(self) -> str:
        return str(getattr(self.cfg, "reserve_btc_key", "reserve_btc_qty"))

    def _get_reserve_btc_qty(self) -> float:
        v = self.db.get_setting(self._reserve_key(), 0.0)
        try:
            return float(v) if v is not None and str(v) != "" else 0.0
        except Exception:
            return 0.0

    def _set_reserve_btc_qty(self, qty: float):
        try:
            qty = float(qty)
        except Exception:
            qty = 0.0
        self.db.set_setting(self._reserve_key(), qty)
    # -----------------------------
    # Main loop
    # -----------------------------
    def step(self):
        self.sync_orders_and_fills()

        cur_price = self.client.get_price(self.cfg.symbol)
        self._update_price_history(cur_price)

        pos_all = self.db.get_position(self.cfg.symbol)
        pos_pool = self.db.get_pool_position(self.cfg.symbol)

        cap = self._get_trade_cap_snapshot(pos_pool)

        base = self.get_base_price()
        vol = self._estimate_vol()
        grid_mult = self._crash_adjustments(base, cur_price, vol)
        trigger = base * (1.0 - (float(self._p("grid_step_pct", float(self._cfg("grid_step_pct", 0.008)))) * grid_mult))
        drop_pct = 0.0
        if base > 0:
            drop_pct = max(0.0, (base - cur_price) / base)

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

        gate = self.ai.decide_gate(state) if self.cfg.ai_enable_gate else None
        if gate:
            self.db.insert_ai_event("GATE", self.cfg.symbol, asdict(gate))

        cur_params = dict(self._rt_params)
        tune = self.ai.decide_tuning(state, cur_params) if self.cfg.ai_enable_tune else None
        if tune:
            self._apply_tune(tune)
            self.db.insert_ai_event("TUNE", self.cfg.symbol, asdict(tune))

        report = self.ai.maybe_make_report(state, gate, tune) if self.cfg.ai_enable_report else None
        self.maybe_recenter_base_price(cur_price)

        # ✅ 전체 포지션(pos_all)을 같이 넘겨야 keep_btc_ratio_on_tp 캡이 정확히 먹음
        self.ensure_tp_limit_order(cur_price, pos_pool, pos_all=pos_all, gate=gate)
        self.maybe_buy_on_drop(cur_price, pos_pool, gate=gate)

    def run_forever(self):
        self.db.log("INFO", "BTCStackingTrader start")
        self.maybe_initial_market_entry()

        while True:
            try:
                self.step()
            except Exception as e:
                self.db.log("ERROR", f"loop error: {e}")
            time.sleep(int(self._cfg("loop_interval_sec", 60)))


if __name__ == "__main__":
    cfg = BTCConfig()
    trader = BTCStackingTrader(cfg)
    trader.run_forever()
