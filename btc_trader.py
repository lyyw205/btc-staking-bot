# btc_trader.py
from __future__ import annotations

import time
from btc_client import BTCClient
from btc_config import BTCConfig
from btc_db_manager import BTCDB, PositionSnapshot


class BTCStackingTrader:
    """
    BTC lot-stacking core logic.
    - initial core buy (never sell)
    - buy lots on drop from base price
    - take profit per lot and compound net profit into core
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

        # base_price 초기화
        if self.db.get_setting("base_price") is None:
            p = self.client.get_price(cfg.symbol)
            self.db.set_setting("base_price", p)
            self.db.log("INFO", f"base_price init -> {p:.2f}")

        # 고정 레퍼런스
        self._init_usdt_reference()
        self._tune_fields = {
            "lot_drop_pct": float,
            "lot_tp_pct": float,
            "lot_prebuy_pct": float,
            "lot_cancel_rebound_pct": float,
            "lot_buy_usdt": float,
            "verbose": bool,
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
        if key in self._tune_fields:
            raw = self.db.get_setting(f"tune.{key}", None)
            if raw is not None and str(raw).strip() != "":
                caster = self._tune_fields[key]
                try:
                    if caster is bool:
                        val = str(raw).strip().lower() in ("1", "true", "yes", "on")
                    else:
                        val = caster(raw)
                    if key == "verbose":
                        self.db.verbose = bool(val)
                    return val
                except Exception:
                    pass
        return getattr(self.cfg, key, default)

    def _get_float_setting(self, key: str, default: float = 0.0) -> float:
        raw = self.db.get_setting(key, None)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _place_limit_buy_by_quote(
        self,
        *,
        quote_usdt: float,
        price: float,
        symbol: str,
        client_oid: str | None = None,
    ):
        try:
            return self.client.place_limit_buy_by_quote(
                quote_usdt=quote_usdt,
                price=price,
                symbol=symbol,
                client_oid=client_oid,
            )
        except TypeError:
            return self.client.place_limit_buy_by_quote(
                quote_usdt=quote_usdt,
                price=price,
                symbol=symbol,
                clientOrderId=client_oid,
            )

    def _place_limit_sell(self, *, qty: float, price: float, symbol: str, client_oid: str | None = None):
        try:
            return self.client.place_limit_sell(
                qty_base=qty,
                price=price,
                symbol=symbol,
                client_oid=client_oid,
            )
        except TypeError:
            return self.client.place_limit_sell(
                qty_base=qty,
                price=price,
                symbol=symbol,
                clientOrderId=client_oid,
            )

    def _pending_buy_key(self, suffix: str) -> str:
        return f"pending_buy_{suffix}"

    def _get_pending_buy_order_id(self) -> int | None:
        v = self.db.get_setting(self._pending_buy_key("order_id"), None)
        if not v:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _set_pending_buy(
        self,
        *,
        order_id: int,
        order_time_ms: int,
        core_bucket_usdt: float,
        kind: str,
        trigger_price: float | None = None,
    ):
        self.db.set_setting(self._pending_buy_key("order_id"), int(order_id))
        self.db.set_setting(self._pending_buy_key("time_ms"), int(order_time_ms))
        self.db.set_setting(self._pending_buy_key("core_bucket_usdt"), float(core_bucket_usdt))
        self.db.set_setting(self._pending_buy_key("kind"), str(kind))
        if trigger_price is None:
            self.db.set_setting(self._pending_buy_key("trigger_price"), "")
        else:
            self.db.set_setting(self._pending_buy_key("trigger_price"), float(trigger_price))

    def _clear_pending_buy(self):
        self.db.set_setting(self._pending_buy_key("order_id"), "")
        self.db.set_setting(self._pending_buy_key("time_ms"), "")
        self.db.set_setting(self._pending_buy_key("core_bucket_usdt"), "")
        self.db.set_setting(self._pending_buy_key("kind"), "")
        self.db.set_setting(self._pending_buy_key("trigger_price"), "")

    def _get_pending_buy_context(self) -> dict:
        order_id = self._get_pending_buy_order_id()
        if order_id is None:
            return {}
        time_ms = int(self._get_float_setting(self._pending_buy_key("time_ms"), 0.0))
        core_bucket = float(self._get_float_setting(self._pending_buy_key("core_bucket_usdt"), 0.0))
        kind = str(self.db.get_setting(self._pending_buy_key("kind"), "") or "")
        trigger_price = float(self._get_float_setting(self._pending_buy_key("trigger_price"), 0.0))
        return {
            "order_id": order_id,
            "time_ms": time_ms,
            "core_bucket_usdt": core_bucket,
            "kind": kind,
            "trigger_price": trigger_price,
        }

    def _extract_fee_usdt_from_order(self, order: dict) -> float:
        quote_asset = str(self._cfg("quote_asset", "USDT"))
        fee_usdt = 0.0
        fills = order.get("fills") or []
        for f in fills:
            try:
                commission = float(f.get("commission") or 0.0)
            except Exception:
                commission = 0.0
            asset = str(f.get("commissionAsset") or "")
            if commission <= 0:
                continue
            if asset == quote_asset:
                fee_usdt += commission
        return float(fee_usdt)

    def _handle_filled_buy(self, *, order: dict, core_bucket_locked: float, kind: str):
        try:
            bought_qty = float(order.get("executedQty") or order.get("origQty") or 0.0)
        except Exception:
            bought_qty = 0.0
        try:
            spent_usdt = float(order.get("cummulativeQuoteQty") or 0.0)
        except Exception:
            spent_usdt = 0.0

        if spent_usdt <= 0 and bought_qty > 0:
            try:
                order_price = float(order.get("price") or 0.0)
            except Exception:
                order_price = 0.0
            if order_price > 0:
                spent_usdt = order_price * bought_qty

        if bought_qty > 0 and spent_usdt > 0:
            avg_price = spent_usdt / bought_qty
        else:
            avg_price = self.client.get_price(self.cfg.symbol)

        if kind == "INIT":
            reserve_qty = self.client.adjust_qty(bought_qty, self.cfg.symbol)
            self._set_reserve_btc_qty(reserve_qty)
            self._set_reserve_cost_usdt(spent_usdt)
            if self.db.get_setting("core_btc_initial", None) in (None, "", 0, 0.0):
                self.db.set_setting("core_btc_initial", reserve_qty)
            self.db.set_setting("init_entry_done", "1")
            self.db.log("INFO", f"RESERVE set: reserve_btc_qty={reserve_qty:.8f}")
        else:
            core_used = min(max(0.0, core_bucket_locked), spent_usdt)
            if core_used > 0 and spent_usdt > 0 and bought_qty > 0:
                core_btc_add = bought_qty * (core_used / spent_usdt)
                reserve_qty = self._get_reserve_btc_qty()
                self._set_reserve_btc_qty(reserve_qty + core_btc_add)
                reserve_cost = self._get_reserve_cost_usdt()
                self._set_reserve_cost_usdt(reserve_cost + core_used)
                cur_bucket = self._get_float_setting("core_bucket_usdt", 0.0)
                self.db.set_setting("core_bucket_usdt", max(0.0, cur_bucket - core_used))

            lot_btc_qty = max(
                0.0,
                bought_qty - (bought_qty * (core_used / spent_usdt) if spent_usdt > 0 else 0.0),
            )
            if lot_btc_qty > 0:
                lot_id = self.db.insert_lot(
                    symbol=self.cfg.symbol,
                    buy_price=avg_price,
                    buy_btc_qty=lot_btc_qty,
                    buy_time_ms=int(self._now() * 1000),
                )
                self.db.log(
                    "INFO",
                    f"LOT BUY saved: lot_id={lot_id} qty={lot_btc_qty:.8f} price={avg_price:.2f}",
                )

        self.set_base_price(avg_price)

    def _process_pending_buy(self, cur_price: float | None = None) -> bool:
        ctx = self._get_pending_buy_context()
        if not ctx:
            return False

        order_id = ctx["order_id"]
        try:
            o = self.client.get_order(order_id=order_id, symbol=self.cfg.symbol)
            self.db.upsert_order(o)
        except Exception as e:
            self.db.log("WARN", f"pending buy get_order failed: order_id={order_id} err={e}")
            return True

        status = str(o.get("status") or "")
        now_ms = int(self._now() * 1000)
        order_time_ms = int(ctx.get("time_ms") or 0)

        if status == "FILLED":
            self._handle_filled_buy(
                order=o,
                core_bucket_locked=float(ctx.get("core_bucket_usdt") or 0.0),
                kind=str(ctx.get("kind") or ""),
            )
            self._clear_pending_buy()
            return True

        if status in ("CANCELED", "REJECTED", "EXPIRED"):
            self._clear_pending_buy()
            return True

        if status == "NEW":
            trigger_price = float(ctx.get("trigger_price") or 0.0)
            if cur_price is not None and trigger_price > 0 and str(ctx.get("kind")) == "LOT":
                rebound_pct = float(self._p("lot_cancel_rebound_pct", 0.005))
                if cur_price >= trigger_price * (1.0 + rebound_pct):
                    try:
                        self.client.cancel_order(order_id=order_id, symbol=self.cfg.symbol)
                    except Exception as e:
                        self.db.log("WARN", f"pending buy cancel failed: order_id={order_id} err={e}")
                    self._clear_pending_buy()
                    return True

        return True

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
        trade_cap_ratio = float(self._p("trade_cap_ratio", float(self._cfg("trade_cap_ratio", 0.30))))
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
        used_total = max(0.0, float(pos.cost_basis_usdt))
        reserve_cost = self._get_reserve_cost_usdt()
        used = max(0.0, used_total - reserve_cost)
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
        - initial_buy_usdt/initial_entry_usdt가 있으면 '고정 금액'으로 매수
        - 초기 매수는 전량 core(reserve_btc_qty)로 저장
        """
        if not self._cfg("initial_buy_on_start", False):
            return

        if self.db.get_setting("init_entry_done", None) == "1":
            return

        cur_price = self.client.get_price(self.cfg.symbol)
        if self._process_pending_buy(cur_price):
            return

        quote_asset = self._cfg("quote_asset", "USDT")
        reserve_buffer = float(self._cfg("usdt_reserve_buffer", 0.0))

        # ✅ 1) 초기 코어 매수 금액
        fixed_usdt = getattr(self.cfg, "initial_buy_usdt", None)
        if fixed_usdt is None:
            fixed_usdt = getattr(self.cfg, "initial_entry_usdt", None)
        if fixed_usdt is None:
            fixed_usdt = float(getattr(self.cfg, "initial_core_usdt", 2000.0))

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
        min_notional = max(float(filters.min_notional), float(self._p("min_trade_usdt", float(self._cfg("min_trade_usdt", 0.0)))))
        if buy_usdt < min_notional:
            self.db.log("WARN", f"INIT BUY skipped: buy_usdt({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
            return

        if not self._cooldown_ok():
            return

        try:
            prefix = str(getattr(self.cfg, "client_order_prefix", "BTCSTACK_"))
            client_oid = f"{prefix}INIT_{int(self._now()*1000)}"
            limit_price = self.client.adjust_price(self.client.get_price(self.cfg.symbol), self.cfg.symbol)

            o = self._place_limit_buy_by_quote(
                quote_usdt=buy_usdt,
                price=limit_price,
                symbol=self.cfg.symbol,
                client_oid=client_oid,
            )
            self.db.upsert_order(o)
            self._touch_order()

            order_time_ms = int(o.get("transactTime") or (self._now() * 1000))
            self._set_pending_buy(
                order_id=int(o.get("orderId")),
                order_time_ms=order_time_ms,
                core_bucket_usdt=0.0,
                kind="INIT",
            )
            self.db.log(
                "INFO",
                f"INIT BUY placed: quote={buy_usdt:.2f} {quote_asset} price={limit_price:.2f}",
            )
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
        if self._p("verbose", self.cfg.verbose):
            self.db.log(
                "INFO",
                f"position -> symbol={snap.symbol} btc={snap.btc_qty:.8f} avg={snap.avg_entry:.2f} cost={snap.cost_basis_usdt:.2f}",
            )

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

    # -----------------------------
    # Cooldown
    # -----------------------------
    def _cooldown_ok(self) -> bool:
        return (self._now() - self._last_order_ts) >= int(self._cfg("order_cooldown_sec", 7))

    def _touch_order(self):
        self._last_order_ts = self._now()

    # -----------------------------
    # Buy on drop
    # -----------------------------
    def maybe_buy_on_drop(self, cur_price: float, pos_all: PositionSnapshot):
        if self._process_pending_buy():
            return
        if not self._cooldown_ok():
            return

        base = self.get_base_price()
        if base <= 0:
            self.set_base_price(cur_price)
            return

        drop_pct = float(self._p("lot_drop_pct", 0.01))
        trigger = base * (1.0 - drop_pct)
        prebuy_pct = float(self._p("lot_prebuy_pct", 0.002))
        prebuy_price = trigger * (1.0 + prebuy_pct)

        if cur_price > prebuy_price:
            return

        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, float(self._p("min_trade_usdt", float(self._cfg("min_trade_usdt", 0.0)))))

        cap = self._get_trade_cap_snapshot(pos_all)
        spendable_free = float(cap["spendable_free"])

        core_bucket = self._get_float_setting("core_bucket_usdt", 0.0)
        lot_buy_usdt = float(self._p("lot_buy_usdt", 100.0))
        buy_usdt = lot_buy_usdt + max(0.0, core_bucket)
        buy_usdt = min(buy_usdt, spendable_free)

        if buy_usdt < min_notional:
            self.db.log("WARN", f"buy_usdt too small ({buy_usdt:.2f}) < min_notional({min_notional:.2f})")
            return

        try:
            prefix = str(getattr(self.cfg, "client_order_prefix", "BTCSTACK_"))
            client_oid = f"{prefix}LOTBUY_{int(self._now()*1000)}"
            limit_price = self.client.adjust_price(trigger, self.cfg.symbol)

            o = self._place_limit_buy_by_quote(
                quote_usdt=buy_usdt,
                price=limit_price,
                symbol=self.cfg.symbol,
                client_oid=client_oid,
            )
            self.db.upsert_order(o)
            self._touch_order()

            order_time_ms = int(o.get("transactTime") or (self._now() * 1000))
            self._set_pending_buy(
                order_id=int(o.get("orderId")),
                order_time_ms=order_time_ms,
                core_bucket_usdt=max(0.0, core_bucket),
                kind="LOT",
                trigger_price=trigger,
            )

            self.db.log(
                "INFO",
                f"BUY placed: quote={buy_usdt:.2f} | base={base:.2f} trig={trigger:.2f} limit={limit_price:.2f}",
            )
        except Exception as e:
            self.db.log("ERROR", f"BUY failed: {e}")

    # -----------------------------
    # Lot take-profit
    # -----------------------------
    def maybe_take_profit_lots(self, cur_price: float):
        open_lots = self.db.get_open_lots(self.cfg.symbol)
        if not open_lots:
            return

        filters = self.client.get_symbol_filters(self.cfg.symbol)
        min_notional = max(filters.min_notional, float(self._p("min_trade_usdt", float(self._cfg("min_trade_usdt", 0.0)))))

        for lot in open_lots:
            try:
                lot_id = int(lot.get("lot_id"))
                buy_price = float(lot.get("buy_price") or 0.0)
                buy_qty = float(lot.get("buy_btc_qty") or 0.0)
                sell_order_id = lot.get("sell_order_id")
                sell_order_id = int(sell_order_id) if sell_order_id is not None else None
                sell_order_time_ms = int(lot.get("sell_order_time_ms") or 0)
            except Exception:
                continue

            if buy_price <= 0 or buy_qty <= 0:
                continue

            target_price = buy_price * (1.0 + float(self._p("lot_tp_pct", 0.03)))

            qty = self.client.adjust_qty(buy_qty, self.cfg.symbol)
            if qty <= 0:
                continue

            notional = qty * target_price
            if notional < min_notional:
                continue

            if sell_order_id is not None:
                try:
                    o = self.client.get_order(order_id=sell_order_id, symbol=self.cfg.symbol)
                    self.db.upsert_order(o)
                except Exception as e:
                    self.db.log("WARN", f"LOT SELL get_order failed: lot_id={lot_id} err={e}")
                    continue

                status = str(o.get("status") or "")
                now_ms = int(self._now() * 1000)
                if status == "FILLED":
                    try:
                        sold_qty = float(o.get("executedQty") or o.get("origQty") or 0.0)
                    except Exception:
                        sold_qty = 0.0
                    try:
                        recv_usdt = float(o.get("cummulativeQuoteQty") or 0.0)
                    except Exception:
                        recv_usdt = 0.0

                    if sold_qty > 0 and recv_usdt > 0:
                        sell_price = recv_usdt / sold_qty
                    else:
                        try:
                            order_price = float(o.get("price") or 0.0)
                        except Exception:
                            order_price = 0.0
                        sell_price = order_price if order_price > 0 else target_price

                    fee_usdt = self._extract_fee_usdt_from_order(o)
                    gross_profit = (sell_price - buy_price) * sold_qty
                    net_profit = gross_profit - fee_usdt

                    if net_profit > 0:
                        core_bucket = self._get_float_setting("core_bucket_usdt", 0.0)
                        self.db.set_setting("core_bucket_usdt", core_bucket + net_profit)

                    self.db.close_lot(
                        lot_id=lot_id,
                        sell_price=sell_price,
                        sell_time_ms=int(self._now() * 1000),
                        fee_usdt=fee_usdt,
                        net_profit_usdt=net_profit,
                    )
                    self.set_base_price(sell_price)
                    self.db.log(
                        "INFO",
                        f"LOT SELL done: lot_id={lot_id} qty={sold_qty:.8f} sell={sell_price:.2f} net={net_profit:.2f}",
                    )
                    continue

                if status in ("CANCELED", "REJECTED", "EXPIRED"):
                    self.db.clear_lot_sell_order(lot_id=lot_id)
                    continue

                if status == "NEW":
                    continue

            if not self._cooldown_ok():
                return

            try:
                prefix = str(getattr(self.cfg, "client_order_prefix", "BTCSTACK_"))
                client_oid = f"{prefix}LOTSELL_{lot_id}_{int(self._now()*1000)}"
                o = self._place_limit_sell(
                    qty=qty,
                    price=target_price,
                    symbol=self.cfg.symbol,
                    client_oid=client_oid,
                )
                self.db.upsert_order(o)
                self._touch_order()
                order_time_ms = int(o.get("transactTime") or (self._now() * 1000))
                self.db.set_lot_sell_order(
                    lot_id=lot_id,
                    order_id=int(o.get("orderId")),
                    order_time_ms=order_time_ms,
                )
                self.db.log(
                    "INFO",
                    f"LOT SELL placed: lot_id={lot_id} qty={qty:.8f} price={target_price:.2f}",
                )
            except Exception as e:
                self.db.log("ERROR", f"LOT SELL failed: lot_id={lot_id} err={e}")

    # -----------------------------
    # Reserve (core BTC) helpers
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

    def _get_reserve_cost_usdt(self) -> float:
        v = self.db.get_setting("reserve_cost_usdt", 0.0)
        try:
            return float(v) if v is not None and str(v) != "" else 0.0
        except Exception:
            return 0.0

    def _set_reserve_cost_usdt(self, cost: float):
        try:
            cost = float(cost)
        except Exception:
            cost = 0.0
        self.db.set_setting("reserve_cost_usdt", cost)
    # -----------------------------
    # Main loop
    # -----------------------------
    def step(self):
        self.sync_orders_and_fills()

        cur_price = self.client.get_price(self.cfg.symbol)
        self._p("verbose", self.cfg.verbose)

        pos_all = self.db.get_position(self.cfg.symbol)

        if self.db.get_setting("init_entry_done", None) != "1":
            self.maybe_initial_market_entry()

        self.maybe_take_profit_lots(cur_price)
        self.maybe_buy_on_drop(cur_price, pos_all)

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
