# back_test.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class BTCBacktestConfig:
    """
    Backtest-only config. Keep only fields used by backtest execution.
    Env overrides supported via BT_* prefix.
    """

    # Mode / Logging
    verbose: bool = False
    ai_enable_report: bool = True
    ai_enable_gate: bool = True
    ai_enable_tune: bool = True

    # Symbol
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    # Backtest Account Assumptions
    init_usdt: float = 100000.0
    init_btc: float = 0.0
    initial_buy_on_start: bool = True
    initial_buy_ratio: float = 0.70
    initial_buy_usdt: Optional[float] = None
    initial_entry_usdt: Optional[float] = None
    initial_reserve_ratio: float = 0.50
    reserve_btc_key: str = "reserve_btc_qty"

    # Execution Assumptions
    slippage_bps: float = 1.0
    fee_bps_taker: float = 10.0
    fee_bps_maker: float = 2.0

    # Exchange Filters
    tick_size: float = 0.1
    step_size: float = 0.000001
    min_notional: float = 10.0

    # Order ID Prefix (pool tagging)
    client_order_prefix: str = "BTCSTACK_"

    # Strategy (Stacking/Grid)
    grid_step_pct: float = 0.02
    take_profit_pct: float = 0.020
    buy_quote_usdt: float = 20.0
    sell_fraction_on_tp: float = 0.15
    min_trade_usdt: float = 0.0

    # TP as LIMIT
    use_tp_limit_orders: bool = True
    tp_refresh_sec: int = 30
    tp_price_bump_ticks: int = 0
    trailing_sell_enable: bool = False
    trailing_activate_pct: float = 0.02
    trailing_ratio: float = 0.70

    # Buy sizing
    fixed_grid_buy: bool = True
    grid_buy_usdt: float = 20.0
    dynamic_buy: bool = True
    buy_min_usdt: float = 5.0
    buy_max_usdt: float = 60.0
    exposure_power: float = 1.6
    price_vol_window: int = 30
    vol_low: float = 0.005
    vol_high: float = 0.012
    vol_boost_max: float = 1.25
    vol_cut_min: float = 0.70

    # Risk / Limits
    max_quote_exposure_usdt: float = 1e18

    # Base price recentering
    recenter_threshold_pct: float = 0.020
    recenter_cooldown_sec: int = 60

    # Crash-aware sizing
    crash_drop_pct: float = 0.035
    crash_vol_threshold: float = 0.015
    crash_grid_mult: float = 2.0

    # Timing
    order_cooldown_sec: int = 7
    loop_interval_sec: int = 60

    # Capital split
    trade_cap_ratio: float = 1.0
    usdt_reserve_buffer: float = 0.0
    use_fixed_usdt_reference: bool = False

    # Optional: BT_* env override
    @staticmethod
    def _getenv(name: str, default: str) -> str:
        return os.getenv(f"BT_{name}", default)

    @classmethod
    def from_env(cls) -> "BTCBacktestConfig":
        def f(name: str, default: float) -> float:
            return float(cls._getenv(name, str(default)))

        def i(name: str, default: int) -> int:
            return int(float(cls._getenv(name, str(default))))

        def b(name: str, default: bool) -> bool:
            return cls._getenv(name, "true" if default else "false").lower() == "true"

        def s(name: str, default: str) -> str:
            return cls._getenv(name, default)

        return cls(
            verbose=b("VERBOSE", cls.verbose),
            ai_enable_report=b("AI_ENABLE_REPORT", cls.ai_enable_report),
            ai_enable_gate=b("AI_ENABLE_GATE", cls.ai_enable_gate),
            ai_enable_tune=b("AI_ENABLE_TUNE", cls.ai_enable_tune),

            symbol=s("SYMBOL", cls.symbol),
            base_asset=s("BASE_ASSET", cls.base_asset),
            quote_asset=s("QUOTE_ASSET", cls.quote_asset),

            init_usdt=f("INIT_USDT", cls.init_usdt),
            init_btc=f("INIT_BTC", cls.init_btc),
            initial_buy_on_start=b("INITIAL_BUY_ON_START", cls.initial_buy_on_start),
            initial_buy_ratio=f("INITIAL_BUY_RATIO", cls.initial_buy_ratio),
            initial_buy_usdt=f("INITIAL_BUY_USDT", cls.initial_buy_usdt) if os.getenv("BT_INITIAL_BUY_USDT") is not None else None,
            initial_entry_usdt=f("INITIAL_ENTRY_USDT", cls.initial_entry_usdt) if os.getenv("BT_INITIAL_ENTRY_USDT") is not None else None,
            initial_reserve_ratio=f("INITIAL_RESERVE_RATIO", cls.initial_reserve_ratio),
            reserve_btc_key=s("RESERVE_BTC_KEY", cls.reserve_btc_key),

            slippage_bps=f("SLIPPAGE_BPS", cls.slippage_bps),
            fee_bps_taker=f("FEE_BPS_TAKER", cls.fee_bps_taker),
            fee_bps_maker=f("FEE_BPS_MAKER", cls.fee_bps_maker),

            tick_size=f("TICK_SIZE", cls.tick_size),
            step_size=f("STEP_SIZE", cls.step_size),
            min_notional=f("MIN_NOTIONAL", cls.min_notional),

            client_order_prefix=s("CLIENT_ORDER_PREFIX", cls.client_order_prefix),

            grid_step_pct=f("GRID_STEP_PCT", cls.grid_step_pct),
            take_profit_pct=f("TAKE_PROFIT_PCT", cls.take_profit_pct),
            buy_quote_usdt=f("BUY_QUOTE_USDT", cls.buy_quote_usdt),
            sell_fraction_on_tp=f("SELL_FRACTION_ON_TP", cls.sell_fraction_on_tp),
            min_trade_usdt=f("MIN_TRADE_USDT", cls.min_trade_usdt),

            use_tp_limit_orders=b("USE_TP_LIMIT_ORDERS", cls.use_tp_limit_orders),
            tp_refresh_sec=i("TP_REFRESH_SEC", cls.tp_refresh_sec),
            tp_price_bump_ticks=i("TP_PRICE_BUMP_TICKS", cls.tp_price_bump_ticks),
            trailing_sell_enable=b("TRAILING_SELL_ENABLE", cls.trailing_sell_enable),
            trailing_activate_pct=f("TRAILING_ACTIVATE_PCT", cls.trailing_activate_pct),
            trailing_ratio=f("TRAILING_RATIO", cls.trailing_ratio),

            fixed_grid_buy=b("FIXED_GRID_BUY", cls.fixed_grid_buy),
            grid_buy_usdt=f("GRID_BUY_USDT", cls.grid_buy_usdt),
            dynamic_buy=b("DYNAMIC_BUY", cls.dynamic_buy),
            buy_min_usdt=f("BUY_MIN_USDT", cls.buy_min_usdt),
            buy_max_usdt=f("BUY_MAX_USDT", cls.buy_max_usdt),
            exposure_power=f("EXPOSURE_POWER", cls.exposure_power),
            price_vol_window=i("PRICE_VOL_WINDOW", cls.price_vol_window),
            vol_low=f("VOL_LOW", cls.vol_low),
            vol_high=f("VOL_HIGH", cls.vol_high),
            vol_boost_max=f("VOL_BOOST_MAX", cls.vol_boost_max),
            vol_cut_min=f("VOL_CUT_MIN", cls.vol_cut_min),

            max_quote_exposure_usdt=f("MAX_QUOTE_EXPOSURE_USDT", cls.max_quote_exposure_usdt),

            recenter_threshold_pct=f("RECENTER_THRESHOLD_PCT", cls.recenter_threshold_pct),
            recenter_cooldown_sec=i("RECENTER_COOLDOWN_SEC", cls.recenter_cooldown_sec),

            crash_drop_pct=f("CRASH_DROP_PCT", cls.crash_drop_pct),
            crash_vol_threshold=f("CRASH_VOL_THRESHOLD", cls.crash_vol_threshold),
            crash_grid_mult=f("CRASH_GRID_MULT", cls.crash_grid_mult),

            order_cooldown_sec=i("ORDER_COOLDOWN_SEC", cls.order_cooldown_sec),
            loop_interval_sec=i("LOOP_INTERVAL_SEC", cls.loop_interval_sec),

            trade_cap_ratio=f("TRADE_CAP_RATIO", cls.trade_cap_ratio),
            usdt_reserve_buffer=f("USDT_RESERVE_BUFFER", cls.usdt_reserve_buffer),
            use_fixed_usdt_reference=b("USE_FIXED_USDT_REFERENCE", cls.use_fixed_usdt_reference),
        )


@dataclass
class SymbolFilters:
    tick_size: float = 0.1
    step_size: float = 0.000001
    min_notional: float = 10.0


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(float(x) / step) * step


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick


class BacktestClient:
    """
    Binance-like 최소 인터페이스를 흉내내는 백테스트용 Client.

    - market buy: 즉시 체결 (fill_price는 close 기반 + slippage)
    - limit sell(TP): 이후 봉의 high가 price 이상이면 체결 처리
    - 추가입금(topup) 가정은 하지 않음 (free 부족하면 주문 실패)
    """

    def __init__(
        self,
        symbol: str,
        *,
        init_usdt: float,
        init_btc: float,
        filters: SymbolFilters,
        taker_fee_rate: float,
        maker_fee_rate: float,
        slippage_bps: float,
        quote_asset: str,
        base_asset: str,
        client_order_prefix: str = "BTCSTACK_",
    ):
        self.symbol = symbol
        self.quote_asset = quote_asset
        self.base_asset = base_asset
        self.client_order_prefix = str(client_order_prefix)

        self.filters = filters
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)
        self.slippage_bps = float(slippage_bps)

        # 잔고 (free/locked 단순 모델)
        self._quote_free = float(init_usdt)
        self._quote_locked = 0.0
        self._base_free = float(init_btc)
        self._base_locked = 0.0

        # 현재 봉 상태
        self.is_backtest = True
        self._cur_ts = 0.0
        self.cur_ts: float = 0.0
        self.cur_open: float = 0.0
        self.cur_high: float = 0.0
        self.cur_low: float = 0.0
        self.cur_close: float = 0.0

        # 주문/체결
        self._next_order_id = 1000
        self._orders: Dict[int, Dict[str, Any]] = {}
        self._fills: List[Dict[str, Any]] = []
        self._fills_cursor = 0

        # stats (숫자만 모으기)
        self.stats = defaultdict(float)
        self.stats["init_usdt"] = float(init_usdt)
        self.stats["init_btc"] = float(init_btc)
        self.stats["market_buy_orders"] = 0.0
        self.stats["tp_limit_orders"] = 0.0
        self.stats["tp_limit_fills"] = 0.0
        self.stats["cancels"] = 0.0
        self.stats["fee_total"] = 0.0
        self.stats["buy_spent_total"] = 0.0
        self.stats["buy_fee_total"] = 0.0
        self.stats["sell_gross_total"] = 0.0
        self.stats["sell_fee_total"] = 0.0

    # ----------------------------
    # stats
    # ----------------------------
    def get_stats(self) -> dict:
        return {k: float(v) for k, v in dict(self.stats).items()}

    # ----------------------------
    # bar control
    # ----------------------------
    def set_bar(self, ts: float, o: float, h: float, l: float, c: float):
        self.cur_ts = float(ts)
        self._cur_ts = float(ts)
        self.cur_open = float(o)
        self.cur_high = float(h)
        self.cur_low = float(l)
        self.cur_close = float(c)

    def match_open_orders(self):
        """
        현재 봉의 high를 이용해 limit SELL 체결 처리.
        (단순화: high >= limit_price 면 전량 체결)
        """
        for oid, o in list(self._orders.items()):
            if o.get("symbol") != self.symbol:
                continue
            if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
                continue
            if o.get("type") != "LIMIT":
                continue
            if o.get("side") != "SELL":
                continue

            price = float(o.get("price") or 0.0)
            qty = float(o.get("origQty") or 0.0)
            if price <= 0 or qty <= 0:
                continue

            if self.cur_high >= price:
                self._fill_limit_sell(oid, price=price, qty=qty)

    # ----------------------------
    # exchange-like methods
    # ----------------------------
    def get_price(self, symbol: str) -> float:
        assert symbol == self.symbol
        return float(self.cur_close)

    def get_balance(self, asset: str) -> Dict[str, float]:
        if asset == self.quote_asset:
            total = self._quote_free + self._quote_locked
            return {"free": self._quote_free, "locked": self._quote_locked, "total": total}
        if asset == self.base_asset:
            total = self._base_free + self._base_locked
            return {"free": self._base_free, "locked": self._base_locked, "total": total}
        return {"free": 0.0, "locked": 0.0, "total": 0.0}

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        assert symbol == self.symbol
        return self.filters

    def adjust_price(self, price: float, symbol: str) -> float:
        f = self.get_symbol_filters(symbol)
        return float(_round_to_tick(price, f.tick_size))

    def adjust_qty(self, qty: float, symbol: str) -> float:
        f = self.get_symbol_filters(symbol)
        return float(_floor_to_step(qty, f.step_size))

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        assert symbol == self.symbol
        return [
            o.copy()
            for o in self._orders.values()
            if o.get("symbol") == symbol and o.get("status") in ("NEW", "PARTIALLY_FILLED")
        ]

    def get_order(self, order_id: int, symbol: str) -> Dict[str, Any]:
        assert symbol == self.symbol
        o = self._orders.get(int(order_id))
        if not o:
            raise KeyError(f"order not found: {order_id}")
        return o.copy()

    def cancel_order(self, order_id: int, symbol: str) -> Dict[str, Any]:
        assert symbol == self.symbol
        oid = int(order_id)
        o = self._orders.get(oid)
        if not o:
            raise KeyError(f"order not found: {oid}")

        if o.get("status") in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
            return o.copy()

        self.stats["cancels"] += 1.0

        # SELL LIMIT만 lock 해제
        if o.get("type") == "LIMIT" and o.get("side") == "SELL":
            qty = float(o.get("origQty") or 0.0)
            qty = max(0.0, qty)
            unlock = min(self._base_locked, qty)
            self._base_locked -= unlock
            self._base_free += unlock

        o["status"] = "CANCELED"
        self._orders[oid] = o
        return o.copy()

    def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        assert symbol == self.symbol
        new = self._fills[self._fills_cursor :]
        self._fills_cursor = len(self._fills)
        return [t.copy() for t in new]

    # ----------------------------
    # place orders
    # ----------------------------
    def _new_order_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id

    def place_market_buy_by_quote(
        self,
        *,
        quote_usdt: float,
        symbol: str,
        clientOrderId: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert symbol == self.symbol
        q = float(quote_usdt)
        if q <= 0:
            raise ValueError("quote_usdt must be > 0")

        if q > self._quote_free + 1e-12:
            raise ValueError(f"insufficient {self.quote_asset} free: need={q:.8f}, have={self._quote_free:.8f}")

        px = float(self.cur_close) * (1.0 + self.slippage_bps / 10000.0)
        px = self.adjust_price(px, symbol)

        # 수수료를 quote에서 차감한다고 가정
        fee_q = q * self.taker_fee_rate
        q_net = max(0.0, q - fee_q)

        qty = (q_net / px) if px > 0 else 0.0
        qty = self.adjust_qty(qty, symbol)

        notional = qty * px
        if notional < max(self.filters.min_notional, 0.0):
            raise ValueError(f"min_notional not met: notional={notional:.4f}")

        # step에 맞추며 notional 줄 수 있으니 재계산
        spent = qty * px
        fee_q = spent * self.taker_fee_rate
        total_spent = spent + fee_q
        if total_spent > self._quote_free + 1e-12:
            raise ValueError("insufficient free after rounding")

        oid = self._new_order_id()
        coid = clientOrderId or f"{self.client_order_prefix}{oid}"

        order = {
            "symbol": symbol,
            "orderId": oid,
            "clientOrderId": coid,
            "transactTime": int(self.cur_ts * 1000),
            "price": "0",
            "origQty": f"{qty:.8f}",
            "executedQty": f"{qty:.8f}",
            "cummulativeQuoteQty": f"{spent:.8f}",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY",
        }
        self._orders[oid] = order

        self._quote_free -= total_spent
        self._base_free += qty

        # stats
        self.stats["market_buy_orders"] += 1.0
        self.stats["fee_total"] += float(fee_q)
        self.stats["buy_spent_total"] += float(spent)
        self.stats["buy_fee_total"] += float(fee_q)

        trade = {
            "symbol": symbol,
            "id": len(self._fills) + 1,
            "orderId": oid,
            "price": f"{px:.8f}",
            "qty": f"{qty:.8f}",
            "quoteQty": f"{spent:.8f}",
            "commission": f"{fee_q:.8f}",
            "commissionAsset": self.quote_asset,
            "time": int(self.cur_ts * 1000),
            "isBuyer": True,
            "isMaker": False,
        }
        self._fills.append(trade)

        return order.copy()

    def place_limit_sell(
        self,
        *,
        qty_base: float,
        price: float,
        symbol: str,
        clientOrderId: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert symbol == self.symbol
        qty = self.adjust_qty(float(qty_base), symbol)
        px = self.adjust_price(float(price), symbol)

        if qty <= 0:
            raise ValueError("qty must be > 0")
        if px <= 0:
            raise ValueError("price must be > 0")

        notional = qty * px
        if notional < max(self.filters.min_notional, 0.0):
            raise ValueError(f"min_notional not met: notional={notional:.4f}")

        if qty > self._base_free + 1e-12:
            raise ValueError(f"insufficient {self.base_asset} free: need={qty}, have={self._base_free}")

        # lock base
        self._base_free -= qty
        self._base_locked += qty

        oid = self._new_order_id()
        coid = clientOrderId or f"{self.client_order_prefix}{oid}"

        order = {
            "symbol": symbol,
            "orderId": oid,
            "clientOrderId": coid,
            "transactTime": int(self.cur_ts * 1000),
            "price": f"{px:.8f}",
            "origQty": f"{qty:.8f}",
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "SELL",
        }
        self._orders[oid] = order

        self.stats["tp_limit_orders"] += 1.0
        return order.copy()

    def _fill_limit_sell(self, oid: int, *, price: float, qty: float):
        o = self._orders.get(int(oid))
        if not o:
            return
        if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
            return

        px = self.adjust_price(price, self.symbol)
        qty = self.adjust_qty(qty, self.symbol)

        gross = qty * px
        fee_q = gross * self.maker_fee_rate
        net = gross - fee_q

        # balances
        unlock = min(self._base_locked, qty)
        self._base_locked -= unlock
        self._quote_free += net

        # order update
        o["status"] = "FILLED"
        o["executedQty"] = f"{qty:.8f}"
        o["cummulativeQuoteQty"] = f"{gross:.8f}"
        self._orders[int(oid)] = o

        # stats
        self.stats["tp_limit_fills"] += 1.0
        self.stats["fee_total"] += float(fee_q)
        self.stats["sell_gross_total"] += float(gross)
        self.stats["sell_fee_total"] += float(fee_q)

        trade = {
            "symbol": self.symbol,
            "id": len(self._fills) + 1,
            "orderId": int(oid),
            "price": f"{px:.8f}",
            "qty": f"{qty:.8f}",
            "quoteQty": f"{gross:.8f}",
            "commission": f"{fee_q:.8f}",
            "commissionAsset": self.quote_asset,
            "time": int(self.cur_ts * 1000),
            "isBuyer": False,
            "isMaker": True,
        }
        self._fills.append(trade)


@dataclass
class PositionSnapshot:
    symbol: str
    btc_qty: float
    avg_entry: float
    cost_basis_usdt: float


class BacktestDB:
    """
    BTCStackingTrader가 요구하는 최소 DB API를 in-memory로 구현.
    - orders 저장
    - fills 저장
    - settings 저장
    - fills 기반 포지션 재계산(전체/POOL)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._settings: Dict[str, Any] = {}

        self._orders: Dict[int, Dict[str, Any]] = {}
        self._fills: List[Dict[str, Any]] = []
        self._ai_events: List[Dict[str, Any]] = []

        self._pos_by_symbol: Dict[str, PositionSnapshot] = {}
        self._pool_pos_by_symbol: Dict[str, PositionSnapshot] = {}

    # ----------------------------
    # logging/settings
    # ----------------------------
    def log(self, level: str, msg: str):
        if self.verbose or level in ("ERROR", "WARN"):
            print(f"[{level}] {msg}")

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any):
        self._settings[key] = value

    # ----------------------------
    # orders
    # ----------------------------
    def upsert_order(self, o: Dict[str, Any]):
        oid = int(o["orderId"])
        self._orders[oid] = dict(o)

    def get_recent_open_orders(self, limit: int = 50) -> List[int]:
        out = []
        for oid, o in self._orders.items():
            if o.get("status") in ("NEW", "PARTIALLY_FILLED"):
                out.append(int(oid))
        out.sort(reverse=True)
        return out[: int(limit)]

    # ----------------------------
    # fills
    # ----------------------------
    def insert_fill(self, order_id: int, t: Dict[str, Any]):
        rec = dict(t)
        rec["orderId"] = int(order_id)
        self._fills.append(rec)
        o = self._orders.get(int(order_id)) or {}
        cid = (o.get("clientOrderId") or "")
        if "TRAIL" in str(cid):
            side = "BUY" if bool(rec.get("isBuyer")) else "SELL"
            self.log("INFO", f"TRAIL fill: side={side} qty={rec.get('qty')} price={rec.get('price')}")

    # ----------------------------
    # ai events
    # ----------------------------
    def insert_ai_event(self, kind: str, symbol: str, payload: Dict[str, Any]):
        self._ai_events.append({"kind": kind, "symbol": symbol, "payload": dict(payload)})

    # ----------------------------
    # positions
    # ----------------------------
    def _recompute_position(self, symbol: str, *, client_prefix: Optional[str] = None) -> PositionSnapshot:
        """
        단순 평균단가 기반 포지션 계산:
        - BUY: cost_basis += quoteQty (=gross)
        - SELL: cost_basis -= avg_entry * qty_sold (원가만 감소)
        """
        qty = 0.0
        cost = 0.0
        avg = 0.0

        for f in self._fills:
            if f.get("symbol") != symbol:
                continue

            oid = int(f.get("orderId") or 0)
            if oid <= 0:
                continue

            if client_prefix is not None:
                o = self._orders.get(oid)
                cid = (o or {}).get("clientOrderId", "") or ""
                if not str(cid).startswith(str(client_prefix)):
                    continue

            f_qty = float(f.get("qty") or 0.0)
            f_quote = float(f.get("quoteQty") or 0.0)
            is_buyer = bool(f.get("isBuyer"))

            if is_buyer:
                qty += f_qty
                cost += f_quote
                avg = (cost / qty) if qty > 1e-12 else 0.0
            else:
                if qty <= 1e-12:
                    continue
                sold = min(qty, f_qty)
                cost -= avg * sold
                qty -= sold
                if qty <= 1e-12:
                    qty = 0.0
                    cost = 0.0
                    avg = 0.0
                else:
                    avg = cost / qty

        return PositionSnapshot(symbol=symbol, btc_qty=float(qty), avg_entry=float(avg), cost_basis_usdt=float(cost))

    def compute_trailing_stats(self, symbol: str, tag: str = "TRAIL") -> Dict[str, float]:
        qty = 0.0
        cost = 0.0
        profits: List[float] = []

        def _trade_key(t: Dict[str, Any]) -> tuple[int, int]:
            return (int(t.get("time") or 0), int(t.get("id") or 0))

        for f in sorted(self._fills, key=_trade_key):
            if f.get("symbol") != symbol:
                continue
            is_buyer = bool(f.get("isBuyer"))
            f_qty = float(f.get("qty") or 0.0)
            f_quote = float(f.get("quoteQty") or 0.0)
            if f_qty <= 0:
                continue

            if is_buyer:
                qty += f_qty
                cost += f_quote
                continue

            if qty <= 1e-12:
                continue

            avg = cost / qty if qty > 0 else 0.0
            sold = min(qty, f_qty)
            profit = f_quote - (avg * sold)

            oid = int(f.get("orderId") or 0)
            o = self._orders.get(oid) or {}
            cid = (o.get("clientOrderId") or "")
            if tag in str(cid):
                profits.append(profit)

            cost -= avg * sold
            qty -= sold
            if qty <= 1e-12:
                qty = 0.0
                cost = 0.0

        count = float(len(profits))
        total = float(sum(profits)) if profits else 0.0
        avg_profit = float(total / count) if count > 0 else 0.0
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        win_rate = float(len(wins) / count) if count > 0 else 0.0
        avg_win = float(sum(wins) / len(wins)) if wins else 0.0
        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0

        return {
            "trailing_sell_count": count,
            "trailing_profit_total": total,
            "trailing_profit_avg": avg_profit,
            "trailing_win_rate": win_rate,
            "trailing_win_avg": avg_win,
            "trailing_loss_avg": avg_loss,
        }

    def recompute_position_from_fills(self, symbol: str) -> PositionSnapshot:
        snap = self._recompute_position(symbol, client_prefix=None)
        self._pos_by_symbol[symbol] = snap
        return snap

    def recompute_pool_position_from_fills(self, symbol: str, client_prefix: str = "BTCSTACK_") -> PositionSnapshot:
        snap = self._recompute_position(symbol, client_prefix=client_prefix)
        self._pool_pos_by_symbol[symbol] = snap
        return snap

    def get_position(self, symbol: str) -> PositionSnapshot:
        return self._pos_by_symbol.get(symbol, PositionSnapshot(symbol, 0.0, 0.0, 0.0))

    def get_pool_position(self, symbol: str) -> PositionSnapshot:
        return self._pool_pos_by_symbol.get(symbol, PositionSnapshot(symbol, 0.0, 0.0, 0.0))


def run_backtest(df: pd.DataFrame, *, cfg: BTCBacktestConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from btc_trader import BTCStackingTrader

    df = df.copy()
    if "dt" not in df.columns:
        raise RuntimeError("df must have 'dt' column")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    client = BacktestClient(
        symbol=cfg.symbol,
        init_usdt=float(cfg.init_usdt),
        init_btc=float(cfg.init_btc),
        filters=SymbolFilters(
            tick_size=float(cfg.tick_size),
            step_size=float(cfg.step_size),
            min_notional=float(cfg.min_notional),
        ),
        taker_fee_rate=float(cfg.fee_bps_taker) / 10000.0,
        maker_fee_rate=float(cfg.fee_bps_maker) / 10000.0,
        slippage_bps=float(cfg.slippage_bps),
        quote_asset=cfg.quote_asset,
        base_asset=cfg.base_asset,
        client_order_prefix=cfg.client_order_prefix,
    )
    db = BacktestDB(verbose=bool(cfg.verbose))

    # 첫 캔들 바 세팅
    dt0 = df.loc[0, "dt"]
    o0 = float(df.loc[0, "open"])
    h0 = float(df.loc[0, "high"])
    l0 = float(df.loc[0, "low"])
    c0 = float(df.loc[0, "close"])
    client.set_bar(ts=dt0.timestamp(), o=o0, h=h0, l=l0, c=c0)

    trader = BTCStackingTrader(cfg, client=client, db=db)
    trader.maybe_initial_market_entry()
    trader.sync_orders_and_fills()

    stats_after_init = client.get_stats()
    initial_executed = int(stats_after_init.get("market_buy_orders", 0)) > 0
    init_buy_spent = float(stats_after_init.get("buy_spent_total", 0.0))

    rows: List[Dict[str, Any]] = []
    first_buy_dt: Optional[pd.Timestamp] = None
    btc_at_first_buy: Optional[float] = None

    for i in range(len(df)):
        dt = df.loc[i, "dt"]
        o = float(df.loc[i, "open"])
        h = float(df.loc[i, "high"])
        l = float(df.loc[i, "low"])
        c = float(df.loc[i, "close"])

        client.set_bar(ts=dt.timestamp(), o=o, h=h, l=l, c=c)
        client.match_open_orders()
        trader.step()

        bal_u = client.get_balance(cfg.quote_asset)
        bal_b = client.get_balance(cfg.base_asset)

        usdt_total = float(bal_u["total"])
        btc_total = float(bal_b["total"])
        equity_usdt = usdt_total + btc_total * c

        if first_buy_dt is None and btc_total > 0:
            first_buy_dt = dt
            btc_at_first_buy = btc_total

        rows.append(
            {
                "dt": dt,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "usdt_total": usdt_total,
                "btc_total": btc_total,
                "equity_usdt": equity_usdt,
            }
        )

    out = pd.DataFrame(rows)

    stats = client.get_stats()

    total_buy_orders = int(stats.get("market_buy_orders", 0))
    total_buy_spent = float(stats.get("buy_spent_total", 0.0))

    grid_buy_count = max(0, total_buy_orders - (1 if initial_executed else 0))
    grid_buy_spent_total = max(0.0, total_buy_spent - (init_buy_spent if initial_executed else 0.0))

    # 매도 횟수 = TP 지정가 체결 횟수
    sell_count = int(stats.get("tp_limit_fills", 0))

    # 마지막 종가/지표들
    first_close = float(out["close"].iloc[0])
    last_close = float(out["close"].iloc[-1])
    btc_start = (float(cfg.init_usdt) / first_close + float(cfg.init_btc)) if first_close > 0 else 0.0
    equity_end_usdt = float(out["equity_usdt"].iloc[-1])
    btc_end = (equity_end_usdt / last_close) if last_close > 0 else 0.0

    btc_delta = btc_end - btc_start
    btc_delta_pct = (btc_delta / btc_start * 100.0) if btc_start > 0 else None
    btc_equiv_end = btc_end

    trailing_stats = db.compute_trailing_stats(cfg.symbol)
    summary = {
        "first_buy_dt": str(first_buy_dt) if first_buy_dt is not None else None,
        "end_dt": str(out["dt"].iloc[-1]),

        "btc_start": btc_start,
        "btc_end": btc_end,
        "equity_end_usdt": equity_end_usdt,
        "reserve_btc_qty": float(db.get_setting(cfg.reserve_btc_key, 0.0) or 0.0),
        "reserve_cost_usdt": float(db.get_setting("reserve_cost_usdt", 0.0) or 0.0),
        "fee_total": float(stats.get("fee_total", 0.0)),

        "grid_buy_count": int(grid_buy_count),
        "grid_buy_spent_total": float(grid_buy_spent_total),
        "sell_count": int(sell_count),

        "btc_start_after_first_buy": float(btc_at_first_buy or 0.0),
        "btc_delta": float(btc_delta),
        "btc_delta_pct": btc_delta_pct,
        "btc_equiv_end": float(btc_equiv_end),
        **trailing_stats,

        "trades": {
            "market_buys": int(stats.get("market_buy_orders", 0)),
            "tp_limit_orders": int(stats.get("tp_limit_orders", 0)),
            "tp_fills": int(stats.get("tp_limit_fills", 0)),
            "cancels": int(stats.get("cancels", 0)),
        },
    }
    for k, v in trailing_stats.items():
        out[k] = v
    return out, summary
