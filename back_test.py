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

    # Symbol
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    # Backtest Account Assumptions
    init_usdt: float = 100000.0
    init_btc: float = 0.0
    initial_buy_on_start: bool = True
    initial_buy_ratio: float = 0.30
    initial_buy_usdt: Optional[float] = None
    initial_entry_usdt: Optional[float] = None
    initial_core_usdt: float = 2000.0
    reserve_btc_key: str = "reserve_btc_qty"

    # Execution Assumptions
    slippage_bps: float = 1.0
    fee_bps_taker: float = 10.0
    fee_bps_maker: float = 2.0

    # Exchange Filters
    tick_size: float = 0.1
    step_size: float = 0.000001
    min_notional: float = 0.0

    # Order ID Prefix (pool tagging)
    client_order_prefix: str = "BTCSTACK_"

    # Lot stacking
    lot_buy_usdt: float = 100.0
    lot_tp_pct: float = 0.03
    lot_drop_pct: float = 0.01
    lot_prebuy_pct: float = 0.0015
    lot_cancel_rebound_pct: float = 0.004
    min_trade_usdt: float = 0.0

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
            symbol=s("SYMBOL", cls.symbol),
            base_asset=s("BASE_ASSET", cls.base_asset),
            quote_asset=s("QUOTE_ASSET", cls.quote_asset),

            init_usdt=f("INIT_USDT", cls.init_usdt),
            init_btc=f("INIT_BTC", cls.init_btc),
            initial_buy_on_start=b("INITIAL_BUY_ON_START", cls.initial_buy_on_start),
            initial_buy_ratio=f("INITIAL_BUY_RATIO", cls.initial_buy_ratio),
            initial_buy_usdt=f("INITIAL_BUY_USDT", cls.initial_buy_usdt) if os.getenv("BT_INITIAL_BUY_USDT") is not None else cls.initial_buy_usdt,
            initial_entry_usdt=f("INITIAL_ENTRY_USDT", cls.initial_entry_usdt) if os.getenv("BT_INITIAL_ENTRY_USDT") is not None else None,
            initial_core_usdt=f("INITIAL_CORE_USDT", cls.initial_core_usdt),
            reserve_btc_key=s("RESERVE_BTC_KEY", cls.reserve_btc_key),

            slippage_bps=f("SLIPPAGE_BPS", cls.slippage_bps),
            fee_bps_taker=f("FEE_BPS_TAKER", cls.fee_bps_taker),
            fee_bps_maker=f("FEE_BPS_MAKER", cls.fee_bps_maker),

            tick_size=f("TICK_SIZE", cls.tick_size),
            step_size=f("STEP_SIZE", cls.step_size),
            min_notional=f("MIN_NOTIONAL", cls.min_notional),

            client_order_prefix=s("CLIENT_ORDER_PREFIX", cls.client_order_prefix),

            order_cooldown_sec=i("ORDER_COOLDOWN_SEC", cls.order_cooldown_sec),
            loop_interval_sec=i("LOOP_INTERVAL_SEC", cls.loop_interval_sec),

            trade_cap_ratio=f("TRADE_CAP_RATIO", cls.trade_cap_ratio),
            usdt_reserve_buffer=f("USDT_RESERVE_BUFFER", cls.usdt_reserve_buffer),
            use_fixed_usdt_reference=b("USE_FIXED_USDT_REFERENCE", cls.use_fixed_usdt_reference),

            lot_buy_usdt=f("LOT_BUY_USDT", cls.lot_buy_usdt),
            lot_tp_pct=f("LOT_TP_PCT", cls.lot_tp_pct),
            lot_drop_pct=f("LOT_DROP_PCT", cls.lot_drop_pct),
            lot_prebuy_pct=f("LOT_PREBUY_PCT", cls.lot_prebuy_pct),
            lot_cancel_rebound_pct=f("LOT_CANCEL_REBOUND_PCT", cls.lot_cancel_rebound_pct),
            min_trade_usdt=f("MIN_TRADE_USDT", cls.min_trade_usdt),
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
        self.stats["market_sell_orders"] = 0.0
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
        현재 봉의 low/high를 이용해 limit BUY/SELL 체결 처리.
        (단순화: 조건 충족 시 전량 체결)
        """
        for oid, o in list(self._orders.items()):
            if o.get("symbol") != self.symbol:
                continue
            if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
                continue
            if o.get("type") != "LIMIT":
                continue

            price = float(o.get("price") or 0.0)
            qty = float(o.get("origQty") or 0.0)
            if price <= 0 or qty <= 0:
                continue

            if o.get("side") == "BUY" and self.cur_low <= price:
                self._fill_limit_buy(oid, price=price, qty=qty)
            elif o.get("side") == "SELL" and self.cur_high >= price:
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

        # LIMIT 주문 lock 해제
        if o.get("type") == "LIMIT":
            if o.get("side") == "SELL":
                qty = float(o.get("origQty") or 0.0)
                qty = max(0.0, qty)
                unlock = min(self._base_locked, qty)
                self._base_locked -= unlock
                self._base_free += unlock
            elif o.get("side") == "BUY":
                locked = float(o.get("lockedQuote") or 0.0)
                unlock = min(self._quote_locked, locked)
                self._quote_locked -= unlock
                self._quote_free += unlock

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

    def place_limit_buy_by_quote(
        self,
        *,
        quote_usdt: float,
        price: float,
        symbol: str,
        clientOrderId: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert symbol == self.symbol
        px = self.adjust_price(float(price), symbol)
        if px <= 0:
            raise ValueError("price must be > 0")

        qty = float(quote_usdt) / px
        qty = self.adjust_qty(qty, symbol)
        if qty <= 0:
            raise ValueError("qty must be > 0")

        notional = qty * px
        if notional < max(self.filters.min_notional, 0.0):
            raise ValueError(f"min_notional not met: notional={notional:.4f}")

        fee_q = notional * self.maker_fee_rate
        total = notional + fee_q
        if total > self._quote_free + 1e-12:
            raise ValueError(f"insufficient {self.quote_asset} free: need={total:.8f}, have={self._quote_free:.8f}")

        oid = self._new_order_id()
        coid = clientOrderId or f"{self.client_order_prefix}{oid}"

        self._quote_free -= total
        self._quote_locked += total

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
            "side": "BUY",
            "lockedQuote": f"{total:.8f}",
        }
        self._orders[oid] = order
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
        return order.copy()

    def _fill_limit_buy(self, oid: int, *, price: float, qty: float):
        o = self._orders.get(int(oid))
        if not o:
            return
        if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
            return

        px = self.adjust_price(price, self.symbol)
        qty = self.adjust_qty(qty, self.symbol)

        gross = qty * px
        fee_q = gross * self.maker_fee_rate
        total = gross + fee_q

        unlock = min(self._quote_locked, float(o.get("lockedQuote") or 0.0))
        self._quote_locked -= unlock
        self._quote_free += max(0.0, unlock - total)
        self._base_free += qty

        o["status"] = "FILLED"
        o["executedQty"] = f"{qty:.8f}"
        o["cummulativeQuoteQty"] = f"{gross:.8f}"
        self._orders[int(oid)] = o

        self.stats["fee_total"] += float(fee_q)
        self.stats["buy_spent_total"] += float(gross)
        self.stats["buy_fee_total"] += float(fee_q)

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
            "isBuyer": True,
            "isMaker": True,
        }
        self._fills.append(trade)
        self.stats["market_buy_orders"] += 1.0

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

        unlock = min(self._base_locked, qty)
        self._base_locked -= unlock
        self._quote_free += net

        o["status"] = "FILLED"
        o["executedQty"] = f"{qty:.8f}"
        o["cummulativeQuoteQty"] = f"{gross:.8f}"
        self._orders[int(oid)] = o

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
        self.stats["market_sell_orders"] += 1.0


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
        self._lots: List[Dict[str, Any]] = []
        self._next_lot_id = 1

        self._pos_by_symbol: Dict[str, PositionSnapshot] = {}

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
    # lots
    # ----------------------------
    def insert_lot(
        self,
        *,
        symbol: str,
        buy_price: float,
        buy_btc_qty: float,
        buy_time_ms: Optional[int] = None,
        status: str = "OPEN",
    ) -> int:
        lot_id = self._next_lot_id
        self._next_lot_id += 1
        rec = {
            "lot_id": lot_id,
            "symbol": symbol,
            "buy_price": float(buy_price),
            "buy_btc_qty": float(buy_btc_qty),
            "buy_time_ms": int(buy_time_ms) if buy_time_ms is not None else None,
            "status": status,
            "sell_order_id": None,
            "sell_order_time_ms": None,
            "sell_price": None,
            "sell_time_ms": None,
            "fee_usdt": None,
            "net_profit_usdt": None,
        }
        self._lots.append(rec)
        return lot_id

    def get_open_lots(self, symbol: str) -> List[Dict[str, Any]]:
        out = []
        for lot in self._lots:
            if lot.get("symbol") != symbol:
                continue
            if lot.get("status") != "OPEN":
                continue
            out.append(dict(lot))
        out.sort(key=lambda x: (x.get("buy_time_ms") or 0, x.get("lot_id") or 0))
        return out

    def close_lot(
        self,
        *,
        lot_id: int,
        sell_price: float,
        sell_time_ms: Optional[int] = None,
        fee_usdt: float = 0.0,
        net_profit_usdt: float = 0.0,
        status: str = "CLOSED",
    ):
        for lot in self._lots:
            if int(lot.get("lot_id") or 0) != int(lot_id):
                continue
            lot["status"] = status
            lot["sell_price"] = float(sell_price)
            lot["sell_time_ms"] = int(sell_time_ms) if sell_time_ms is not None else None
            lot["fee_usdt"] = float(fee_usdt)
            lot["net_profit_usdt"] = float(net_profit_usdt)
            break

    def set_lot_sell_order(self, *, lot_id: int, order_id: int, order_time_ms: Optional[int] = None):
        for lot in self._lots:
            if int(lot.get("lot_id") or 0) != int(lot_id):
                continue
            lot["sell_order_id"] = int(order_id)
            lot["sell_order_time_ms"] = int(order_time_ms) if order_time_ms is not None else None
            break

    def clear_lot_sell_order(self, *, lot_id: int):
        for lot in self._lots:
            if int(lot.get("lot_id") or 0) != int(lot_id):
                continue
            lot["sell_order_id"] = None
            lot["sell_order_time_ms"] = None
            break

    # ----------------------------
    # positions
    # ----------------------------
    def _recompute_position(self, symbol: str) -> PositionSnapshot:
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

    def recompute_position_from_fills(self, symbol: str) -> PositionSnapshot:
        snap = self._recompute_position(symbol)
        self._pos_by_symbol[symbol] = snap
        return snap

    def get_position(self, symbol: str) -> PositionSnapshot:
        return self._pos_by_symbol.get(symbol, PositionSnapshot(symbol, 0.0, 0.0, 0.0))


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
    bal_u_init = client.get_balance(cfg.quote_asset)
    bal_b_init = client.get_balance(cfg.base_asset)
    init_usdt_left = float(bal_u_init["total"])
    init_btc_total = float(bal_b_init["total"])
    init_buy_qty = max(0.0, init_btc_total - float(cfg.init_btc))

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

    market_sell_count = int(stats.get("market_sell_orders", 0))
    sell_count = market_sell_count

    # 마지막 종가/지표들
    first_close = float(out["close"].iloc[0])
    last_close = float(out["close"].iloc[-1])
    init_btc_equiv_total = (init_usdt_left / first_close + init_btc_total) if first_close > 0 else 0.0
    end_usdt_total = float(out["usdt_total"].iloc[-1])
    end_btc_total = float(out["btc_total"].iloc[-1])
    end_btc_equiv_total = (end_usdt_total / last_close + end_btc_total) if last_close > 0 else 0.0
    btc_equiv_delta = end_btc_equiv_total - init_btc_equiv_total
    btc_equiv_delta_pct = (btc_equiv_delta / init_btc_equiv_total * 100.0) if init_btc_equiv_total > 0 else None
    btc_start = (float(cfg.init_usdt) / first_close + float(cfg.init_btc)) if first_close > 0 else 0.0
    equity_end_usdt = float(out["equity_usdt"].iloc[-1])
    btc_end = (equity_end_usdt / last_close) if last_close > 0 else 0.0

    btc_delta = btc_end - btc_start
    btc_delta_pct = (btc_delta / btc_start * 100.0) if btc_start > 0 else None
    btc_equiv_end = btc_end
    total_buy_spent = float(stats.get("buy_spent_total", 0.0))
    total_sell_gross = float(stats.get("sell_gross_total", 0.0))
    total_sell_fee = float(stats.get("sell_fee_total", 0.0))
    total_sell_net = total_sell_gross - total_sell_fee
    add_buy_spent = max(0.0, total_buy_spent - (init_buy_spent if initial_executed else 0.0))
    end_bal = client.get_balance(cfg.quote_asset)
    end_usdt_free = float(end_bal.get("free", 0.0))
    end_usdt_locked = float(end_bal.get("locked", 0.0))
    core_btc_qty = float(db.get_setting(cfg.reserve_btc_key, 0.0) or 0.0)
    core_btc_initial = float(db.get_setting("core_btc_initial", 0.0) or 0.0)
    core_btc_added = core_btc_qty - core_btc_initial

    summary = {
        "first_buy_dt": str(first_buy_dt) if first_buy_dt is not None else None,
        "end_dt": str(out["dt"].iloc[-1]),

        "btc_start": btc_start,
        "btc_end": btc_end,
        "equity_end_usdt": equity_end_usdt,
        "reserve_btc_qty": core_btc_qty,
        "reserve_cost_usdt": float(db.get_setting("reserve_cost_usdt", 0.0) or 0.0),
        "core_bucket_usdt": float(db.get_setting("core_bucket_usdt", 0.0) or 0.0),
        "open_lots_count": int(len(db.get_open_lots(cfg.symbol))),
        "fee_total": float(stats.get("fee_total", 0.0)),
        "add_buy_spent_usdt": add_buy_spent,
        "sell_net_usdt": total_sell_net,
        "end_usdt_free": end_usdt_free,
        "end_usdt_locked": end_usdt_locked,
        "core_btc_initial": core_btc_initial,
        "core_btc_added": core_btc_added,

        "sell_count": int(sell_count),

        "btc_start_after_first_buy": float(btc_at_first_buy or 0.0),
        "btc_delta": float(btc_delta),
        "btc_delta_pct": btc_delta_pct,
        "btc_equiv_end": float(btc_equiv_end),
        "init_usdt_total": float(cfg.init_usdt),
        "init_buy_spent": float(init_buy_spent),
        "init_buy_qty": float(init_buy_qty),
        "init_usdt_left": float(init_usdt_left),
        "init_btc_total": float(init_btc_total),
        "init_btc_equiv_total": float(init_btc_equiv_total),
        "end_usdt_total": float(end_usdt_total),
        "end_btc_total": float(end_btc_total),
        "end_btc_equiv_total": float(end_btc_equiv_total),
        "btc_equiv_delta": float(btc_equiv_delta),
        "btc_equiv_delta_pct": btc_equiv_delta_pct,
        "trades": {
            "market_buys": int(stats.get("market_buy_orders", 0)),
            "market_sells": int(stats.get("market_sell_orders", 0)),
            "cancels": int(stats.get("cancels", 0)),
        },
        "stats": stats,
    }
    return out, summary
