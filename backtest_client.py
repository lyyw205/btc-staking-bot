# btc_backtest_client.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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
    """

    def __init__(
        self,
        symbol: str,
        *,
        init_usdt: float = 1000.0,
        init_btc: float = 0.0,
        filters: Optional[SymbolFilters] = None,
        taker_fee_rate: float = 0.001,   # 0.1%
        maker_fee_rate: float = 0.001,   # 단순화: maker도 동일
        slippage_bps: float = 0.0,       # 시장가 슬리피지 (bps)
        quote_asset: str = "USDT",
        base_asset: str = "BTC",
    ):
        self.symbol = symbol
        self.quote_asset = quote_asset
        self.base_asset = base_asset

        self.filters = filters or SymbolFilters()
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)
        self.slippage_bps = float(slippage_bps)

        # 잔고 (free/locked 단순 모델)
        self._quote_free = float(init_usdt)
        self._quote_locked = 0.0
        self._base_free = float(init_btc)
        self._base_locked = 0.0

        # 현재 봉 상태
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

    # ----------------------------
    # bar control
    # ----------------------------
    def set_bar(self, ts: float, o: float, h: float, l: float, c: float):
        self.cur_ts = float(ts)
        self.cur_open = float(o)
        self.cur_high = float(h)
        self.cur_low = float(l)
        self.cur_close = float(c)

    def match_open_orders(self):
        """
        현재 봉의 high/low를 이용해 limit SELL 체결 처리.
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

            # 체결 조건: high >= limit_price
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
        # tick rounding
        return float(_round_to_tick(price, f.tick_size))

    def adjust_qty(self, qty: float, symbol: str) -> float:
        f = self.get_symbol_filters(symbol)
        # step flooring
        return float(_floor_to_step(qty, f.step_size))

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        assert symbol == self.symbol
        out = []
        for o in self._orders.values():
            if o.get("symbol") != symbol:
                continue
            if o.get("status") in ("NEW", "PARTIALLY_FILLED"):
                out.append(o.copy())
        return out

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

        # lock 해제(SELL LIMIT만 lock)
        if o.get("type") == "LIMIT" and o.get("side") == "SELL":
            qty = float(o.get("origQty") or 0.0)
            self._base_locked = max(0.0, self._base_locked - qty)
            self._base_free = self._base_free + qty

        o["status"] = "CANCELED"
        self._orders[oid] = o
        return o.copy()

    def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        assert symbol == self.symbol
        # 마지막 이후 새 fill만 반환
        new = self._fills[self._fills_cursor :]
        self._fills_cursor = len(self._fills)
        return [t.copy() for t in new]

    # ----------------------------
    # place orders
    # ----------------------------
    def _new_order_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id

    def place_market_buy_by_quote(self, *, quote_usdt: float, symbol: str, clientOrderId: Optional[str] = None) -> Dict[str, Any]:
        assert symbol == self.symbol
        q = float(quote_usdt)
        if q <= 0:
            raise ValueError("quote_usdt must be > 0")

        # spendable check
        if q > self._quote_free + 1e-12:
            raise ValueError(f"insufficient {self.quote_asset} free: need={q}, have={self._quote_free}")

        # fill price = close * (1 + slippage)
        px = float(self.cur_close) * (1.0 + self.slippage_bps / 10000.0)
        px = self.adjust_price(px, symbol)

        # taker fee in quote: q * fee
        fee_q = q * self.taker_fee_rate
        q_net = max(0.0, q - fee_q)

        qty = q_net / px if px > 0 else 0.0
        qty = self.adjust_qty(qty, symbol)

        notional = qty * px
        if notional < max(self.filters.min_notional, 0.0):
            raise ValueError(f"min_notional not met: notional={notional:.4f}")

        # 실제로 쓴 quote는 qty*px + fee(단순화: fee_q를 그대로 사용)
        # qty를 step에 맞추며 notional이 줄어들 수 있으니, spent를 맞춰 재계산
        spent = qty * px
        fee_q = spent * self.taker_fee_rate
        total_spent = spent + fee_q
        if total_spent > self._quote_free + 1e-12:
            raise ValueError("insufficient free after rounding")

        oid = self._new_order_id()
        order = {
            "symbol": symbol,
            "orderId": oid,
            "clientOrderId": clientOrderId or f"BT_{oid}",
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

        # balances
        self._quote_free -= total_spent
        self._base_free += qty

        # fill record (Binance myTrades 느낌)
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

    def place_limit_sell(self, *, qty_base: float, price: float, symbol: str, clientOrderId: Optional[str] = None) -> Dict[str, Any]:
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

        # lock base
        if qty > self._base_free + 1e-12:
            raise ValueError(f"insufficient {self.base_asset} free: need={qty}, have={self._base_free}")
        self._base_free -= qty
        self._base_locked += qty

        oid = self._new_order_id()
        order = {
            "symbol": symbol,
            "orderId": oid,
            "clientOrderId": clientOrderId or f"BT_{oid}",
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

        # balances: release locked base, add quote
        self._base_locked = max(0.0, self._base_locked - qty)
        self._quote_free += net

        # order update
        o["status"] = "FILLED"
        o["executedQty"] = f"{qty:.8f}"
        o["cummulativeQuoteQty"] = f"{gross:.8f}"
        self._orders[int(oid)] = o

        # fill
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
