# btc_client.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

from binance.client import Client


@dataclass(frozen=True)
class SymbolFilters:
    step_size: float
    tick_size: float
    min_notional: float


class BTCClient:
    """
    ✅ v2: init 호출 형태가 섞여도 안전하게 동작하도록 통일
    지원:
      - BTCClient(cfg)
      - BTCClient(api_key, api_secret)
      - BTCClient(api_key, api_secret, symbol)
      - BTCClient(api_key=..., api_secret=..., symbol=...)
    """

    def __init__(
        self,
        api_key: Optional[Any] = None,
        api_secret: str = "",
        symbol: str = "BTCUSDT",
    ):
        # 1) BTCConfig 객체가 들어온 경우
        if api_key is not None and hasattr(api_key, "binance_api_key") and hasattr(api_key, "binance_api_secret"):
            cfg = api_key
            self.api_key = getattr(cfg, "binance_api_key", "") or ""
            self.api_secret = getattr(cfg, "binance_api_secret", "") or ""
            self.symbol = getattr(cfg, "symbol", symbol) or symbol
        else:
            # 2) 일반 인자 케이스
            self.api_key = api_key or ""
            self.api_secret = api_secret or ""
            self.symbol = symbol or "BTCUSDT"

        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)

        self._filters_cache: Dict[str, SymbolFilters] = {}

    # -----------------------------
    # Market data
    # -----------------------------
    def get_price(self, symbol: Optional[str] = None) -> float:
        sym = symbol or self.symbol
        ticker = self.client.get_symbol_ticker(symbol=sym)
        return float(ticker["price"])

    # -----------------------------
    # Exchange filters
    # -----------------------------
    def get_symbol_filters(self, symbol: Optional[str] = None) -> SymbolFilters:
        sym = symbol or self.symbol
        if sym in self._filters_cache:
            return self._filters_cache[sym]

        info = self.client.get_symbol_info(sym)
        if not info:
            raise RuntimeError(f"symbol info not found: {sym}")

        step_size = 0.0
        tick_size = 0.0
        min_notional = 0.0

        for f in info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = float(f.get("stepSize", "0"))
            elif f.get("filterType") == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", "0"))
            elif f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("minNotional", "0"))

        out = SymbolFilters(step_size=step_size, tick_size=tick_size, min_notional=min_notional)
        self._filters_cache[sym] = out
        return out

    def adjust_qty(self, qty: float, symbol: Optional[str] = None) -> float:
        sym = symbol or self.symbol
        f = self.get_symbol_filters(sym)
        if f.step_size <= 0:
            return float(qty)

        step = f.step_size
        # floor to step
        adj = (float(qty) // step) * step
        # float rounding noise 제거
        return float(f"{adj:.12f}")

    def adjust_price(self, price: float, symbol: Optional[str] = None) -> float:
        sym = symbol or self.symbol
        f = self.get_symbol_filters(sym)
        if f.tick_size <= 0:
            return float(price)

        tick = f.tick_size
        adj = (float(price) // tick) * tick
        return float(f"{adj:.12f}")

    # -----------------------------
    # Orders
    # -----------------------------
    def get_open_orders(self, symbol: Optional[str] = None):
        sym = symbol or self.symbol
        return self.client.get_open_orders(symbol=sym)

    def get_order(self, order_id: int, symbol: Optional[str] = None):
        sym = symbol or self.symbol
        return self.client.get_order(symbol=sym, orderId=int(order_id))

    def cancel_order(self, order_id: int, symbol: Optional[str] = None):
        sym = symbol or self.symbol
        return self.client.cancel_order(symbol=sym, orderId=int(order_id))

    def get_my_trades(self, symbol: Optional[str] = None, limit: int = 1000):
        sym = symbol or self.symbol
        return self.client.get_my_trades(symbol=sym, limit=int(limit))

    def _make_client_oid(self, tag: str = "GEN") -> str:
        # Binance clientOrderId 제한(보통 32~36자) 걸릴 수 있으니 짧게
        return f"BTCSTACK_{tag}_{int(time.time()*1000)}"

    def place_market_buy_by_quote(self, quote_usdt: float, symbol: Optional[str] = None, client_oid: Optional[str] = None):
        sym = symbol or self.symbol
        coid = client_oid or self._make_client_oid("BUY")
        return self.client.order_market_buy(
            symbol=sym,
            quoteOrderQty=f"{float(quote_usdt):.8f}",
            newClientOrderId=coid,
        )

    def place_limit_sell(self, qty_base: float, price: float, symbol: Optional[str] = None, client_oid: Optional[str] = None):
        sym = symbol or self.symbol
        qty = self.adjust_qty(qty_base, sym)
        px = self.adjust_price(price, sym)
        coid = client_oid or self._make_client_oid("TP")
        return self.client.order_limit_sell(
            symbol=sym,
            quantity=f"{qty:.8f}",
            price=f"{px:.8f}",
            timeInForce="GTC",
            newClientOrderId=coid,
        )

    def get_balance(self, asset: str) -> dict:
        """
        Spot 기준 asset 잔고 조회.
        return 예: {"free": 123.4, "locked": 5.6, "total": 129.0}
        """
        b = self.client.get_asset_balance(asset=asset)
        if not b:
            return {"free": 0.0, "locked": 0.0, "total": 0.0}

        free = float(b.get("free", 0.0) or 0.0)
        locked = float(b.get("locked", 0.0) or 0.0)
        return {"free": free, "locked": locked, "total": free + locked}

    def get_free_balance(self, asset: str) -> float:
        return float(self.get_balance(asset).get("free", 0.0))