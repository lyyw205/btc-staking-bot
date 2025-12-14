# btc_backtest.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd


# -----------------------------
# Filters (Binance BTCUSDT 기본에 맞춘 최소치)
# -----------------------------
@dataclass
class SymbolFilters:
    tick_size: float = 0.10
    step_size: float = 0.000001
    min_notional: float = 5.0


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(float(x) / step) * step


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick


# -----------------------------
# In-memory DB (BTCDB 최소 인터페이스)
# -----------------------------
@dataclass
class PositionSnapshot:
    btc_qty: float = 0.0
    avg_entry: float = 0.0
    cost_basis_usdt: float = 0.0


class BacktestDB:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.settings: Dict[str, Any] = {}
        self.orders: Dict[int, Dict[str, Any]] = {}
        self.fills: List[Dict[str, Any]] = []  # {id, orderId, side, qty, price, ts, commission, commissionAsset}
        self.ai_events: List[Dict[str, Any]] = []
        self.stats = defaultdict(int)
        self._printed = 0

    def log(self, level: str, msg: str):
        # 너무 많이 찍히면 백테스트가 “멈춘 것처럼” 보여서 제한
        if level in ("ERROR", "WARN"):
            print(f"[{level}] {msg}")
            return
        if self.verbose:
            self._printed += 1
            if self._printed % 200 == 0:
                print(f"[{level}] {msg}")

    def get_setting(self, k: str, default=None):
        return self.settings.get(k, default)

    def set_setting(self, k: str, v: Any):
        self.settings[k] = v

    def insert_ai_event(self, kind: str, symbol: str, payload: Dict[str, Any]):
        self.ai_events.append({"kind": kind, "symbol": symbol, "payload": payload})

    def upsert_order(self, o: Dict[str, Any]):
        oid = int(o["orderId"])
        self.orders[oid] = dict(o)

    def get_recent_open_orders(self, limit: int = 50) -> List[int]:
        out = []
        for oid, o in self.orders.items():
            if o.get("status") in ("NEW", "PARTIALLY_FILLED"):
                out.append(int(oid))
        out.sort(reverse=True)
        return out[:limit]

    def insert_fill(self, order_id: int, t: Dict[str, Any]):
        # trade id로 중복 방지
        tid = int(t.get("id", 0))
        if tid and any(int(x.get("id", 0)) == tid for x in self.fills):
            return
        self.fills.append(dict(t))

    def _recompute_from_fills(self, *, symbol: str, only_prefix: Optional[str] = None) -> PositionSnapshot:
        qty = 0.0
        cost = 0.0
        avg = 0.0

        def ok_fill(f):
            if only_prefix is None:
                return True
            oid = int(f.get("orderId", 0))
            o = self.orders.get(oid, {})
            coid = str(o.get("clientOrderId", ""))
            return coid.startswith(only_prefix)

        for f in self.fills:
            if not ok_fill(f):
                continue
            side = str(f.get("side", "")).upper()
            q = float(f.get("qty", 0.0) or 0.0)
            p = float(f.get("price", 0.0) or 0.0)
            fee = float(f.get("commission", 0.0) or 0.0)
            fee_asset = str(f.get("commissionAsset", "USDT"))

            # fee를 USDT로만 처리(백테스트 단순화)
            fee_usdt = fee if fee_asset.upper() == "USDT" else 0.0

            if side == "BUY":
                new_cost = cost + (q * p) + fee_usdt
                new_qty = qty + q
                qty, cost = new_qty, new_cost
                avg = (cost / qty) if qty > 0 else 0.0

            elif side == "SELL":
                if qty <= 0:
                    continue
                sell_qty = min(qty, q)
                reduce_cost = avg * sell_qty
                qty = qty - sell_qty
                cost = max(0.0, cost - reduce_cost)
                avg = (cost / qty) if qty > 0 else 0.0

        if qty <= 1e-12:
            return PositionSnapshot(0.0, 0.0, 0.0)
        return PositionSnapshot(float(qty), float(avg), float(cost))

    def recompute_position_from_fills(self, symbol: str) -> PositionSnapshot:
        return self._recompute_from_fills(symbol=symbol, only_prefix=None)

    def recompute_pool_position_from_fills(self, symbol: str, client_prefix: str = "BTCSTACK_") -> PositionSnapshot:
        snap = self._recompute_from_fills(symbol=symbol, only_prefix=client_prefix)
        self.settings["_pool_pos"] = snap
        return snap

    def get_position(self, symbol: str) -> PositionSnapshot:
        return self.recompute_position_from_fills(symbol)

    def get_pool_position(self, symbol: str) -> PositionSnapshot:
        return self.settings.get("_pool_pos") or PositionSnapshot()


# -----------------------------
# Backtest Client (BTCClient 최소 인터페이스)
# -----------------------------
class BacktestClient:
    """
    - market buy: 현재 close에서 체결(슬리피지/수수료 적용)
    - limit sell(TP): 다음 캔들에서 high가 price 이상이면 체결(lookahead 방지)
    """

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        quote_asset: str = "USDT",
        base_asset: str = "BTC",
        init_usdt: float = 1000.0,
        slippage_bps: float = 1.0,
        fee_bps_taker: float = 10.0,
        fee_bps_maker: float = 2.0,
    ):
        self.symbol = symbol
        self.quote_asset = quote_asset
        self.base_asset = base_asset

        self.filters = SymbolFilters()

        self.slippage_bps = float(slippage_bps)
        self.fee_bps_taker = float(fee_bps_taker)
        self.fee_bps_maker = float(fee_bps_maker)

        # ✅ stats (반드시 존재)
        self.stats = defaultdict(int)

        # balances
        self._free: Dict[str, float] = {quote_asset: float(init_usdt), base_asset: 0.0}
        self._locked: Dict[str, float] = {quote_asset: 0.0, base_asset: 0.0}

        # orders/trades
        self._oid = 1000
        self._tid = 5000
        self._orders: Dict[int, Dict[str, Any]] = {}
        self._trades: List[Dict[str, Any]] = []

        # current candle
        self._cur_price = 0.0
        self._cur_ts = 0.0

        # flags
        self.is_backtest = True

    def set_market(self, *, price: float, ts: float):
        self._cur_price = float(price)
        self._cur_ts = float(ts)

    def get_price(self, symbol: str) -> float:
        return float(self._cur_price)

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        return self.filters

    def adjust_price(self, price: float, symbol: str) -> float:
        return float(_round_to_tick(price, self.filters.tick_size))

    def adjust_qty(self, qty: float, symbol: str) -> float:
        return float(_floor_to_step(qty, self.filters.step_size))

    def get_balance(self, asset: str) -> Dict[str, float]:
        free = float(self._free.get(asset, 0.0))
        locked = float(self._locked.get(asset, 0.0))
        return {"free": free, "locked": locked, "total": free + locked}

    def _new_order_id(self) -> int:
        self._oid += 1
        return self._oid

    def _new_trade_id(self) -> int:
        self._tid += 1
        return self._tid

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        out = []
        for o in self._orders.values():
            if o.get("status") in ("NEW", "PARTIALLY_FILLED"):
                out.append(dict(o))
        return out

    def get_order(self, order_id: int, symbol: str) -> Dict[str, Any]:
        return dict(self._orders[int(order_id)])

    def cancel_order(self, order_id: int, symbol: str):
        oid = int(order_id)
        o = self._orders.get(oid)
        self.stats["cancels"] += 1
        if not o:
            return
        if o.get("status") in ("NEW", "PARTIALLY_FILLED"):
            # locked 해제
            if o.get("side") == "SELL" and o.get("type") == "LIMIT":
                q = float(o.get("origQty", 0.0) or 0.0)
                self._locked[self.base_asset] = max(0.0, self._locked[self.base_asset] - q)
                self._free[self.base_asset] += q
            o["status"] = "CANCELED"

    def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        # 누적 반환(DB에서 trade id로 중복 제거)
        return list(self._trades)

    def place_market_buy_by_quote(self, *, quote_usdt: float, symbol: str) -> Dict[str, Any]:
        quote = float(quote_usdt)
        if quote <= 0:
            raise RuntimeError("quote_usdt must be > 0")
        if self._free[self.quote_asset] + 1e-12 < quote:
            raise RuntimeError(f"insufficient {self.quote_asset}: free={self._free[self.quote_asset]:.2f} need={quote:.2f}")

        px = float(self._cur_price) * (1.0 + self.slippage_bps / 10000.0)
        fee = quote * (self.fee_bps_taker / 10000.0)  # USDT fee
        net_quote = max(0.0, quote - fee)

        qty = net_quote / px
        qty = self.adjust_qty(qty, symbol)
        if qty <= 0:
            raise RuntimeError("qty too small after adjust")

        # spend exactly quote (simplified)
        self._free[self.quote_asset] -= quote
        self._free[self.base_asset] += qty

        self.stats["market_buy_orders"] += 1

        oid = self._new_order_id()
        coid = f"BTCSTACK_{oid}"
        o = {
            "orderId": oid,
            "clientOrderId": coid,
            "symbol": symbol,
            "status": "FILLED",
            "type": "MARKET",
            "side": "BUY",
            "price": str(px),
            "origQty": str(qty),
            "executedQty": str(qty),
            "time": int(self._cur_ts * 1000),
        }
        self._orders[oid] = o

        tid = self._new_trade_id()
        t = {
            "id": tid,
            "orderId": oid,
            "symbol": symbol,
            "side": "BUY",
            "qty": float(qty),
            "price": float(px),
            "time": int(self._cur_ts * 1000),
            "commission": float(fee),
            "commissionAsset": "USDT",
        }
        self._trades.append(t)
        return dict(o)

    def get_stats(self) -> dict:
        return dict(self.stats)

    def place_limit_sell(self, *, qty_base: float, price: float, symbol: str) -> Dict[str, Any]:
        self.stats["tp_limit_orders"] += 1

        qty = self.adjust_qty(float(qty_base), symbol)
        px = self.adjust_price(float(price), symbol)

        if qty <= 0:
            raise RuntimeError("qty too small")
        notional = qty * px
        if notional < self.filters.min_notional:
            raise RuntimeError(f"notional too small: {notional:.2f} < {self.filters.min_notional:.2f}")

        # ✅ 현실처럼 “BTC 락”이 필요 (TP가 실제 보유분만 걸리게)
        if self._free[self.base_asset] + 1e-12 < qty:
            raise RuntimeError(f"insufficient {self.base_asset} to place TP: free={self._free[self.base_asset]:.8f} need={qty:.8f}")

        self._free[self.base_asset] -= qty
        self._locked[self.base_asset] += qty

        oid = self._new_order_id()
        coid = f"BTCSTACK_{oid}"
        o = {
            "orderId": oid,
            "clientOrderId": coid,
            "symbol": symbol,
            "status": "NEW",
            "type": "LIMIT",
            "side": "SELL",
            "price": str(px),
            "origQty": str(qty),
            "executedQty": "0",
            "time": int(self._cur_ts * 1000),
        }
        self._orders[oid] = o
        return dict(o)

    def match_orders_on_candle(self, *, high: float, low: float):
        """
        lookahead 방지:
        - 이 함수는 “캔들 시작 시점”에 호출한다고 가정
        - 따라서 '직전 step에서 걸린 주문'만 이번 캔들의 high/low로 체결 판정
        """
        H = float(high)

        for oid, o in list(self._orders.items()):
            if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
                continue
            if o.get("type") != "LIMIT":
                continue
            if o.get("side") != "SELL":
                continue

            px = float(o.get("price", 0.0) or 0.0)
            qty = float(o.get("origQty", 0.0) or 0.0)

            # limit sell: high가 price 이상이면 체결로 간주
            if H + 1e-12 >= px:
                fee = (qty * px) * (self.fee_bps_maker / 10000.0)  # USDT fee
                proceeds = (qty * px) - fee

                # locked BTC 해제(=팔림)
                self._locked[self.base_asset] = max(0.0, self._locked[self.base_asset] - qty)
                self._free[self.quote_asset] += proceeds

                o["status"] = "FILLED"
                o["executedQty"] = str(qty)

                self.stats["tp_limit_fills"] += 1

                tid = self._new_trade_id()
                t = {
                    "id": tid,
                    "orderId": int(oid),
                    "symbol": self.symbol,
                    "side": "SELL",
                    "qty": float(qty),
                    "price": float(px),
                    "time": int(self._cur_ts * 1000),
                    "commission": float(fee),
                    "commissionAsset": "USDT",
                }
                self._trades.append(t)


# -----------------------------
# Main backtest runner
# -----------------------------
def run_backtest(
    df: pd.DataFrame,
    *,
    cfg,
    init_usdt: float = 1000.0,
    slippage_bps: float = 1.0,
    fee_bps_taker: float = 10.0,
    fee_bps_maker: float = 2.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    df: columns = dt, open, high, low, close, volume
    return: (equity dataframe, summary dict)
    """
    from btc_trader import BTCStackingTrader  # 실매매 로직 그대로 사용

    df = df.copy()
    if "dt" not in df.columns:
        raise RuntimeError("df must have 'dt' column")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    client = BacktestClient(
        symbol=getattr(cfg, "symbol", "BTCUSDT"),
        quote_asset=getattr(cfg, "quote_asset", "USDT") if hasattr(cfg, "quote_asset") else "USDT",
        base_asset=getattr(cfg, "base_asset", "BTC") if hasattr(cfg, "base_asset") else "BTC",
        init_usdt=float(init_usdt),
        slippage_bps=float(slippage_bps),
        fee_bps_taker=float(fee_bps_taker),
        fee_bps_maker=float(fee_bps_maker),
    )
    db = BacktestDB(verbose=False)

    # 70/30 기준 총액 고정
    db.set_setting("usdt_ref_total", float(init_usdt))

    # base_price 초기값
    db.set_setting("base_price", float(df.loc[0, "close"]))

    trader = BTCStackingTrader(cfg, client=client, db=db)

    rows: List[Dict[str, Any]] = []
    first_buy_dt: Optional[pd.Timestamp] = None
    btc_at_first_buy: Optional[float] = None

    for i in range(len(df)):
        dt = df.loc[i, "dt"]
        o = float(df.loc[i, "open"])
        h = float(df.loc[i, "high"])
        l = float(df.loc[i, "low"])
        c = float(df.loc[i, "close"])

        ts = dt.timestamp()
        client.set_market(price=c, ts=ts)

        # 이전 주문을 이번 캔들의 high/low로 체결 처리
        client.match_orders_on_candle(high=h, low=l)

        trader.step()

        bal_u = client.get_balance(client.quote_asset)
        bal_b = client.get_balance(client.base_asset)

        usdt_total = float(bal_u["total"])
        btc_total = float(bal_b["total"])
        equity_usdt = usdt_total + btc_total * c
        btc_equiv = btc_total + (usdt_total / c if c > 0 else 0.0)

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
                "btc_equiv": btc_equiv,
            }
        )

    out = pd.DataFrame(rows)

    end_dt = out["dt"].iloc[-1]
    btc_end = float(out["btc_total"].iloc[-1])
    btc_start = float(btc_at_first_buy or 0.0)

    summary: Dict[str, Any] = {
        "start_usdt": float(init_usdt),
        "first_buy_dt": str(first_buy_dt) if first_buy_dt is not None else None,
        "end_dt": str(end_dt),
        "btc_start_after_first_buy": btc_start,
        "btc_end": btc_end,
        "btc_delta": btc_end - btc_start,
        "btc_delta_pct": ((btc_end - btc_start) / btc_start * 100.0) if btc_start > 0 else None,
        "equity_end_usdt": float(out["equity_usdt"].iloc[-1]),
        "btc_equiv_end": float(out["btc_equiv"].iloc[-1]),
        "stats": client.get_stats() if hasattr(client, "get_stats") else {},
    }

    return out, summary
