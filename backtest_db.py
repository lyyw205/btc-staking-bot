# btc_backtest_db.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
    - 설정(settings) 저장
    - fills 기반 포지션 재계산(전체/POOL)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._settings: Dict[str, Any] = {}

        self._orders: Dict[int, Dict[str, Any]] = {}  # orderId -> order
        self._fills: List[Dict[str, Any]] = []        # trade fills
        self._ai_events: List[Dict[str, Any]] = []

        self._pos_by_symbol: Dict[str, PositionSnapshot] = {}
        self._pool_pos_by_symbol: Dict[str, PositionSnapshot] = {}

    # ----------------------------
    # logging/settings
    # ----------------------------
    def log(self, level: str, msg: str):
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
        # trader가 넘방향이든 상관없이 raw 저장
        rec = dict(t)
        rec["orderId"] = int(order_id)
        self._fills.append(rec)

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
        - BUY: cost_basis += quoteQty(= gross)   (수수료는 fill의 commission에 있지만 단순화)
        - SELL: cost_basis -= avg_entry * qty_sold (즉, 원가만 감소)
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
                # SELL: 원가만 비례 감소 (실현손익은 별도 추적 안 함)
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
