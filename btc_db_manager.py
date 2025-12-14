# btc_db_manager.py (v2.1)
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class PositionSnapshot:
    symbol: str
    btc_qty: float
    cost_basis_usdt: float
    avg_entry: float
    updated_at: str


class BTCDB:
    def __init__(self, db_url: str, verbose: bool = True):
        self.db_url = db_url
        self.verbose = verbose
        self.init_db()

    # -----------------------------
    # connection / init
    # -----------------------------
    def get_connection(self):
        # db_url은 "postgresql://..." 형태 문자열이어야 함
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def init_db(self):
        """
        1) 테이블이 없으면 생성
        2) 이미 있으면: 없는 컬럼만 '자동 추가' (migration)
        """
        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # 1) create if not exists
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS btc_settings (
                          key TEXT PRIMARY KEY,
                          value TEXT,
                          updated_at TIMESTAMPTZ DEFAULT now()
                        );

                        CREATE TABLE IF NOT EXISTS btc_orders (
                          order_id BIGINT PRIMARY KEY,
                          symbol TEXT NOT NULL,
                          side TEXT,
                          type TEXT,
                          status TEXT,
                          price NUMERIC,
                          orig_qty NUMERIC,
                          executed_qty NUMERIC,
                          cum_quote_qty NUMERIC,
                          update_time_ms BIGINT,
                          raw_json JSONB,
                          updated_at TIMESTAMPTZ DEFAULT now()
                        );

                        CREATE TABLE IF NOT EXISTS btc_fills (
                          trade_id BIGINT PRIMARY KEY,
                          order_id BIGINT,
                          symbol TEXT NOT NULL,
                          side TEXT,
                          price NUMERIC,
                          qty NUMERIC,
                          quote_qty NUMERIC,
                          commission NUMERIC,
                          commission_asset TEXT,
                          trade_time_ms BIGINT,
                          raw_json JSONB,
                          inserted_at TIMESTAMPTZ DEFAULT now()
                        );

                        CREATE TABLE IF NOT EXISTS btc_position (
                          symbol TEXT PRIMARY KEY,
                          btc_qty NUMERIC NOT NULL,
                          cost_basis_usdt NUMERIC NOT NULL,
                          avg_entry NUMERIC NOT NULL,
                          updated_at TIMESTAMPTZ DEFAULT now()
                        );
                       

                        CREATE TABLE IF NOT EXISTS btc_pool_position (
                            symbol TEXT PRIMARY KEY,
                            btc_qty NUMERIC NOT NULL,
                            cost_basis_usdt NUMERIC NOT NULL,
                            avg_entry NUMERIC NOT NULL,
                            updated_at TIMESTAMPTZ DEFAULT now()
                            );

                        CREATE TABLE IF NOT EXISTS btc_ai_events (
                            id BIGSERIAL PRIMARY KEY,
                            kind TEXT NOT NULL,          -- 'REPORT' | 'GATE' | 'TUNE'
                            symbol TEXT NOT NULL,
                            payload JSONB,
                            inserted_at TIMESTAMPTZ DEFAULT now()
                            );
                        """
                    )

            # 2) migration: 기존 테이블이 있어도 "없는 컬럼만" 추가
            self._migrate_schema()
            # orders: client_order_id 추가
            self._add_col_if_missing("btc_orders", "client_order_id", "TEXT")
            # pool_position updated_at 보장
            self._add_col_if_missing("btc_pool_position", "updated_at", "TIMESTAMPTZ DEFAULT now()")

        finally:
            conn.close()

    # -----------------------------
    # schema helpers / migration
    # -----------------------------
    @lru_cache(maxsize=256)
    def _table_columns(self, table: str) -> set[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=%s
                    """,
                    (table,),
                )
                rows = cur.fetchall() or []
                return {r["column_name"] for r in rows}
        finally:
            conn.close()

    def _has_col(self, table: str, col: str) -> bool:
        return col in self._table_columns(table)

    def _add_col_if_missing(self, table: str, col: str, col_def_sql: str):
        """
        Postgres: ALTER TABLE ... ADD COLUMN IF NOT EXISTS 지원.
        """
        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_def_sql};")
        finally:
            conn.close()
        # 캐시 갱신
        self._table_columns.cache_clear()

    def _migrate_schema(self):
        """
        너가 겪었던 에러(컬럼 없음)를 여기서 자동으로 해결.
        - btc_orders.updated_at 없으면 추가
        - btc_fills.trade_time_ms 없으면 추가
        - btc_fills.inserted_at 없으면 추가
        - btc_position.updated_at 없으면 추가
        """
        # orders
        self._add_col_if_missing("btc_orders", "updated_at", "TIMESTAMPTZ DEFAULT now()")
        self._add_col_if_missing("btc_orders", "update_time_ms", "BIGINT")

        # fills
        self._add_col_if_missing("btc_fills", "trade_time_ms", "BIGINT")
        self._add_col_if_missing("btc_fills", "inserted_at", "TIMESTAMPTZ DEFAULT now()")

        # position
        self._add_col_if_missing("btc_position", "updated_at", "TIMESTAMPTZ DEFAULT now()")

        # settings
        self._add_col_if_missing("btc_settings", "updated_at", "TIMESTAMPTZ DEFAULT now()")

    def _pick_order_col(self, table: str, candidates: List[str]) -> Optional[str]:
        cols = self._table_columns(table)
        for c in candidates:
            if c in cols:
                return c
        return None

    # -----------------------------
    # logging (호환형)
    # -----------------------------
    def log(self, level_or_message: str, message: str | None = None):
        if message is None:
            level = "INFO"
            msg = str(level_or_message)
        else:
            level = str(level_or_message)
            msg = str(message)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"[{ts}] {level} {msg}"
        if self.verbose:
            print(line, flush=True)

    # -----------------------------
    # settings
    # -----------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM btc_settings WHERE key=%s", (key,))
                row = cur.fetchone()
                if not row:
                    return default
                return row["value"]
        finally:
            conn.close()

    def set_setting(self, key: str, value: Any):
        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO btc_settings(key, value, updated_at)
                        VALUES (%s, %s, now())
                        ON CONFLICT (key)
                        DO UPDATE SET value=EXCLUDED.value, updated_at=now()
                        """,
                        (key, str(value)),
                    )
        finally:
            conn.close()

    # -----------------------------
    # orders
    # -----------------------------
    def get_recent_open_orders(self, limit: int = 50) -> List[int]:
        """
        DB에 저장된 open 상태 주문(order_id) 목록 반환.
        기존 스키마에 updated_at이 없을 수 있으니 fallback 정렬.
        """
        order_col = self._pick_order_col(
            "btc_orders",
            candidates=["updated_at", "update_time_ms", "order_id"],
        )

        sql = """
            SELECT order_id
            FROM btc_orders
            WHERE status IN ('NEW', 'PARTIALLY_FILLED')
        """
        if order_col:
            sql += f" ORDER BY {order_col} DESC"
        sql += " LIMIT %s"

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (int(limit),))
                rows = cur.fetchall() or []
                return [int(r["order_id"]) for r in rows]
        finally:
            conn.close()

    def upsert_order(self, o: Dict[str, Any]):
        """
        Binance order dict를 btc_orders에 upsert.
        """
        order_id = int(o["orderId"])
        symbol = o.get("symbol", "")
        side = o.get("side")
        typ = o.get("type")
        status = o.get("status")
        client_order_id = o.get("clientOrderId") or o.get("clientOrderID") or None

        def _f(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)

        price = _f(o.get("price", 0))
        orig_qty = _f(o.get("origQty", 0))
        executed_qty = _f(o.get("executedQty", 0))
        cum_quote_qty = _f(o.get("cummulativeQuoteQty", o.get("cumQuote", 0)), 0.0)

        update_time_ms = o.get("updateTime") or o.get("transactTime") or o.get("time") or 0
        try:
            update_time_ms = int(update_time_ms)
        except Exception:
            update_time_ms = 0

        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO btc_orders(
                        order_id, symbol, side, type, status,
                        price, orig_qty, executed_qty, cum_quote_qty,
                        update_time_ms, client_order_id, raw_json, updated_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                        ON CONFLICT (order_id)
                        DO UPDATE SET
                        symbol=EXCLUDED.symbol,
                        side=EXCLUDED.side,
                        type=EXCLUDED.type,
                        status=EXCLUDED.status,
                        price=EXCLUDED.price,
                        orig_qty=EXCLUDED.orig_qty,
                        executed_qty=EXCLUDED.executed_qty,
                        cum_quote_qty=EXCLUDED.cum_quote_qty,
                        update_time_ms=EXCLUDED.update_time_ms,
                        client_order_id=EXCLUDED.client_order_id,
                        raw_json=EXCLUDED.raw_json,
                        updated_at=now()
                        """,
                        (
                            order_id, symbol, side, typ, status,
                            price, orig_qty, executed_qty, cum_quote_qty,
                            update_time_ms, client_order_id, json.dumps(o),
                        ),
                    )
        finally:
            conn.close()

    # -----------------------------
    # fills (myTrades)
    # -----------------------------
    def insert_fill(self, order_id: int, t: Dict[str, Any]):
        """
        Binance myTrades의 trade dict를 btc_fills에 insert (중복 방지: trade_id PK)
        """
        trade_id = int(t.get("id"))
        symbol = t.get("symbol", "")
        price = float(t.get("price", 0.0))
        qty = float(t.get("qty", 0.0))
        quote_qty = float(t.get("quoteQty", price * qty))
        commission = float(t.get("commission", 0.0))
        commission_asset = t.get("commissionAsset")
        time_ms = int(t.get("time", 0))

        side = "BUY" if bool(t.get("isBuyer", False)) else "SELL"

        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO btc_fills(
                          trade_id, order_id, symbol, side,
                          price, qty, quote_qty,
                          commission, commission_asset,
                          trade_time_ms, raw_json
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (trade_id) DO NOTHING
                        """,
                        (
                            trade_id,
                            int(order_id),
                            symbol,
                            side,
                            price,
                            qty,
                            quote_qty,
                            commission,
                            commission_asset,
                            time_ms,
                            json.dumps(t),
                        ),
                    )
        finally:
            conn.close()

    # -----------------------------
    # position (computed from fills)
    # -----------------------------
    def recompute_position_from_fills(self, symbol: str) -> PositionSnapshot:
        """
        fills 전체를 기반으로 평단 포지션 재계산.
        trade_time_ms가 없을 수 있으니 inserted_at / trade_id로 fallback 정렬.
        """
        order_col = self._pick_order_col(
            "btc_fills",
            candidates=["trade_time_ms", "inserted_at", "trade_id"],
        )

        sql = """
            SELECT side, price, qty, quote_qty
            FROM btc_fills
            WHERE symbol=%s
        """
        if order_col:
            sql += f" ORDER BY {order_col} ASC"

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol,))
                rows = cur.fetchall() or []

            qty = 0.0
            cost = 0.0

            for r in rows:
                side = (r.get("side") or "").upper()
                q = float(r.get("qty") or 0.0)
                quote = float(r.get("quote_qty") or (float(r.get("price") or 0.0) * q))

                if q <= 0:
                    continue

                if side == "BUY":
                    qty += q
                    cost += quote
                else:  # SELL
                    if qty <= 1e-12:
                        continue
                    avg = cost / qty if qty > 0 else 0.0
                    sell_q = min(q, qty)
                    qty -= sell_q
                    cost -= avg * sell_q
                    if qty <= 1e-12:
                        qty = 0.0
                        cost = 0.0

            avg_entry = (cost / qty) if qty > 0 else 0.0
            snap = PositionSnapshot(
                symbol=symbol,
                btc_qty=float(qty),
                cost_basis_usdt=float(cost),
                avg_entry=float(avg_entry),
                updated_at=datetime.utcnow().isoformat(),
            )

            # upsert snapshot
            conn2 = self.get_connection()
            try:
                with conn2:
                    with conn2.cursor() as cur2:
                        cur2.execute(
                            """
                            INSERT INTO btc_position(symbol, btc_qty, cost_basis_usdt, avg_entry, updated_at)
                            VALUES (%s,%s,%s,%s,now())
                            ON CONFLICT (symbol)
                            DO UPDATE SET
                              btc_qty=EXCLUDED.btc_qty,
                              cost_basis_usdt=EXCLUDED.cost_basis_usdt,
                              avg_entry=EXCLUDED.avg_entry,
                              updated_at=now()
                            """,
                            (symbol, snap.btc_qty, snap.cost_basis_usdt, snap.avg_entry),
                        )
            finally:
                conn2.close()

            return snap
        finally:
            conn.close()

    def get_position(self, symbol: str) -> PositionSnapshot:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, btc_qty, cost_basis_usdt, avg_entry, updated_at
                    FROM btc_position
                    WHERE symbol=%s
                    """,
                    (symbol,),
                )
                row = cur.fetchone()
                if not row:
                    return self.recompute_position_from_fills(symbol)

                return PositionSnapshot(
                    symbol=row["symbol"],
                    btc_qty=float(row["btc_qty"]),
                    cost_basis_usdt=float(row["cost_basis_usdt"]),
                    avg_entry=float(row["avg_entry"]),
                    updated_at=str(row["updated_at"]),
                )
        finally:
            conn.close()


    def recompute_pool_position_from_fills(self, symbol: str, client_prefix: str = "BTCSTACK_") -> PositionSnapshot:
        """
        ✅ 봇 주문(client_order_id prefix)로 발생한 fills만 모아서 pool 포지션 계산.
        fills에는 clientOrderId가 없으니, fills.order_id -> orders.client_order_id로 조인해서 필터링.
        """
        sql = """
            SELECT f.side, f.price, f.qty, f.quote_qty
            FROM btc_fills f
            JOIN btc_orders o ON o.order_id = f.order_id
            WHERE f.symbol=%s
            AND o.client_order_id LIKE %s
            ORDER BY COALESCE(f.trade_time_ms, 0) ASC, f.trade_id ASC
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, f"{client_prefix}%"))
                rows = cur.fetchall() or []

            qty = 0.0
            cost = 0.0

            for r in rows:
                side = (r.get("side") or "").upper()
                q = float(r.get("qty") or 0.0)
                quote = float(r.get("quote_qty") or (float(r.get("price") or 0.0) * q))
                if q <= 0:
                    continue

                if side == "BUY":
                    qty += q
                    cost += quote
                else:  # SELL
                    if qty <= 1e-12:
                        continue
                    avg = cost / qty if qty > 0 else 0.0
                    sell_q = min(q, qty)
                    qty -= sell_q
                    cost -= avg * sell_q
                    if qty <= 1e-12:
                        qty = 0.0
                        cost = 0.0

            avg_entry = (cost / qty) if qty > 0 else 0.0
            snap = PositionSnapshot(
                symbol=symbol,
                btc_qty=float(qty),
                cost_basis_usdt=float(cost),
                avg_entry=float(avg_entry),
                updated_at=datetime.utcnow().isoformat(),
            )

            # upsert pool snapshot
            conn2 = self.get_connection()
            try:
                with conn2:
                    with conn2.cursor() as cur2:
                        cur2.execute(
                            """
                            INSERT INTO btc_pool_position(symbol, btc_qty, cost_basis_usdt, avg_entry, updated_at)
                            VALUES (%s,%s,%s,%s,now())
                            ON CONFLICT (symbol)
                            DO UPDATE SET
                            btc_qty=EXCLUDED.btc_qty,
                            cost_basis_usdt=EXCLUDED.cost_basis_usdt,
                            avg_entry=EXCLUDED.avg_entry,
                            updated_at=now()
                            """,
                            (snap.symbol, snap.btc_qty, snap.cost_basis_usdt, snap.avg_entry),
                        )
            finally:
                conn2.close()

            return snap
        finally:
            conn.close()


    def get_pool_position(self, symbol: str) -> PositionSnapshot:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, btc_qty, cost_basis_usdt, avg_entry, updated_at
                    FROM btc_pool_position
                    WHERE symbol=%s
                    """,
                    (symbol,),
                )
                row = cur.fetchone()
                if not row:
                    return self.recompute_pool_position_from_fills(symbol)

                return PositionSnapshot(
                    symbol=row["symbol"],
                    btc_qty=float(row["btc_qty"]),
                    cost_basis_usdt=float(row["cost_basis_usdt"]),
                    avg_entry=float(row["avg_entry"]),
                    updated_at=str(row["updated_at"]),
                )
        finally:
            conn.close()

    # -----------------------------
    # AI
    # -----------------------------  
    def insert_ai_event(self, kind: str, symbol: str, payload: Dict[str, Any]):
        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO btc_ai_events(kind, symbol, payload)
                        VALUES (%s, %s, %s)
                        """,
                        (kind, symbol, json.dumps(payload)),
                    )
        finally:
            conn.close()