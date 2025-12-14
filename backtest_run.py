# btc_run_test.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from dataclasses import fields, is_dataclass, replace

from btc_config import BTCConfig
from backtest import run_backtest


def safe_replace(cfg, **kwargs):
    # BTCConfig가 frozen/dataclass여도, 실제 필드만 골라 replace
    if not is_dataclass(cfg):
        return cfg
    allowed = {f.name for f in fields(cfg)}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return replace(cfg, **filtered)


def build_db_url_from_env() -> str:
    host = os.getenv("BTC_DB_HOST", "").strip()
    name = os.getenv("BTC_DB_NAME", "postgres").strip()
    user = os.getenv("BTC_DB_USER", "").strip()
    pw   = os.getenv("BTC_DB_PASS", "")
    port = os.getenv("BTC_DB_PORT", "5432").strip()

    if not all([host, name, user, pw, port]):
        raise RuntimeError("BTC_DB_HOST/NAME/USER/PASS/PORT 중 비어있는 값이 있어")

    pw_enc = quote_plus(pw)  # 특수문자 인코딩
    return f"postgresql://{user}:{pw_enc}@{host}:{port}/{name}?sslmode=require"


def load_ohlcv_from_supabase(
    *,
    db_url: str,
    region: str,
    symbol: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    where = ["region=%s", "symbol=%s", "interval=%s"]
    params = [region, symbol, interval]

    if start:
        where.append("dt >= %s")
        params.append(start)
    if end:
        where.append("dt <= %s")
        params.append(end)

    sql = f"""
    SELECT dt, open, high, low, close, volume
    FROM ohlcv_data
    WHERE {" AND ".join(where)}
    ORDER BY dt ASC
    """
    if limit:
        sql += " LIMIT %s"
        params.append(int(limit))

    with psycopg2.connect(db_url) as conn:
        df = pd.read_sql(sql, conn, params=params)

    return df


if __name__ == "__main__":
    load_dotenv()

    # DB URL: SUPABASE_DB_URL 있으면 그거, 없으면 BTC_DB_*로 조립
    db_url = os.getenv("SUPABASE_DB_URL", "").strip() or build_db_url_from_env()

    # ✅ cfg는 frozen일 수 있으니 safe_replace로 “존재하는 필드만” 반영
    cfg0 = BTCConfig()
    cfg = safe_replace(
        cfg0,
        symbol="BTCUSDT",
        ai_enable_report=False,  # 있으면 적용, 없으면 무시
        quote_asset="USDT",
        base_asset="BTC",
    )

    df = load_ohlcv_from_supabase(
        db_url=db_url,
        region="BI",
        symbol="BTCUSDT",
        interval="5m",
        start=None,  # 예: "2024-01-01"
        end=None,    # 예: "2025-01-01"
        limit=None,
    )

    if df.empty:
        raise RuntimeError("ohlcv_data에서 데이터가 안 나왔어 (region/symbol/interval 확인)")

    equity, summary = run_backtest(
        df,
        init_usdt=1000.0,
        cfg=cfg,
        slippage_bps=1.0,
        fee_bps_taker=10.0,
        fee_bps_maker=2.0,
    )

    equity.to_csv("backtest_equity.csv", index=False)

    print("\n=== BACKTEST SUMMARY ===")
    print("- initial: USDT=1000, BTC=0")
    print(f"- stacking period: {summary['first_buy_dt']}  ~  {summary['end_dt']}")
    print(f"- BTC start(after first buy): {summary['btc_start_after_first_buy']:.8f}")
    print(f"- BTC end:               {summary['btc_end']:.8f}")
    print(f"- BTC delta:             {summary['btc_delta']:.8f}")
    if summary["btc_delta_pct"] is not None:
        print(f"- BTC delta %:           {summary['btc_delta_pct']:.2f}%")
    print(f"- equity_end_usdt:       {summary['equity_end_usdt']:.2f}")
    print(f"- btc_equiv_end:         {summary['btc_equiv_end']:.8f}")

    stats = summary.get("stats") or {}
    if stats:
        print("\n--- TRADING COUNTS ---")
        print(f"- market buys:     {int(stats.get('market_buy_orders', 0))}")
        print(f"- TP limit placed: {int(stats.get('tp_limit_orders', 0))}")
        print(f"- TP limit fills:  {int(stats.get('tp_limit_fills', 0))}")
        print(f"- cancels:         {int(stats.get('cancels', 0))}")

    print("saved: backtest_equity.csv")
