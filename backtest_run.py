# btc_run_test.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus


from backtest_config import BTCBacktestConfig
from back_test import run_backtest


def build_db_url_from_env() -> str:
    host = os.getenv("BTC_DB_HOST", "").strip()
    name = os.getenv("BTC_DB_NAME", "postgres").strip()
    user = os.getenv("BTC_DB_USER", "").strip()
    pw   = os.getenv("BTC_DB_PASS", "")
    port = os.getenv("BTC_DB_PORT", "5432").strip()

    if not all([host, name, user, pw, port]):
        raise RuntimeError("BTC_DB_HOST/NAME/USER/PASS/PORT 중 비어있는 값이 있어")

    pw_enc = quote_plus(pw)
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

    # ✅ 백테스트 config는 backtest_config.py에서 관리
    cfg = BTCBacktestConfig()  # 또는 BTCBacktestConfig.from_env()

    df = load_ohlcv_from_supabase(
        db_url=db_url,
        region="BI",
        symbol=cfg.symbol,
        interval="5m",
        start=None,
        end=None,
        limit=None,
    )

    if df.empty:
        raise RuntimeError("ohlcv_data에서 데이터가 안 나왔어 (region/symbol/interval 확인)")

    equity, summary = run_backtest(df, cfg=cfg)

    equity.to_csv("backtest_equity.csv", index=False)

    print("\n=== BACKTEST SUMMARY ===")
    print(f"- grid buys count:         {int(summary.get('grid_buy_count', 0))}")
    print(f"- grid buys spent total:   {float(summary.get('grid_buy_spent_total', 0.0)):.2f} USDT")
    print(f"- sells count:             {int(summary.get('sell_count', 0))}")
    print(f"- initial: USDT={cfg.init_usdt}, BTC={cfg.init_btc}")
    print(f"- stacking period: {summary['first_buy_dt']}  ~  {summary['end_dt']}")
    print(f"- BTC start(after first buy): {summary['btc_start_after_first_buy']:.8f}")
    print(f"- BTC end:               {summary['btc_end']:.8f}")
    print(f"- BTC delta:             {summary['btc_delta']:.8f}")
    if summary["btc_delta_pct"] is not None:
        print(f"- BTC delta %:           {summary['btc_delta_pct']:.2f}%")
    print(f"- equity_end_usdt:       {summary['equity_end_usdt']:.2f}")
    print(f"- btc_equiv_end:         {summary['btc_equiv_end']:.8f}")

    print(f"- total_deposit(incl topups): {summary.get('deposit_total', 0.0):.2f}")
    print(f"- topup_total:               {summary.get('topup_total', 0.0):.2f}")
    print(f"- fee_total:                {summary.get('fee_total', 0.0):.2f}")
    print(f"- btc_start_init:           {summary.get('btc_start_init', 0.0):.8f}")
    print(f"- btc_end:                  {summary['btc_end']:.8f}")

    stats = summary.get("stats") or {}
    print(f"- market buys:              {int(stats.get('market_buy_orders', 0))}")
    print(f"- TP limit fills:           {int(stats.get('tp_limit_fills', 0))}")
    print(f"- cancels:                  {int(stats.get('cancels', 0))}")


    stats = summary.get("stats") or {}
    if stats:
        print("\n--- TRADING COUNTS ---")
        print(f"- market buys:     {int(stats.get('market_buy_orders', 0))}")
        print(f"- TP limit placed: {int(stats.get('tp_limit_orders', 0))}")
        print(f"- TP limit fills:  {int(stats.get('tp_limit_fills', 0))}")
        print(f"- cancels:         {int(stats.get('cancels', 0))}")

    print("saved: backtest_equity.csv")
