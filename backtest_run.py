# btc_run_test.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus


from back_test import BTCBacktestConfig, run_backtest


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

    # ✅ 백테스트 config는 back_test.py에서 관리
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

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "backtest_equity.csv")
    equity.to_csv(out_path, index=False)

    print("\n=== BACKTEST SUMMARY ===")
    print("설정값")
    print(f"- 추가 매수 하락폭: {cfg.lot_drop_pct:.4f}")
    print(f"- 로트 익절 비율: {cfg.lot_tp_pct:.4f}")
    print(f"- 로트 매수금액: {cfg.lot_buy_usdt:.2f}")

    print("결과")
    core_btc_initial = float(summary.get("core_btc_initial", 0.0) or 0.0)
    core_btc_added = float(summary.get("core_btc_added", 0.0) or 0.0)
    if core_btc_initial > 0:
        core_btc_added_pct = (core_btc_added / core_btc_initial) * 100.0
        added_desc = f"{core_btc_added:.8f} ({core_btc_added_pct:.2f}%)"
    else:
        added_desc = f"{core_btc_added:.8f}"
    print(f"- 코어 BTC 초기: {core_btc_initial:.8f}")
    print(f"- 코어 BTC 증가: {added_desc}")
    print(f"- 코어 BTC 합계: {summary.get('reserve_btc_qty', 0.0):.8f}")
    print(f"- 추가 매수 사용 USDT: {summary.get('add_buy_spent_usdt', 0.0):.2f}")
    print(f"- 매도 회수 USDT(순): {summary.get('sell_net_usdt', 0.0):.2f}")
    print(f"- 종료 USDT 자유: {summary.get('end_usdt_free', 0.0):.2f}")
    print(f"- 종료 USDT 묶임: {summary.get('end_usdt_locked', 0.0):.2f}")
    print(f"- 오픈 로트 수: {int(summary.get('open_lots_count', 0))}")
    print(f"- 매수 횟수: {int(summary.get('stats', {}).get('market_buy_orders', 0))}")
    print(f"- 매도 횟수: {int(summary.get('stats', {}).get('market_sell_orders', 0))}")
    stats = summary.get("stats") or {}
    print(f"- market buys:              {int(stats.get('market_buy_orders', 0))}")
    print(f"- market sells:             {int(stats.get('market_sell_orders', 0))}")
    print(f"- cancels:                  {int(stats.get('cancels', 0))}")

    print(f"saved: {out_path}")
