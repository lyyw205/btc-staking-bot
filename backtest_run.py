# btc_run_test.py
import os
from dataclasses import replace
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


def _bool_from_str(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def apply_db_tune_overrides(cfg: BTCBacktestConfig, db_url: str) -> BTCBacktestConfig:
    """
    Apply tune.* values from btc_settings to backtest config.
    Falls back silently if db_url is empty or query fails.
    """
    if not db_url:
        return cfg

    type_map = {
        "grid_step_pct": float,
        "take_profit_pct": float,
        "buy_quote_usdt": float,
        "tp_refresh_sec": int,
        "sell_fraction_on_tp": float,
        "min_trade_usdt": float,
        "dynamic_buy": "bool",
        "buy_min_usdt": float,
        "buy_max_usdt": float,
        "recenter_threshold_pct": float,
        "recenter_cooldown_sec": int,
        "crash_drop_pct": float,
        "loop_interval_sec": int,
        "trade_cap_ratio": float,
        "price_vol_window": int,
        "vol_low": float,
        "vol_high": float,
        "vol_boost_max": float,
        "vol_cut_min": float,
        "use_tp_limit_orders": "bool",
        "trailing_sell_enable": "bool",
        "trailing_activate_pct": float,
        "trailing_ratio": float,
        "verbose": "bool",
    }

    overrides = {}
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT key, value
                    FROM btc_settings
                    WHERE key LIKE 'tune.%'
                    """
                )
                rows = cur.fetchall() or []
    except Exception:
        return cfg

    for key, value in rows:
        if not key or not isinstance(key, str):
            continue
        field = key.replace("tune.", "", 1)
        if not hasattr(cfg, field):
            continue
        caster = type_map.get(field)
        try:
            if caster == "bool":
                overrides[field] = _bool_from_str(value)
            elif caster is int:
                overrides[field] = int(float(value))
            elif caster is float:
                overrides[field] = float(value)
        except Exception:
            continue

    if not overrides:
        return cfg

    return replace(cfg, **overrides)


if __name__ == "__main__":
    load_dotenv()

    # DB URL: SUPABASE_DB_URL 있으면 그거, 없으면 BTC_DB_*로 조립
    db_url = os.getenv("SUPABASE_DB_URL", "").strip() or build_db_url_from_env()

    # ✅ 백테스트 config는 back_test.py에서 관리
    cfg = BTCBacktestConfig()  # 또는 BTCBacktestConfig.from_env()
    if os.getenv("BT_USE_DB_TUNE", "true").lower() == "true":
        cfg = apply_db_tune_overrides(cfg, db_url)

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
    print(f"- trailing sells:        {int(summary.get('trailing_sell_count', 0))}")
    print(f"- trailing profit total: {summary.get('trailing_profit_total', 0.0):.2f}")
    print(f"- trailing profit avg:   {summary.get('trailing_profit_avg', 0.0):.2f}")
    print(f"- trailing win rate:     {summary.get('trailing_win_rate', 0.0)*100:.2f}%")
    print(f"- trailing loss avg:     {summary.get('trailing_loss_avg', 0.0):.2f}")
    print(f"- equity_end_usdt:       {summary['equity_end_usdt']:.2f}")
    print(f"- btc_equiv_end:         {summary['btc_equiv_end']:.8f}")

    print(f"- total_deposit(incl topups): {summary.get('deposit_total', 0.0):.2f}")
    print(f"- topup_total:               {summary.get('topup_total', 0.0):.2f}")
    print(f"- fee_total:                {summary.get('fee_total', 0.0):.2f}")
    print(f"- reserve_cost_usdt:        {summary.get('reserve_cost_usdt', 0.0):.2f}")
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

    print(f"saved: {out_path}")
