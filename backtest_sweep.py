# backtest_sweep.py
import os
from dataclasses import replace

import pandas as pd
from dotenv import load_dotenv

from back_test import BTCBacktestConfig, run_backtest
from backtest_run import build_db_url_from_env, load_ohlcv_from_supabase


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() != "" else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(float(raw)) if raw is not None and raw.strip() != "" else int(default)


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")
    vals: list[float] = []
    i = 0
    while True:
        v = start + step * i
        if v > stop + 1e-12:
            break
        vals.append(round(v, 10))
        i += 1
    return vals


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    load_dotenv()

    db_url = os.getenv("SUPABASE_DB_URL", "").strip() or build_db_url_from_env()
    cfg_base = BTCBacktestConfig.from_env()

    interval = os.getenv("BT_SWEEP_INTERVAL", "5m")
    start = os.getenv("BT_SWEEP_START") or None
    end = os.getenv("BT_SWEEP_END") or None
    limit_raw = os.getenv("BT_SWEEP_LIMIT") or None
    limit = int(limit_raw) if limit_raw else None

    df = load_ohlcv_from_supabase(
        db_url=db_url,
        region="BI",
        symbol=cfg_base.symbol,
        interval=interval,
        start=start,
        end=end,
        limit=limit,
    )
    if df.empty:
        raise RuntimeError("ohlcv_data에서 데이터가 안 나왔어 (region/symbol/interval 확인)")

    drop_min = _env_float("BT_SWEEP_DROP_MIN", 0.001)
    drop_max = _env_float("BT_SWEEP_DROP_MAX", 0.02)
    drop_step = _env_float("BT_SWEEP_DROP_STEP", 0.0005)
    tp_min = _env_float("BT_SWEEP_TP_MIN", 0.002)
    tp_max = _env_float("BT_SWEEP_TP_MAX", 0.03)
    tp_step = _env_float("BT_SWEEP_TP_STEP", 0.0005)
    top_n = _env_int("BT_SWEEP_TOP_N", 10)
    verbose = _is_true(os.getenv("BT_SWEEP_VERBOSE"))

    drop_vals = _float_range(drop_min, drop_max, drop_step)
    tp_vals = _float_range(tp_min, tp_max, tp_step)

    results: list[dict] = []
    total = len(drop_vals) * len(tp_vals)
    idx = 0

    for drop in drop_vals:
        for tp in tp_vals:
            idx += 1
            if verbose:
                print(f"[{idx}/{total}] drop={drop:.6f} tp={tp:.6f}")

            cfg = replace(cfg_base, lot_drop_pct=drop, lot_tp_pct=tp)
            _, summary = run_backtest(df, cfg=cfg)
            core_btc_initial = float(summary.get("core_btc_initial", 0.0) or 0.0)
            core_btc_added = float(summary.get("core_btc_added", 0.0) or 0.0)
            if core_btc_initial > 0:
                core_btc_added_pct = (core_btc_added / core_btc_initial) * 100.0
                score = core_btc_added_pct
            else:
                core_btc_added_pct = None
                score = float("-inf")

            results.append(
                {
                    "lot_drop_pct": drop,
                    "lot_tp_pct": tp,
                    "core_btc_initial": core_btc_initial,
                    "core_btc_added": core_btc_added,
                    "core_btc_added_pct": core_btc_added_pct,
                    "reserve_btc_qty": float(summary.get("reserve_btc_qty", 0.0) or 0.0),
                    "score": score,
                }
            )

    out = pd.DataFrame(results).sort_values(
        by=["score", "core_btc_added"], ascending=False
    )

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", "backtest_sweep.csv")
    out.to_csv(out_path, index=False)

    best = out.iloc[0] if len(out) else None
    print("\n=== SWEEP SUMMARY ===")
    print(f"- drop range: {drop_min:.6f} ~ {drop_max:.6f} (step {drop_step:.6f})")
    print(f"- tp range:   {tp_min:.6f} ~ {tp_max:.6f} (step {tp_step:.6f})")
    print(f"- total runs: {total}")
    if best is not None:
        print("BEST (core btc 증가율 기준)")
        print(f"- lot_drop_pct: {best['lot_drop_pct']:.6f}")
        print(f"- lot_tp_pct:   {best['lot_tp_pct']:.6f}")
        print(f"- core_btc_added_pct: {best['core_btc_added_pct']:.4f}")
        print(f"- core_btc_added: {best['core_btc_added']:.8f}")
        print(f"- reserve_btc_qty: {best['reserve_btc_qty']:.8f}")
    print(f"saved: {out_path}")

    if top_n > 0:
        top_n = min(top_n, len(out))
        print(f"\nTOP {top_n}")
        cols = ["lot_drop_pct", "lot_tp_pct", "core_btc_added_pct", "core_btc_added"]
        print(out.head(top_n)[cols].to_string(index=False))
