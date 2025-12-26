# Repository Guidelines

## Project Structure & Module Organization
- Core trading runtime: `btc_run.py` (entrypoint) and `btc_trader.py` (strategy/loop).
- Configuration: `btc_config.py` (live trading) and `back_test.py` (backtest defaults).
- Backtesting: `back_test.py` (engine + backtest env) and `backtest_run.py` (loads data, runs, writes `backtest_equity.csv`).
- Logging UI: `log_dashboard.py` (Flask dashboard for `journalctl` service logs).
- Dependencies: `requirements.txt`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (create and activate a local venv).
- `pip install -r requirements.txt` installs runtime dependencies.
- `python btc_run.py` runs the live stacking bot loop.
- `python backtest_run.py` runs the backtest and emits `backtest_equity.csv`.
- `python log_dashboard.py` starts the local log dashboard on `127.0.0.1:8080`.

## Coding Style & Naming Conventions
- Python, 4-space indentation, PEP 8-ish style.
- Module/file names are snake_case (e.g., `btc_trader.py`).
- Class names use CapWords (e.g., `BTCStackingTrader`), constants use UPPER_SNAKE.
- No formatter or linter is configured; keep diffs small and consistent with nearby code.

## Testing Guidelines
- Tests live in `tests/` and use the standard library `unittest` framework.
- Run all tests with `python -m unittest discover -s tests`.
- Backtests are the primary validation flow for strategy changes; prefer running `python backtest_run.py` after logic updates.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative messages (e.g., "Add Flask log dashboard..."). Keep messages concise.
- PRs should include: a summary of behavior changes, any new env vars, and a quick verification note (command + outcome).
- If changes affect trading logic, include a small backtest run summary or a sample log snippet.

## Configuration & Secrets
- Runtime config is read from environment variables; see `btc_config.py` and `back_test.py` for supported keys.
- For local use, a `.env` file is supported via `python-dotenv` (loaded in `btc_config.py`).
- Never commit API keys or database credentials; document new vars in this guide.
