# backtest_config.py
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BTCBacktestConfig:
    """
    Backtest-only config. Keep only fields used by backtest execution.
    Env overrides supported via BT_* prefix.
    """

    # Mode / Logging
    verbose: bool = False
    ai_enable_report: bool = True
    ai_enable_gate: bool = True
    ai_enable_tune: bool = True

    # Symbol
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    # Backtest Account Assumptions
    init_usdt: float = 1000000.0
    init_btc: float = 0.0
    initial_buy_usdt: float = 1000.0
    initial_reserve_ratio: float = 0.50
    reserve_btc_key: str = "reserve_btc_qty"

    # Execution Assumptions
    slippage_bps: float = 1.0
    fee_bps_taker: float = 10.0
    fee_bps_maker: float = 2.0

    # Exchange Filters
    tick_size: float = 0.1
    step_size: float = 0.000001
    min_notional: float = 10.0

    # Order ID Prefix (pool tagging)
    client_order_prefix: str = "BTCSTACK_"

    # Strategy (Stacking/Grid)
    grid_step_pct: float = 0.02
    take_profit_pct: float = 0.020
    buy_quote_usdt: float = 20.0
    sell_fraction_on_tp: float = 0.15
    min_trade_usdt: float = 0.0

    # TP as LIMIT
    use_tp_limit_orders: bool = True
    tp_refresh_sec: int = 30
    tp_price_bump_ticks: int = 0

    # Buy sizing
    fixed_grid_buy: bool = True
    grid_buy_usdt: float = 20.0
    dynamic_buy: bool = True
    buy_min_usdt: float = 5.0
    buy_max_usdt: float = 60.0
    exposure_power: float = 1.6
    price_vol_window: int = 30
    vol_low: float = 0.005
    vol_high: float = 0.012
    vol_boost_max: float = 1.25
    vol_cut_min: float = 0.70

    # Risk / Limits
    max_quote_exposure_usdt: float = 1e18

    # Base price recentering
    recenter_threshold_pct: float = 0.020
    recenter_cooldown_sec: int = 60

    # Crash-aware sizing
    crash_drop_pct: float = 0.035
    crash_vol_threshold: float = 0.015
    crash_grid_mult: float = 2.0

    # Timing
    order_cooldown_sec: int = 7

    # Capital split
    trade_cap_ratio: float = 1.0
    usdt_reserve_buffer: float = 0.0
    use_fixed_usdt_reference: bool = False

    # Optional: BT_* env override
    @staticmethod
    def _getenv(name: str, default: str) -> str:
        return os.getenv(f"BT_{name}", default)

    @classmethod
    def from_env(cls) -> "BTCBacktestConfig":
        def f(name: str, default: float) -> float:
            return float(cls._getenv(name, str(default)))

        def i(name: str, default: int) -> int:
            return int(float(cls._getenv(name, str(default))))

        def b(name: str, default: bool) -> bool:
            return cls._getenv(name, "true" if default else "false").lower() == "true"

        def s(name: str, default: str) -> str:
            return cls._getenv(name, default)

        return cls(
            verbose=b("VERBOSE", cls.verbose),
            ai_enable_report=b("AI_ENABLE_REPORT", cls.ai_enable_report),
            ai_enable_gate=b("AI_ENABLE_GATE", cls.ai_enable_gate),
            ai_enable_tune=b("AI_ENABLE_TUNE", cls.ai_enable_tune),

            symbol=s("SYMBOL", cls.symbol),
            base_asset=s("BASE_ASSET", cls.base_asset),
            quote_asset=s("QUOTE_ASSET", cls.quote_asset),

            init_usdt=f("INIT_USDT", cls.init_usdt),
            init_btc=f("INIT_BTC", cls.init_btc),
            initial_buy_usdt=f("INITIAL_BUY_USDT", cls.initial_buy_usdt),
            initial_reserve_ratio=f("INITIAL_RESERVE_RATIO", cls.initial_reserve_ratio),
            reserve_btc_key=s("RESERVE_BTC_KEY", cls.reserve_btc_key),

            slippage_bps=f("SLIPPAGE_BPS", cls.slippage_bps),
            fee_bps_taker=f("FEE_BPS_TAKER", cls.fee_bps_taker),
            fee_bps_maker=f("FEE_BPS_MAKER", cls.fee_bps_maker),

            tick_size=f("TICK_SIZE", cls.tick_size),
            step_size=f("STEP_SIZE", cls.step_size),
            min_notional=f("MIN_NOTIONAL", cls.min_notional),

            client_order_prefix=s("CLIENT_ORDER_PREFIX", cls.client_order_prefix),

            grid_step_pct=f("GRID_STEP_PCT", cls.grid_step_pct),
            take_profit_pct=f("TAKE_PROFIT_PCT", cls.take_profit_pct),
            buy_quote_usdt=f("BUY_QUOTE_USDT", cls.buy_quote_usdt),
            sell_fraction_on_tp=f("SELL_FRACTION_ON_TP", cls.sell_fraction_on_tp),
            min_trade_usdt=f("MIN_TRADE_USDT", cls.min_trade_usdt),

            use_tp_limit_orders=b("USE_TP_LIMIT_ORDERS", cls.use_tp_limit_orders),
            tp_refresh_sec=i("TP_REFRESH_SEC", cls.tp_refresh_sec),
            tp_price_bump_ticks=i("TP_PRICE_BUMP_TICKS", cls.tp_price_bump_ticks),

            fixed_grid_buy=b("FIXED_GRID_BUY", cls.fixed_grid_buy),
            grid_buy_usdt=f("GRID_BUY_USDT", cls.grid_buy_usdt),
            dynamic_buy=b("DYNAMIC_BUY", cls.dynamic_buy),
            buy_min_usdt=f("BUY_MIN_USDT", cls.buy_min_usdt),
            buy_max_usdt=f("BUY_MAX_USDT", cls.buy_max_usdt),
            exposure_power=f("EXPOSURE_POWER", cls.exposure_power),
            price_vol_window=i("PRICE_VOL_WINDOW", cls.price_vol_window),
            vol_low=f("VOL_LOW", cls.vol_low),
            vol_high=f("VOL_HIGH", cls.vol_high),
            vol_boost_max=f("VOL_BOOST_MAX", cls.vol_boost_max),
            vol_cut_min=f("VOL_CUT_MIN", cls.vol_cut_min),

            max_quote_exposure_usdt=f("MAX_QUOTE_EXPOSURE_USDT", cls.max_quote_exposure_usdt),

            recenter_threshold_pct=f("RECENTER_THRESHOLD_PCT", cls.recenter_threshold_pct),
            recenter_cooldown_sec=i("RECENTER_COOLDOWN_SEC", cls.recenter_cooldown_sec),

            crash_drop_pct=f("CRASH_DROP_PCT", cls.crash_drop_pct),
            crash_vol_threshold=f("CRASH_VOL_THRESHOLD", cls.crash_vol_threshold),
            crash_grid_mult=f("CRASH_GRID_MULT", cls.crash_grid_mult),

            order_cooldown_sec=i("ORDER_COOLDOWN_SEC", cls.order_cooldown_sec),

            trade_cap_ratio=f("TRADE_CAP_RATIO", cls.trade_cap_ratio),
            usdt_reserve_buffer=f("USDT_RESERVE_BUFFER", cls.usdt_reserve_buffer),
            use_fixed_usdt_reference=b("USE_FIXED_USDT_REFERENCE", cls.use_fixed_usdt_reference),
        )
