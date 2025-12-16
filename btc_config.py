# btc_config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import quote_plus

# .env 자동 로딩
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class BTCConfig:
    # ---- Exchange / Symbol ----
    exchange: str = os.getenv("BTC_EXCHANGE", "binance")
    symbol: str = os.getenv("BTC_SYMBOL", "BTCUSDT")
    base_asset: str = os.getenv("BTC_BASE_ASSET", "BTC")
    quote_asset: str = os.getenv("BTC_QUOTE_ASSET", "USDT")

    # ---- Binance ----
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")

    # ---- DB (prefer BTC_DB_URL, else build from parts) ----
    db_url: str = os.getenv("BTC_DB_URL", "")
    db_host: str = os.getenv("BTC_DB_HOST", "")
    db_name: str = os.getenv("BTC_DB_NAME", "postgres")
    db_user: str = os.getenv("BTC_DB_USER", "")
    db_pass: str = os.getenv("BTC_DB_PASS", "")
    db_port: str = os.getenv("BTC_DB_PORT", "6543")
    db_sslmode: str = os.getenv("BTC_DB_SSLMODE", "require")

    def build_db_url(self) -> str:
        """
        Supabase pooler는 보통 SSL 필요.
        비밀번호 특수문자(@ 등) 대응을 위해 quote_plus로 인코딩.
        """
        if self.db_url:
            return self.db_url

        if not (self.db_host and self.db_user and self.db_pass and self.db_name and self.db_port):
            return ""

        encoded_pass = quote_plus(self.db_pass)
        return (
            f"postgresql://{self.db_user}:{encoded_pass}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_sslmode}"
        )

    # ---- Strategy (Stacking/Grid) ----
    grid_step_pct: float = float(os.getenv("BTC_GRID_STEP_PCT", "0.008"))
    take_profit_pct: float = float(os.getenv("BTC_TAKE_PROFIT_PCT", "0.010"))
    buy_quote_usdt: float = float(os.getenv("BTC_BUY_QUOTE_USDT", "25.0"))
    sell_fraction_on_tp: float = float(os.getenv("BTC_SELL_FRACTION_ON_TP", "0.15"))
    min_trade_usdt: float = float(os.getenv("BTC_MIN_TRADE_USDT", "6.0"))

    # ---- TP as LIMIT (v2.1) ----
    use_tp_limit_orders: bool = os.getenv("BTC_USE_TP_LIMIT_ORDERS", "true").lower() == "true"
    tp_refresh_sec: int = int(os.getenv("BTC_TP_REFRESH_SEC", "30"))
    tp_price_bump_ticks: int = int(os.getenv("BTC_TP_PRICE_BUMP_TICKS", "0"))

    # ---- Dynamic Buy Sizing (v2.1) ----
    dynamic_buy: bool = os.getenv("BTC_DYNAMIC_BUY", "true").lower() == "true"
    buy_min_usdt: float = float(os.getenv("BTC_BUY_MIN_USDT", "10.0"))
    buy_max_usdt: float = float(os.getenv("BTC_BUY_MAX_USDT", "60.0"))
    exposure_power: float = float(os.getenv("BTC_EXPOSURE_POWER", "1.6"))
    price_vol_window: int = int(os.getenv("BTC_PRICE_VOL_WINDOW", "30"))
    vol_low: float = float(os.getenv("BTC_VOL_LOW", "0.005"))
    vol_high: float = float(os.getenv("BTC_VOL_HIGH", "0.012"))
    vol_boost_max: float = float(os.getenv("BTC_VOL_BOOST_MAX", "1.25"))
    vol_cut_min: float = float(os.getenv("BTC_VOL_CUT_MIN", "0.70"))

    # ---- Risk / Limits ----
    max_quote_exposure_usdt: float = float(os.getenv("BTC_MAX_QUOTE_EXPOSURE_USDT", "500.0"))

    # ---- Base price recentering ----
    recenter_threshold_pct: float = float(os.getenv("BTC_RECENTER_THRESHOLD_PCT", "0.020"))
    recenter_cooldown_sec: int = int(os.getenv("BTC_RECENTER_COOLDOWN_SEC", "60"))

    # ---- Loop ----
    loop_interval_sec: int = int(os.getenv("BTC_LOOP_INTERVAL_SEC", "60"))
    order_cooldown_sec: int = int(os.getenv("BTC_ORDER_COOLDOWN_SEC", "7"))

    # ---- Logging ----
    verbose: bool = os.getenv("BTC_VERBOSE", "true").lower() == "true"

    # ---- Capital split (USDT) ----
    trade_cap_ratio: float = float(os.getenv("BTC_TRADE_CAP_RATIO", "0.30"))  # 운용 비율(30%)
    cash_reserve_ratio: float = float(os.getenv("BTC_CASH_RESERVE_RATIO", "0.70"))  # 고정 보유(70%)
    usdt_reserve_buffer: float = float(os.getenv("BTC_USDT_RESERVE_BUFFER", "2.0"))  # 주문 실패 방지용 소액 버퍼
    use_fixed_usdt_reference: bool = os.getenv("BTC_USE_FIXED_USDT_REFERENCE", "true").lower() == "true"

    # ---- Initial entry (one-shot) ----
    initial_buy_on_start: bool = os.getenv("BTC_INITIAL_BUY_ON_START", "false").lower() == "true"
    initial_buy_ratio: float = float(os.getenv("BTC_INITIAL_BUY_RATIO", "0.70"))  # 기본 70%