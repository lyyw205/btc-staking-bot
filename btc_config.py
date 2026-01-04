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

    # ---- Lot Stacking ----
    lot_buy_usdt: float = float(os.getenv("BTC_LOT_BUY_USDT", "100.0"))
    lot_tp_pct: float = float(os.getenv("BTC_LOT_TP_PCT", "0.03"))
    lot_drop_pct: float = float(os.getenv("BTC_LOT_DROP_PCT", "0.01"))
    lot_prebuy_pct: float = float(os.getenv("BTC_LOT_PREBUY_PCT", "0.0015"))
    lot_cancel_rebound_pct: float = float(os.getenv("BTC_LOT_CANCEL_REBOUND_PCT", "0.004"))
    initial_core_usdt: float = float(os.getenv("BTC_INITIAL_CORE_USDT", "2000.0"))
    min_trade_usdt: float = float(os.getenv("BTC_MIN_TRADE_USDT", "6.0"))

    # ---- Loop ----
    loop_interval_sec: int = int(os.getenv("BTC_LOOP_INTERVAL_SEC", "60"))
    order_cooldown_sec: int = int(os.getenv("BTC_ORDER_COOLDOWN_SEC", "7"))

    # ---- Logging ----
    verbose: bool = os.getenv("BTC_VERBOSE", "true").lower() == "true"

    # ---- Capital split (USDT) ----
    trade_cap_ratio: float = float(os.getenv("BTC_TRADE_CAP_RATIO", "0.30"))  # 운용 비율(30%)
    usdt_reserve_buffer: float = float(os.getenv("BTC_USDT_RESERVE_BUFFER", "2.0"))  # 주문 실패 방지용 소액 버퍼
    use_fixed_usdt_reference: bool = os.getenv("BTC_USE_FIXED_USDT_REFERENCE", "true").lower() == "true"

    # ---- Initial entry (one-shot) ----
    initial_buy_on_start: bool = os.getenv("BTC_INITIAL_BUY_ON_START", "true").lower() == "true"
    initial_buy_usdt: float | None = float(os.getenv("BTC_INITIAL_BUY_USDT")) if os.getenv("BTC_INITIAL_BUY_USDT") else None
    initial_buy_ratio: float = float(os.getenv("BTC_INITIAL_BUY_RATIO", "0.30"))  # 기본 30%
