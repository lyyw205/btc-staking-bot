from btc_config import BTCConfig
from btc_trader import BTCStackingTrader

def main():
    cfg = BTCConfig()
    trader = BTCStackingTrader(cfg)
    trader.run_forever()

if __name__ == "__main__":
    main()