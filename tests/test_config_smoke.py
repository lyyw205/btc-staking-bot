import unittest

from btc_config import BTCConfig


class TestBTCConfigSmoke(unittest.TestCase):
    def test_build_db_url_empty_returns_empty(self):
        cfg = BTCConfig(
            db_url="",
            db_host="",
            db_name="",
            db_user="",
            db_pass="",
            db_port="",
            db_sslmode="require",
        )
        self.assertEqual(cfg.build_db_url(), "")


if __name__ == "__main__":
    unittest.main()
