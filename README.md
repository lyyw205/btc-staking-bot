# BTC Stacking Bot

## 프로젝트 구조
- `btc_run.py`: 실거래 실행 엔트리 포인트.
- `btc_trader.py`: 로트 스태킹 전략 핵심 로직과 루프.
- `btc_config.py`: 실거래 환경변수 기반 설정 로더.
- `btc_client.py`: 거래소 API 연동 클라이언트.
- `btc_db_manager.py`: 거래/상태 저장을 위한 DB 매니저.
- `backtest_run.py`: 백테스트 실행 스크립트, `artifacts/backtest_equity.csv` 출력.
- `back_test.py`: 백테스트 엔진 + 설정 + 모의 거래소/DB (통합).

- `log_dashboard.py`: `journalctl` 로그를 보여주는 Flask 대시보드.
- `tests/`: `unittest` 기반 스모크 테스트.
- `artifacts/`: 실행 결과물 저장 폴더(예: `backtest_equity.csv`).
- `requirements.txt`: 런타임 의존성 목록.

## 빠른 실행
- 가상환경 생성: `python -m venv .venv && source .venv/bin/activate`
- 의존성 설치: `pip install -r requirements.txt`
- 실거래 실행: `python btc_run.py`
- 백테스트 실행: `python backtest_run.py`
- 로그 대시보드: `python log_dashboard.py`
