"""Configuration from environment / ``.env``.

The backtester runs fully offline on synthetic data by default. Set
``DATA_SOURCE=yfinance`` to pull real NSE history instead (requires the optional
``yfinance`` dependency and an internet connection).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# A handful of large-cap NSE tickers used for the synthetic universe. The
# ``.NS`` suffix is what Yahoo Finance expects, so the same symbols work for the
# optional live loader.
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]


@dataclass(frozen=True)
class Settings:
    """Backtest settings resolved from the environment."""

    data_source: str = _get("DATA_SOURCE", "synthetic")  # "synthetic" | "yfinance"
    data_dir: str = _get("DATA_DIR", "data")

    start_date: str = _get("START_DATE", "2018-01-01")
    end_date: str = _get("END_DATE", "2023-12-31")

    initial_capital: float = float(_get("INITIAL_CAPITAL", "1000000"))  # INR 10 lakh
    # Round-trip cost as a fraction of traded notional (brokerage + STT + slippage).
    cost_per_trade: float = float(_get("COST_PER_TRADE", "0.0010"))
    # Trading days per year on the NSE, used to annualise risk metrics.
    trading_days: int = int(_get("TRADING_DAYS", "252"))
    risk_free_rate: float = float(_get("RISK_FREE_RATE", "0.06"))  # annual, INR

    seed: int = int(_get("SEED", "42"))
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))


@lru_cache
def get_settings() -> Settings:
    return Settings()
