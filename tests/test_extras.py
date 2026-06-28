from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio, sortino_ratio
from src.strategies import BollingerMeanReversion, DonchianBreakout, default_strategies


def _trending_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000},
        index=idx,
    )


def test_sortino_zero_when_no_downside():
    # All-positive returns => no downside deviation => 0.0 by convention.
    r = pd.Series([0.01, 0.02, 0.015, 0.03])
    assert sortino_ratio(r, risk_free_rate=0.0) == 0.0


def test_sortino_rewards_upside_skew_over_sharpe():
    # Returns with big upside but small, rare downside: Sortino should exceed Sharpe.
    r = pd.Series([0.05, 0.04, -0.005, 0.06, 0.03, -0.004, 0.05])
    assert sortino_ratio(r, risk_free_rate=0.0) > sharpe_ratio(r, risk_free_rate=0.0)


def test_donchian_breakout_is_binary_and_long_in_uptrend():
    # high == close so every up-bar prints a fresh N-day high and triggers a breakout.
    idx = pd.bdate_range("2021-01-01", periods=120)
    close = pd.Series(np.linspace(100, 200, 120), index=idx)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1},
        index=idx,
    )
    pos = DonchianBreakout(20).signal(ohlcv)
    assert set(pos.unique()).issubset({0.0, 1.0})
    assert pos.iloc[-1] == 1.0          # clean uptrend keeps breaking out higher


def test_bollinger_is_binary_and_indexed_like_input():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=200)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=idx).abs() + 5
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1},
        index=idx,
    )
    pos = BollingerMeanReversion(20, 2.0).signal(ohlcv)
    assert set(pos.unique()).issubset({0.0, 1.0})
    assert pos.index.equals(ohlcv.index)


def test_default_strategies_now_includes_breakout_strategies():
    names = {s.name for s in default_strategies()}
    assert any("Donchian" in n for n in names)
    assert any("Bollinger" in n for n in names)
