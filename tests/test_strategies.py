from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import BuyAndHold, RSIMeanReversion, SMACrossover


def _trending_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000},
        index=idx,
    )


def test_buy_and_hold_is_always_invested():
    pos = BuyAndHold().signal(_trending_ohlcv())
    assert (pos == 1.0).all()


def test_sma_requires_fast_below_slow():
    with pytest.raises(ValueError):
        SMACrossover(50, 20)


def test_sma_goes_long_in_uptrend_after_warmup():
    pos = SMACrossover(10, 30).signal(_trending_ohlcv())
    assert pos.iloc[:29].sum() == 0  # flat until slow window warm
    assert pos.iloc[-1] == 1.0       # long in a clean uptrend
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_rsi_positions_are_binary():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=200)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=idx).abs() + 1
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1},
        index=idx,
    )
    pos = RSIMeanReversion(14, 30, 55).signal(ohlcv)
    assert set(pos.unique()).issubset({0.0, 1.0})
    assert len(pos) == len(ohlcv)
