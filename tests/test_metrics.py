from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_total_return():
    equity = pd.Series([100.0, 110.0, 121.0])
    assert metrics.total_return(equity) == pytest.approx(0.21)


def test_max_drawdown_is_negative_and_correct():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    # peak 120 -> trough 60 = -50%
    assert metrics.max_drawdown(equity) == -0.5


def test_sharpe_zero_when_no_volatility():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert metrics.sharpe_ratio(returns) == 0.0


def test_cagr_matches_compounding():
    # 2 years of 252 days, doubling
    n = 504
    equity = pd.Series(np.linspace(100, 200, n))
    cagr = metrics.cagr(equity, trading_days=252)
    assert abs(cagr - (2 ** 0.5 - 1)) < 1e-6


def test_summary_keys_present():
    equity = pd.Series(np.linspace(100, 150, 300))
    returns = equity.pct_change().fillna(0.0)
    out = metrics.summary(equity, returns, n_trades=5)
    for key in ["Total Return %", "CAGR %", "Sharpe", "Max Drawdown %", "Trades"]:
        assert key in out
    assert out["Trades"] == 5
