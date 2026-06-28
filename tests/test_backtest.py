from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest, run_portfolio
from src.config import Settings
from src.data import generate_synthetic
from src.strategies import BuyAndHold, SMACrossover


def _settings() -> Settings:
    return Settings(start_date="2019-01-01", end_date="2022-12-31", seed=3)


def _flat_ohlcv(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=n)
    s = pd.Series(price, index=idx)
    return pd.DataFrame(
        {"open": s, "high": s, "low": s, "close": s, "volume": 1}, index=idx
    )


def test_buy_and_hold_on_flat_prices_keeps_capital():
    # With zero costs, holding flat prices must preserve capital exactly.
    settings = Settings(**{**_settings().__dict__, "cost_per_trade": 0.0})
    result = run_backtest(_flat_ohlcv(), BuyAndHold(), "FLAT", settings)
    assert abs(result.equity.iloc[-1] - settings.initial_capital) < 1.0
    assert result.n_trades <= 1


def test_no_lookahead_first_bar_is_flat():
    settings = _settings()
    panel = generate_synthetic(settings)
    ticker = next(iter(panel))
    result = run_backtest(panel[ticker], SMACrossover(20, 50), ticker, settings)
    # position is shifted, so the first bar must be flat (0)
    assert result.position.iloc[0] == 0.0


def test_costs_reduce_returns_for_active_strategy():
    settings = _settings()
    panel = generate_synthetic(settings)
    ticker = next(iter(panel))
    no_cost = Settings(**{**settings.__dict__, "cost_per_trade": 0.0})
    with_cost = Settings(**{**settings.__dict__, "cost_per_trade": 0.01})
    strat = SMACrossover(20, 50)
    r0 = run_backtest(panel[ticker], strat, ticker, no_cost)
    r1 = run_backtest(panel[ticker], strat, ticker, with_cost)
    assert r1.equity.iloc[-1] < r0.equity.iloc[-1]


def test_portfolio_equity_starts_near_initial_capital():
    settings = _settings()
    panel = generate_synthetic(settings)
    result = run_portfolio(panel, BuyAndHold(), settings)
    assert abs(result.equity.iloc[0] - settings.initial_capital) < settings.initial_capital * 0.01
    assert result.metrics["Trades"] >= 0
    assert np.isfinite(result.metrics["Sharpe"])
