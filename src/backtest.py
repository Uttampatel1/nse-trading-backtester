"""A vectorised single-asset backtest engine.

Design choices that keep the results honest:

* **No look-ahead.** A signal computed on the close of day *t* is executed at the
  *open* of day *t+1*. We model the bar-to-bar return as open-to-open so the
  position held over a day earns that day's move.
* **Costs on turnover.** Every change in position pays ``cost_per_trade`` on the
  traded notional (a proxy for brokerage + STT + slippage).
* **Long/flat only.** Positions are in {0, 1}; shorting is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .config import Settings, get_settings
from .strategies import Strategy


@dataclass
class BacktestResult:
    strategy: str
    ticker: str
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    n_trades: int
    metrics: dict[str, float]


def run_backtest(
    ohlcv: pd.DataFrame,
    strategy: Strategy,
    ticker: str = "",
    settings: Settings | None = None,
) -> BacktestResult:
    """Backtest one strategy on one ticker's OHLCV frame."""
    settings = settings or get_settings()

    # Target position from the strategy, shifted one bar to trade on next open.
    target = strategy.signal(ohlcv).fillna(0.0)
    position = target.shift(1).fillna(0.0)

    # Open-to-open returns of the underlying.
    open_ = ohlcv["open"]
    asset_ret = open_.pct_change().fillna(0.0)

    # Turnover = |Δ position|; cost charged on the bar the trade happens.
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * settings.cost_per_trade

    strat_ret = position * asset_ret - cost
    equity = settings.initial_capital * (1 + strat_ret).cumprod()

    n_trades = int((turnover > 0).sum())
    stats = metrics.summary(
        equity,
        strat_ret,
        n_trades,
        risk_free_rate=settings.risk_free_rate,
        trading_days=settings.trading_days,
    )

    return BacktestResult(
        strategy=strategy.name,
        ticker=ticker,
        equity=equity.rename("equity"),
        returns=strat_ret.rename("returns"),
        position=position.rename("position"),
        n_trades=n_trades,
        metrics=stats,
    )


def run_portfolio(
    panel: dict[str, pd.DataFrame],
    strategy: Strategy,
    settings: Settings | None = None,
) -> BacktestResult:
    """Backtest a strategy as an equal-weight portfolio across all tickers.

    Capital is split equally across tickers; the portfolio equity is the sum of
    the per-ticker equity curves on the shared (intersected) date index.
    """
    settings = settings or get_settings()
    per_ticker = {
        t: run_backtest(df, strategy, t, settings) for t, df in panel.items()
    }
    if not per_ticker:
        raise ValueError("empty panel")

    weight = 1.0 / len(per_ticker)
    sub_settings = Settings(
        **{**settings.__dict__, "initial_capital": settings.initial_capital * weight}
    )
    # Re-run with per-ticker capital so the curves sum to the full account.
    curves = [
        run_backtest(panel[t], strategy, t, sub_settings).equity for t in per_ticker
    ]
    equity = pd.concat(curves, axis=1).dropna().sum(axis=1).rename("equity")
    port_ret = equity.pct_change().fillna(0.0).rename("returns")
    n_trades = sum(r.n_trades for r in per_ticker.values())

    stats = metrics.summary(
        equity,
        port_ret,
        n_trades,
        risk_free_rate=settings.risk_free_rate,
        trading_days=settings.trading_days,
    )
    return BacktestResult(
        strategy=strategy.name,
        ticker="PORTFOLIO",
        equity=equity,
        returns=port_ret,
        position=pd.Series(dtype=float),
        n_trades=n_trades,
        metrics=stats,
    )
