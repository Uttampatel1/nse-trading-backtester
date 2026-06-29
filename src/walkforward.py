"""Walk-forward parameter optimization — the honest way to tune a strategy.

Grid-searching parameters on your whole price history and reporting the best result
is how backtests lie: you're reading the noise you just fit to. Walk-forward
optimization fixes that. It slides a window across time and, at each step:

1. picks the best parameters on an **in-sample** (IS) block, then
2. trades those parameters on the **next, unseen out-of-sample** (OOS) block.

The OOS blocks are stitched into one continuous equity curve built entirely from
decisions made *before* the data they traded on — a realistic estimate of live
performance. The gap between the in-sample-optimised result and this walk-forward
result is a direct read on **overfitting**: a strategy that looks brilliant in-sample
but mediocre out-of-sample was curve-fit, not discovered.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics
from .backtest import run_backtest
from .config import Settings, get_settings
from .strategies import Strategy

# Objectives where a larger value is better (covers everything in metrics.summary
# except where "less negative" is better, handled below).
_HIGHER_IS_BETTER = {
    "Total Return %", "CAGR %", "Sharpe", "Sortino", "Calmar", "Max Drawdown %",
}

StrategyFactory = Callable[..., Strategy]


def _param_combos(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def _score(stats: dict[str, float], objective: str) -> float:
    """Objective value to maximise (Max Drawdown's 'best' is the least-negative)."""
    return float(stats[objective])


def optimize_on_slice(
    ohlcv: pd.DataFrame,
    factory: StrategyFactory,
    param_grid: dict[str, list],
    objective: str = "Sharpe",
    settings: Settings | None = None,
) -> tuple[dict, float]:
    """Best parameter combo on one slice, by ``objective`` (skips invalid combos)."""
    settings = settings or get_settings()
    best_params, best_score = None, -np.inf
    for params in _param_combos(param_grid):
        try:
            strat = factory(**params)
        except ValueError:
            continue  # e.g. SMA fast >= slow
        stats = run_backtest(ohlcv, strat, settings=settings).metrics
        score = _score(stats, objective)
        if np.isfinite(score) and score > best_score:
            best_params, best_score = params, score
    if best_params is None:
        raise ValueError("no valid parameter combination in the grid")
    return best_params, best_score


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    oos_equity: pd.Series
    metrics: dict[str, float]
    windows: list[dict] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def chosen_params(self) -> pd.DataFrame:
        """One row per window: the IS-optimal params and IS vs OOS objective."""
        return pd.DataFrame(self.windows)


def walk_forward_optimize(
    ohlcv: pd.DataFrame,
    factory: StrategyFactory,
    param_grid: dict[str, list],
    is_size: int,
    oos_size: int,
    objective: str = "Sharpe",
    settings: Settings | None = None,
) -> WalkForwardResult:
    """Tile the series with IS→OOS windows and stitch the OOS returns together.

    Each window optimises on ``ohlcv[start : start+is_size]`` and trades the chosen
    parameters on the following ``oos_size`` bars; the OOS block is evaluated on the
    IS+OOS slice (so indicators are warmed by the IS lead-in) and only the OOS
    portion is kept. Windows step forward by ``oos_size`` (non-overlapping OOS).
    """
    settings = settings or get_settings()
    n = len(ohlcv)
    if is_size <= 0 or oos_size <= 0:
        raise ValueError("is_size and oos_size must be positive")
    if is_size + oos_size > n:
        raise ValueError("series too short for one IS+OOS window")

    oos_return_chunks: list[pd.Series] = []
    windows: list[dict] = []

    start = 0
    while start + is_size + oos_size <= n:
        is_slice = ohlcv.iloc[start : start + is_size]
        full_slice = ohlcv.iloc[start : start + is_size + oos_size]
        oos_index = ohlcv.index[start + is_size : start + is_size + oos_size]

        best_params, is_score = optimize_on_slice(
            is_slice, factory, param_grid, objective, settings
        )

        # Trade the chosen params across IS+OOS, keep only the OOS returns.
        full_res = run_backtest(full_slice, factory(**best_params), settings=settings)
        oos_ret = full_res.returns.loc[oos_index]
        oos_return_chunks.append(oos_ret)

        oos_stats = metrics.summary(
            (1 + oos_ret).cumprod() * settings.initial_capital,
            oos_ret, n_trades=0,
            risk_free_rate=settings.risk_free_rate, trading_days=settings.trading_days,
        )
        windows.append({
            "is_start": is_slice.index[0],
            "oos_start": oos_index[0],
            "oos_end": oos_index[-1],
            **{f"param_{k}": v for k, v in best_params.items()},
            "is_score": round(is_score, 3),
            "oos_score": round(_score(oos_stats, objective), 3),
        })
        start += oos_size

    oos_returns = pd.concat(oos_return_chunks)
    oos_equity = (settings.initial_capital * (1 + oos_returns).cumprod()).rename("equity")
    stats = metrics.summary(
        oos_equity, oos_returns, n_trades=0,
        risk_free_rate=settings.risk_free_rate, trading_days=settings.trading_days,
    )
    return WalkForwardResult(
        oos_returns=oos_returns.rename("returns"),
        oos_equity=oos_equity,
        metrics=stats,
        windows=windows,
    )
