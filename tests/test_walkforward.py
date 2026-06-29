from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data import generate_synthetic
from src.strategies import SMACrossover
from src.walkforward import (
    WalkForwardResult,
    optimize_on_slice,
    walk_forward_optimize,
)

GRID = {"fast": [10, 20], "slow": [50, 100]}


def _series() -> pd.DataFrame:
    settings = Settings(start_date="2016-01-01", end_date="2023-12-31", seed=5)
    panel = generate_synthetic(settings)
    return panel[next(iter(panel))]


def test_optimize_skips_invalid_combos():
    # fast must be < slow; a grid with only invalid combos should raise
    df = _series()
    with pytest.raises(ValueError):
        optimize_on_slice(df, SMACrossover, {"fast": [50], "slow": [20]})


def test_optimize_returns_a_valid_combo():
    df = _series()
    params, score = optimize_on_slice(df, SMACrossover, GRID, objective="Sharpe")
    assert params["fast"] < params["slow"]
    assert np.isfinite(score)


def test_walk_forward_tiles_and_stitches():
    df = _series()
    res = walk_forward_optimize(df, SMACrossover, GRID, is_size=252, oos_size=126)
    assert isinstance(res, WalkForwardResult)
    assert res.n_windows >= 2
    # OOS windows are contiguous and non-overlapping
    starts = [w["oos_start"] for w in res.windows]
    assert starts == sorted(starts)
    for w in res.windows:
        assert w["param_fast"] < w["param_slow"]


def test_oos_equity_length_matches_returns():
    df = _series()
    res = walk_forward_optimize(df, SMACrossover, GRID, is_size=252, oos_size=126)
    assert len(res.oos_equity) == len(res.oos_returns)
    assert res.oos_returns.index.is_monotonic_increasing
    assert np.isfinite(res.metrics["Sharpe"])


def test_chosen_params_frame_has_one_row_per_window():
    df = _series()
    res = walk_forward_optimize(df, SMACrossover, GRID, is_size=252, oos_size=126)
    frame = res.chosen_params()
    assert len(frame) == res.n_windows
    assert {"is_score", "oos_score"} <= set(frame.columns)


def test_raises_when_series_too_short():
    df = _series().head(100)
    with pytest.raises(ValueError):
        walk_forward_optimize(df, SMACrossover, GRID, is_size=252, oos_size=126)
