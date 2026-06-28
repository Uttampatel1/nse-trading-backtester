"""Trading strategies.

Each strategy is a callable that turns an OHLCV frame into a *target position*
series in {0, 1} (flat or long), aligned to the price index. Signals are shifted
by one bar inside the backtester so a signal computed on day *t*'s close is acted
on at day *t+1*'s open — this avoids look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    name: str

    def signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Return a target position series in {0, 1} indexed like ``ohlcv``."""
        ...


@dataclass
class BuyAndHold:
    """Baseline: always fully invested. Every strategy must beat this."""

    name: str = "Buy & Hold"

    def signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=ohlcv.index, name="position")


@dataclass
class SMACrossover:
    """Trend following: long when the fast SMA is above the slow SMA."""

    fast: int = 20
    slow: int = 50
    name: str = "SMA Crossover"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast window must be shorter than slow window")
        self.name = f"SMA {self.fast}/{self.slow}"

    def signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        pos = (fast > slow).astype(float)
        pos[slow.isna()] = 0.0  # no position until the slow window is warm
        return pos.rename("position")


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


@dataclass
class RSIMeanReversion:
    """Mean reversion: buy oversold dips, exit when momentum recovers."""

    period: int = 14
    oversold: float = 30.0
    exit_level: float = 55.0
    name: str = "RSI Mean Reversion"

    def __post_init__(self) -> None:
        self.name = f"RSI({self.period}) {self.oversold:.0f}/{self.exit_level:.0f}"

    def signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        rsi = _rsi(ohlcv["close"], self.period)
        pos = np.zeros(len(rsi))
        holding = False
        for i, value in enumerate(rsi.to_numpy()):
            if not holding and value < self.oversold:
                holding = True
            elif holding and value > self.exit_level:
                holding = False
            pos[i] = 1.0 if holding else 0.0
        return pd.Series(pos, index=ohlcv.index, name="position")


def default_strategies() -> list[Strategy]:
    return [BuyAndHold(), SMACrossover(20, 50), RSIMeanReversion(14, 30, 55)]
