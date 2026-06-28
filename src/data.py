"""Price data: a deterministic synthetic OHLCV generator plus an optional
live loader for real NSE history via Yahoo Finance.

The synthetic generator is the default so the whole project is reproducible
offline with no API keys. It builds daily prices with a geometric random walk
that switches between bull / bear / sideways regimes, which gives trend-following
and mean-reversion strategies something realistic to chew on.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import Settings, get_settings


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def _simulate_one(
    rng: np.random.Generator,
    n: int,
    start_price: float,
) -> pd.DataFrame:
    """Simulate a single ticker's OHLCV series with regime switching."""
    # Regime daily drift / volatility (per trading day).
    regimes = {
        "bull": (0.0009, 0.011),
        "bear": (-0.0011, 0.018),
        "side": (0.0001, 0.009),
    }
    names = list(regimes)
    # Expected regime length ~60 trading days; sample a regime path.
    regime_path: list[str] = []
    current = rng.choice(names)
    while len(regime_path) < n:
        length = int(rng.geometric(1 / 60))
        regime_path.extend([current] * length)
        current = rng.choice(names)
    regime_path = regime_path[:n]

    log_returns = np.empty(n)
    for i, r in enumerate(regime_path):
        mu, sigma = regimes[r]
        log_returns[i] = rng.normal(mu, sigma)

    close = start_price * np.exp(np.cumsum(log_returns))

    # Build OHLC around the close with a plausible intraday range.
    intraday = np.abs(rng.normal(0, 0.008, n)) + 0.002
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    open_ = np.clip(open_, low, high)
    volume = rng.integers(500_000, 5_000_000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_synthetic(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Generate a deterministic synthetic OHLCV panel keyed by ticker."""
    settings = settings or get_settings()
    rng = np.random.default_rng(settings.seed)
    index = _business_days(settings.start_date, settings.end_date)
    n = len(index)

    panel: dict[str, pd.DataFrame] = {}
    for ticker in settings.tickers:
        start_price = float(rng.uniform(300, 3500))
        df = _simulate_one(rng, n, start_price)
        df.index = index
        df.index.name = "date"
        panel[ticker] = df
    return panel


def load_yfinance(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Load real NSE history via yfinance (optional dependency)."""
    settings = settings or get_settings()
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "yfinance is not installed. Run `pip install yfinance` or use "
            "DATA_SOURCE=synthetic."
        ) from exc

    panel: dict[str, pd.DataFrame] = {}
    for ticker in settings.tickers:  # pragma: no cover - network path
        raw = yf.download(
            ticker,
            start=settings.start_date,
            end=settings.end_date,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            continue
        raw = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        raw.index.name = "date"
        panel[ticker] = raw[["open", "high", "low", "close", "volume"]]
    return panel


def load_panel(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Load prices according to the configured data source."""
    settings = settings or get_settings()
    if settings.data_source == "yfinance":
        return load_yfinance(settings)
    return generate_synthetic(settings)


def save_panel(panel: dict[str, pd.DataFrame], data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for ticker, df in panel.items():
        safe = ticker.replace(".", "_")
        df.to_csv(os.path.join(data_dir, f"{safe}.csv"))


def main() -> None:
    settings = get_settings()
    panel = load_panel(settings)
    save_panel(panel, settings.data_dir)
    total = sum(len(df) for df in panel.values())
    print(
        f"Wrote {len(panel)} tickers ({total} rows) to {settings.data_dir}/ "
        f"[source={settings.data_source}]"
    )


if __name__ == "__main__":
    main()
