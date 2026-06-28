from __future__ import annotations

from src.config import Settings
from src.data import generate_synthetic


def _settings() -> Settings:
    return Settings(start_date="2020-01-01", end_date="2020-12-31", seed=7)


def test_panel_has_all_tickers_and_ohlcv_columns():
    panel = generate_synthetic(_settings())
    assert set(panel) == set(_settings().tickers)
    for df in panel.values():
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) > 200  # ~252 business days


def test_high_low_bracket_open_and_close():
    panel = generate_synthetic(_settings())
    for df in panel.values():
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["close"] - 1e-9).all()
        assert (df["low"] <= df["close"] + 1e-9).all()
        assert (df["close"] > 0).all()


def test_generation_is_deterministic():
    a = generate_synthetic(_settings())
    b = generate_synthetic(_settings())
    for t in a:
        assert a[t]["close"].equals(b[t]["close"])
