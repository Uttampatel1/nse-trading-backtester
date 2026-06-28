"""Run every strategy across the universe and print a ranked comparison.

Usage::

    python -m src.run_backtest
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import run_portfolio  # noqa: E402
from .config import get_settings  # noqa: E402
from .data import load_panel  # noqa: E402
from .strategies import default_strategies  # noqa: E402


def run() -> pd.DataFrame:
    settings = get_settings()
    panel = load_panel(settings)

    rows = []
    curves: dict[str, pd.Series] = {}
    for strategy in default_strategies():
        result = run_portfolio(panel, strategy, settings)
        rows.append({"Strategy": result.strategy, **result.metrics})
        curves[result.strategy] = result.equity

    table = (
        pd.DataFrame(rows)
        .set_index("Strategy")
        .sort_values("Sharpe", ascending=False)
    )

    os.makedirs(settings.data_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for name, equity in curves.items():
        plt.plot(equity.index, equity.values, label=name)
    plt.title("Equity curves — equal-weight NSE portfolio (after costs)")
    plt.ylabel("Portfolio value (INR)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(settings.data_dir, "equity_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()

    print(f"\nData source: {settings.data_source} | tickers: {len(panel)} | "
          f"cost/trade: {settings.cost_per_trade:.2%}\n")
    print(table.to_string())
    print(f"\nSaved equity-curve plot to {out}")
    return table


if __name__ == "__main__":
    run()
