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

from .backtest import run_backtest, run_portfolio  # noqa: E402
from .config import get_settings  # noqa: E402
from .data import load_panel  # noqa: E402
from .logging_utils import get_logger, log_timing  # noqa: E402
from .strategies import SMACrossover, default_strategies  # noqa: E402
from .walkforward import optimize_on_slice, walk_forward_optimize  # noqa: E402

log = get_logger(__name__)


def run() -> pd.DataFrame:
    settings = get_settings()
    panel = load_panel(settings)
    log.info("Loaded %d tickers from source=%s", len(panel), settings.data_source)

    rows = []
    curves: dict[str, pd.Series] = {}
    for strategy in default_strategies():
        with log_timing(log, f"backtest {strategy.name}"):
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


def run_walkforward() -> pd.DataFrame:
    """In-sample-optimised vs walk-forward OOS SMA — quantifies overfitting."""
    settings = get_settings()
    panel = load_panel(settings)
    grid = {"fast": [5, 10, 20, 30], "slow": [50, 100, 150, 200]}

    rows = []
    for tkr, df in panel.items():
        best, _ = optimize_on_slice(df, SMACrossover, grid, "Sharpe", settings)
        is_stats = run_backtest(df, SMACrossover(**best), tkr, settings).metrics
        wf = walk_forward_optimize(df, SMACrossover, grid,
                                   is_size=378, oos_size=126, settings=settings)
        rows.append({
            "Ticker": tkr,
            "IS Sharpe (overfit)": is_stats["Sharpe"],
            "OOS Sharpe (honest)": wf.metrics["Sharpe"],
            "OOS Return %": wf.metrics["Total Return %"],
            "Windows": wf.n_windows,
        })
    table = pd.DataFrame(rows).set_index("Ticker")
    print("\n=== Walk-forward parameter optimization (SMA grid) ===")
    print("In-sample-optimised Sharpe is what a naive backtest reports; OOS is what")
    print("you'd actually have earned. The gap is overfitting.\n")
    print(table.to_string())
    print(f"\nMean IS Sharpe {table['IS Sharpe (overfit)'].mean():.2f} vs "
          f"OOS Sharpe {table['OOS Sharpe (honest)'].mean():.2f}")
    return table


if __name__ == "__main__":
    run()
    run_walkforward()
