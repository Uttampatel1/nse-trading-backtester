"""Performance and risk metrics for an equity (portfolio value) curve."""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, trading_days: int = 252) -> float:
    years = len(equity) / trading_days
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1 / years) - 1.0)


def annualised_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    return float(returns.std(ddof=0) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if returns.std(ddof=0) == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(trading_days))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a (negative) fraction."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, trading_days: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, trading_days) / mdd)


def summary(
    equity: pd.Series,
    returns: pd.Series,
    n_trades: int,
    risk_free_rate: float = 0.06,
    trading_days: int = 252,
) -> dict[str, float]:
    """Bundle the headline metrics into a rounded dict."""
    return {
        "Total Return %": round(total_return(equity) * 100, 2),
        "CAGR %": round(cagr(equity, trading_days) * 100, 2),
        "Sharpe": round(sharpe_ratio(returns, risk_free_rate, trading_days), 2),
        "Max Drawdown %": round(max_drawdown(equity) * 100, 2),
        "Calmar": round(calmar_ratio(equity, trading_days), 2),
        "Volatility %": round(annualised_volatility(returns, trading_days) * 100, 2),
        "Trades": int(n_trades),
    }
