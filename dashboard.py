"""Streamlit dashboard to explore strategies on the NSE universe.

Run with::

    streamlit run dashboard.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest import run_backtest, run_portfolio
from src.config import get_settings
from src.data import load_panel
from src.strategies import BuyAndHold, RSIMeanReversion, SMACrossover

st.set_page_config(page_title="NSE Backtester", layout="wide")
settings = get_settings()


@st.cache_data
def _panel():
    return load_panel(settings)


panel = _panel()

st.title("📊 NSE Trading Backtester")
st.caption(
    f"Data source: **{settings.data_source}** · cost/trade "
    f"{settings.cost_per_trade:.2%} · capital ₹{settings.initial_capital:,.0f}"
)

with st.sidebar:
    st.header("Strategy")
    kind = st.selectbox("Type", ["SMA Crossover", "RSI Mean Reversion", "Buy & Hold"])
    if kind == "SMA Crossover":
        fast = st.slider("Fast SMA", 5, 60, 20)
        slow = st.slider("Slow SMA", 30, 200, 50)
        strategy = SMACrossover(fast, min(slow, 200) if slow > fast else fast + 10)
    elif kind == "RSI Mean Reversion":
        period = st.slider("RSI period", 5, 30, 14)
        oversold = st.slider("Oversold", 10, 45, 30)
        exit_level = st.slider("Exit level", 50, 80, 55)
        strategy = RSIMeanReversion(period, oversold, exit_level)
    else:
        strategy = BuyAndHold()

    scope = st.radio("Scope", ["Portfolio", *panel.keys()])

if scope == "Portfolio":
    result = run_portfolio(panel, strategy, settings)
    bh = run_portfolio(panel, BuyAndHold(), settings)
else:
    result = run_backtest(panel[scope], strategy, scope, settings)
    bh = run_backtest(panel[scope], BuyAndHold(), scope, settings)

cols = st.columns(len(result.metrics))
for col, (k, v) in zip(cols, result.metrics.items()):
    col.metric(k, v)

st.subheader("Equity curve vs Buy & Hold")
chart = pd.DataFrame({strategy.name: result.equity, "Buy & Hold": bh.equity}).dropna()
st.line_chart(chart)

st.subheader("Metrics — strategy vs baseline")
st.dataframe(
    pd.DataFrame({strategy.name: result.metrics, "Buy & Hold": bh.metrics})
)
