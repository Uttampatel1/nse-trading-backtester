# 📊 NSE Trading Strategy Backtester

**Business question:** *Does a simple, rules-based strategy actually beat buy-and-hold on Indian large-caps once you pay real trading costs — and does it control the gut-wrenching drawdowns?*

This project is a small but **honest** vectorised backtesting engine for NSE-style equities. It runs trend-following and mean-reversion rules against a buy-and-hold baseline across an equal-weight portfolio, charging costs on every trade and carefully avoiding look-ahead bias.

> ⚠️ Educational backtesting only — **not investment advice**. Default data is **synthetic and deterministic** so results are reproducible with no API key; flip a flag to run on real NSE history.

---

## Key results (5-ticker equal-weight portfolio, 2018–2023, after 0.10%/trade costs)

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown | Trades |
|----------|-------------:|-----:|-------:|-------------:|-------:|
| **SMA 20/50 (trend)** ✅ best | **+6.1%** | +0.96% | **-0.80** | **-22.4%** | 161 |
| Buy & Hold (baseline) | -21.0% | -3.73% | -1.03 | -33.8% | 5 |
| RSI(14) 30/55 (mean-reversion) | -22.8% | -4.07% | -2.28 | -26.6% | 108 |

*(Reproducible from `python -m src.run_backtest` on the default synthetic universe and seed.)*

**What this says:** over a choppy-to-bearish simulated regime, the **trend-following SMA crossover was the only strategy to end positive** and it cut the worst drawdown from **-34% to -22%** — because it moves to cash when the trend breaks. Mean-reversion "bought the dips" all the way down and underperformed even buy-and-hold.

**A practitioner's read on the result:**
- The headline isn't "we found alpha" — it's **risk control**. Trend rules earn their keep mainly by *avoiding* large losses, not by maximising upside.
- **Costs matter.** The SMA rule trades ~160 times; at higher cost-per-trade its edge erodes quickly. Always backtest net of realistic brokerage + STT + slippage (the `COST_PER_TRADE` knob).
- **Always benchmark against buy-and-hold.** Many "strategies" quietly lose to it once costs are in.
- Results are **regime-dependent**. A different synthetic seed (or real history over a bull run) can flip the ranking — which is exactly why a configurable, reproducible backtester matters.

## Demo

![Equity curves](data/equity_curves.png)

*Generated at `data/equity_curves.png` after running the backtest. Launch the dashboard to tune windows and inspect any single ticker.*

## How it works

```
synthetic OHLCV (regime-switching GBM)  ──►  Strategy.signal() ∈ {0,1}
   or real NSE via yfinance                        │  (shift +1 bar: trade next open)
                                                    ▼
                              position × open-to-open return − turnover×cost
                                                    │
                                                    ▼
                       equity curve ──► CAGR · Sharpe · Max Drawdown · Calmar
                                                    │
                                      equal-weight across tickers = portfolio
```

Every strategy implements one tiny interface (`signal(ohlcv) -> position`), so adding a new rule is a few lines. Signals are **shifted one bar** before execution, so a rule computed on today's close trades at tomorrow's open — no look-ahead.

## Tech stack

- **Engine:** pandas, NumPy (fully vectorised)
- **Metrics:** CAGR, annualised Sharpe, max drawdown, Calmar, volatility
- **Data:** deterministic synthetic generator (default) · optional `yfinance` for real NSE history
- **App:** Streamlit dashboard
- **Tests:** pytest (16 tests)

## Setup & run

```bash
cd 10-nse-trading-backtester
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data            # write synthetic OHLCV to data/
python -m src.run_backtest    # backtest all strategies, print table, save plot
streamlit run dashboard.py    # interactive explorer
pytest -q                     # run tests
```

**Optional — run on real NSE data:**

```bash
pip install yfinance
# set DATA_SOURCE=yfinance in your .env (copy from .env.example), then:
python -m src.run_backtest
```

## Project structure

```
10-nse-trading-backtester/
├── dashboard.py            # Streamlit strategy explorer
├── src/
│   ├── config.py           # typed settings from .env
│   ├── data.py             # synthetic OHLCV generator + optional yfinance loader
│   ├── strategies.py       # BuyAndHold, SMACrossover, RSIMeanReversion
│   ├── backtest.py         # vectorised engine (no look-ahead, costs on turnover)
│   ├── metrics.py          # CAGR / Sharpe / drawdown / Calmar
│   └── run_backtest.py     # full comparison + equity-curve plot
├── tests/                  # 16 pytest tests
├── .env.example
├── requirements.txt
└── .gitignore
```

## Possible extensions

- **Long/short and position sizing** (volatility targeting, Kelly-capped).
- **Walk-forward optimisation** to guard against curve-fitting the SMA windows.
- **Slippage & market-impact models** beyond a flat per-trade cost.
- **Benchmark vs NIFTY 50** and report alpha / beta / information ratio.
- **Parameter heatmaps** (fast × slow) to visualise robustness, not just the best cell.
```
