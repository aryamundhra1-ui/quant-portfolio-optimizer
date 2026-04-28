# quant-portfolio-optimizer
Quantitative portfolio optimizer with Efficient Frontier, backtesting engine, and Monte Carlo simulation - built in Python

# Quantitative Portfolio Optimizer & Backtesting Engine

A fully functional quantitative finance tool built in Python that implements
Nobel Prize-winning Modern Portfolio Theory to construct, optimize, and
stress-test a multi-asset portfolio using real market data.

---

## What this project does

- Pulls 5 years of real stock price data from Yahoo Finance (AAPL, MSFT, GOOGL, AMZN, SPY)
- Calculates log returns, annualized volatility, and a full correlation matrix
- Runs a Markowitz Mean-Variance Optimization across 10,000 random portfolios
- Plots the **Efficient Frontier** and identifies the Maximum Sharpe Ratio portfolio
- Backtests the optimized strategy vs. SPY benchmark and equal-weight portfolio
- Produces a professional **performance tearsheet** with Sharpe, Sortino, Calmar, and Max Drawdown
- Runs a **1,000-path Monte Carlo simulation** to forecast 1-year portfolio outcomes
- Outputs a master dashboard combining all visualizations

---

## Results

| Metric | Optimized (77% AAPL / 23% MSFT) | Equal Weight | SPY |
|---|---|---|---|
| Annual Return | ~28% | ~19% | ~13% |
| Annual Volatility | ~30% | ~27% | ~18% |
| Sharpe Ratio | 0.865 | ~0.62 | ~0.45 |
| Max Drawdown | -35.3% | -39.8% | -35.7% |
| $10k grew to | ~$38,000 | ~$24,000 | ~$18,435 |
| P(profit next year) | 82.9% | — | — |

## Project structure
quant_project/
│
├── data_acquisition.py       # Pulls real price data via yfinance
├── risk_analysis.py          # Log returns, volatility, correlation heatmap
├── portfolio_optimizer.py    # Efficient Frontier + Markowitz optimization
├── backtesting.py            # Strategy backtest vs SPY benchmark
├── performance_metrics.py    # Sharpe, Sortino, Calmar, VaR tearsheet
├── monte_carlo.py            # 1,000-path Monte Carlo simulation
├── master_dashboard.py       # Final 6-panel master dashboard
└── portfolio_prices.csv      # Saved historical price data

---

## Setup instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/quant-portfolio-optimizer.git
cd quant-portfolio-optimizer

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install numpy pandas yfinance matplotlib seaborn scipy

# 4. Run in order
python data_acquisition.py
python risk_analysis.py
python portfolio_optimizer.py
python backtesting.py
python performance_metrics.py
python monte_carlo.py
python master_dashboard.py
```

---

## Key concepts implemented

**Modern Portfolio Theory (Markowitz, 1952)** — Mathematical framework for
constructing portfolios that maximize return for a given level of risk.

**Efficient Frontier** — The set of optimal portfolios that offer the highest
expected return for each level of risk. Visualized as a scatter plot of 10,000
simulated weight combinations.

**Sharpe Ratio** — Return earned per unit of total risk (excess return / volatility).
The single most cited risk-adjusted performance metric in finance.

**Sortino Ratio** — Like Sharpe but only penalizes downside volatility. More
appropriate for asymmetric return distributions.

**Value at Risk (VaR)** — The maximum expected loss on 95% of trading days.
Standard risk metric used by every major bank and asset manager.

**Monte Carlo Simulation** — Uses geometric Brownian motion to simulate 1,000
possible future portfolio paths, generating a probability distribution of outcomes.

---

## Technologies used

- **Python 3.14**
- **NumPy** — vectorized math and matrix operations
- **Pandas** — time series data manipulation
- **yfinance** — real market data acquisition
- **Matplotlib / Seaborn** — professional financial visualizations
- **SciPy** — constrained portfolio optimization (SLSQP)

---

## Author

Built as a quantitative finance project applying Modern Portfolio Theory
to real market data. Demonstrates end-to-end quant workflow from data
acquisition through risk management and forward simulation.

---

## Project structure
