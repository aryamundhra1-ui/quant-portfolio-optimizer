import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load data ──────────────────────────────────────────────
prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
log_returns = np.log(prices / prices.shift(1)).dropna()

# ── Portfolio weights from Module 4 optimizer ──────────────
INITIAL_CAPITAL = 10000      # Start with $10,000
TRADING_DAYS    = 252

optimized_weights  = np.array([0.773, 0.000, 0.000, 0.227, 0.000])  # AAPL, AMZN, GOOGL, MSFT, SPY
equal_weights      = np.array([0.200, 0.200, 0.200, 0.200, 0.200])
spy_weights        = np.array([0.000, 0.000, 0.000, 0.000, 1.000])   # 100% SPY

# ── Core backtesting function ───────────────────────────────
def backtest(weights, returns, initial_capital=10000):
    """
    Simulate investing initial_capital with fixed weights.
    Returns a Series of daily portfolio values.
    """
    weights = np.array(weights)

    # Daily portfolio return = weighted sum of each stock's daily return
    # e.g. if AAPL is up 1% and we hold 77.3%, it contributes 0.773 * 0.01
    daily_portfolio_returns = returns.dot(weights)

    # Convert log returns to simple returns, then compound them
    # Starting at 1.0, multiply each day's growth factor
    cumulative_growth = (1 + daily_portfolio_returns).cumprod()

    # Scale to dollar value
    portfolio_value = cumulative_growth * initial_capital

    return portfolio_value

# ── Run all three backtests ─────────────────────────────────
optimized_values  = backtest(optimized_weights,  log_returns)
equal_values      = backtest(equal_weights,       log_returns)
spy_values        = backtest(spy_weights,         log_returns)

# ── Print final results ─────────────────────────────────────
def print_summary(name, values, initial=INITIAL_CAPITAL):
    final_value   = values.iloc[-1]
    total_return  = (final_value - initial) / initial
    years         = len(values) / TRADING_DAYS
    annual_return = (final_value / initial) ** (1 / years) - 1
    print(f"\n{name}")
    print(f"  Starting value:  ${initial:,.0f}")
    print(f"  Final value:     ${final_value:,.0f}")
    print(f"  Total return:    {total_return:.1%}")
    print(f"  Annual return:   {annual_return:.1%}")

print("=" * 45)
print("BACKTEST RESULTS — $10,000 invested Jan 2019")
print("=" * 45)
print_summary("Optimized Portfolio (77% AAPL / 23% MSFT)", optimized_values)
print_summary("Equal Weight (20% each)",                   equal_values)
print_summary("SPY Benchmark",                             spy_values)

# ── Drawdown calculator ─────────────────────────────────────
def calculate_drawdown(portfolio_values):
    """
    Drawdown = how far the portfolio has fallen from its previous peak.
    Max drawdown = the worst single drop from peak to trough.
    """
    rolling_max  = portfolio_values.cummax()
    drawdown     = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

opt_dd,  opt_max_dd  = calculate_drawdown(optimized_values)
eq_dd,   eq_max_dd   = calculate_drawdown(equal_values)
spy_dd,  spy_max_dd  = calculate_drawdown(spy_values)

print("\n--- MAX DRAWDOWN (worst peak-to-trough drop) ---")
print(f"  Optimized:    {opt_max_dd:.1%}")
print(f"  Equal Weight: {eq_max_dd:.1%}")
print(f"  SPY:          {spy_max_dd:.1%}")

# ── Plot 1: Portfolio growth curves ────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10),
                                gridspec_kw={'height_ratios': [3, 1]})

# Growth chart
ax1.plot(optimized_values.index, optimized_values,
         color='#E63946', linewidth=2.5, label='Optimized (77% AAPL / 23% MSFT)')
ax1.plot(equal_values.index,     equal_values,
         color='#FF9800', linewidth=1.8, linestyle='--', label='Equal Weight')
ax1.plot(spy_values.index,       spy_values,
         color='#2196F3', linewidth=1.8, linestyle=':', label='SPY Benchmark')

# Mark the COVID crash (March 2020) and 2022 rate hike crash
ax1.axvline(pd.Timestamp('2020-03-23'), color='gray', linestyle='--',
            alpha=0.6, linewidth=1)
ax1.text(pd.Timestamp('2020-03-25'), ax1.get_ylim()[0] * 1.05,
         'COVID\nCrash', fontsize=8, color='gray')

ax1.axvline(pd.Timestamp('2022-01-03'), color='gray', linestyle='--',
            alpha=0.6, linewidth=1)
ax1.text(pd.Timestamp('2022-01-10'), ax1.get_ylim()[0] * 1.05,
         '2022\nRate Hikes', fontsize=8, color='gray')

ax1.axhline(INITIAL_CAPITAL, color='black', linestyle='-', alpha=0.2, linewidth=1)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.set_title('Backtesting Results — $10,000 Initial Investment (2019–2024)',
              fontsize=14, pad=15)
ax1.legend(fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Drawdown chart
ax2.fill_between(opt_dd.index,  opt_dd,  alpha=0.4, color='#E63946', label='Optimized')
ax2.fill_between(eq_dd.index,   eq_dd,   alpha=0.4, color='#FF9800', label='Equal Weight')
ax2.fill_between(spy_dd.index,  spy_dd,  alpha=0.4, color='#2196F3', label='SPY')
ax2.set_ylabel('Drawdown', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nBacktest chart saved as backtest_results.png")