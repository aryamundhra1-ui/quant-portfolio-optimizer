import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data we saved in Module 2
prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)

print("Prices loaded!")
print(prices.tail())

# Calculate log returns
# np.log(today / yesterday) for every day, every stock
log_returns = np.log(prices / prices.shift(1)).dropna()

print(f"\nReturns shape: {log_returns.shape}")
print("\nFirst 5 rows of returns:")
print(log_returns.head())

# Number of trading days in a year
TRADING_DAYS = 252

# Annualized return for each stock
annual_return = log_returns.mean() * TRADING_DAYS

# Annualized volatility for each stock
annual_volatility = log_returns.std() * np.sqrt(TRADING_DAYS)

# Build a clean summary table
summary = pd.DataFrame({
    'Annual Return': annual_return,
    'Annual Volatility': annual_volatility,
    'Return/Risk Ratio': annual_return / annual_volatility
})

# Format as percentages for readability
print("\n--- RISK & RETURN SUMMARY ---")
print(f"{'Ticker':<8} {'Ann. Return':>12} {'Ann. Volatility':>16} {'Return/Risk':>12}")
print("-" * 50)
for ticker in summary.index:
    ret = summary.loc[ticker, 'Annual Return']
    vol = summary.loc[ticker, 'Annual Volatility']
    ratio = summary.loc[ticker, 'Return/Risk Ratio']
    print(f"{ticker:<8} {ret:>11.2%} {vol:>15.2%} {ratio:>12.2f}")

    # Calculate correlation between every pair of stocks
correlation_matrix = log_returns.corr()

print("\n--- CORRELATION MATRIX ---")
print(correlation_matrix.round(2))

# Create a professional correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,          # Show numbers inside each cell
    fmt='.2f',           # 2 decimal places
    cmap='RdYlGn',       # Red = low correlation, Green = high
    vmin=0, vmax=1,      # Scale from 0 to 1
    square=True,         # Make cells square
    linewidths=0.5,      # Grid lines between cells
    cbar_kws={'label': 'Correlation coefficient'}
)

plt.title('Portfolio Correlation Matrix (2019-2024)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nHeatmap saved as correlation_heatmap.png")

