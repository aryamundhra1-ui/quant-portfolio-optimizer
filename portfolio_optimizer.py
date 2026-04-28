import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data and calculate returns
prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
log_returns = np.log(prices / prices.shift(1)).dropna()

TRADING_DAYS = 252
tickers = list(prices.columns)
n_assets = len(tickers)
RISK_FREE_RATE = 0.05

print(f"Portfolio assets: {tickers}")
print(f"Number of assets: {n_assets}")

def portfolio_performance(weights, log_returns):
    weights = np.array(weights)
    port_return = np.sum(log_returns.mean() * weights) * TRADING_DAYS
    cov_matrix = log_returns.cov() * TRADING_DAYS
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - RISK_FREE_RATE) / port_volatility
    return port_return, port_volatility, sharpe_ratio

n_portfolios = 10000
results = np.zeros((3, n_portfolios))
all_weights = np.zeros((n_portfolios, n_assets))
np.random.seed(42)

print("Simulating 10,000 portfolios...")
for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights = weights / np.sum(weights)
    all_weights[i, :] = weights
    ret, vol, sharpe = portfolio_performance(weights, log_returns)
    results[0, i] = ret
    results[1, i] = vol
    results[2, i] = sharpe

print("Done!")

def neg_sharpe(weights):
    return -portfolio_performance(weights, log_returns)[2]

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.array([1/n_assets] * n_assets)

max_sharpe_result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
max_sharpe_weights = max_sharpe_result.x
max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = portfolio_performance(max_sharpe_weights, log_returns)

def portfolio_volatility(weights):
    return portfolio_performance(weights, log_returns)[1]

min_var_result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
min_var_weights = min_var_result.x
min_var_ret, min_var_vol, min_var_sr = portfolio_performance(min_var_weights, log_returns)

equal_weights = np.array([1/n_assets] * n_assets)
eq_ret, eq_vol, eq_sr = portfolio_performance(equal_weights, log_returns)

print("\n" + "="*55)
print("OPTIMAL PORTFOLIO RESULTS")
print("="*55)
print(f"\nMaximum Sharpe Portfolio:")
print(f"  Return:     {max_sharpe_ret:.2%}")
print(f"  Volatility: {max_sharpe_vol:.2%}")
print(f"  Sharpe:     {max_sharpe_sr:.3f}")
print(f"  Weights:")
for ticker, w in zip(tickers, max_sharpe_weights):
    print(f"    {ticker}: {w:.1%}")

print(f"\nMinimum Variance Portfolio:")
print(f"  Return:     {min_var_ret:.2%}")
print(f"  Volatility: {min_var_vol:.2%}")
print(f"  Sharpe:     {min_var_sr:.3f}")

print(f"\nEqual Weight Portfolio:")
print(f"  Return:     {eq_ret:.2%}")
print(f"  Volatility: {eq_vol:.2%}")
print(f"  Sharpe:     {eq_sr:.3f}")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='red', s=500, zorder=5, label=f'Max Sharpe (SR={max_sharpe_sr:.2f})')
plt.scatter(min_var_vol, min_var_ret, marker='D', color='blue', s=200, zorder=5, label=f'Min Variance (SR={min_var_sr:.2f})')
plt.scatter(eq_vol, eq_ret, marker='o', color='orange', s=200, zorder=5, label=f'Equal Weight (SR={eq_sr:.2f})')
plt.xlabel('Annual Volatility (Risk)', fontsize=13)
plt.ylabel('Annual Return', fontsize=13)
plt.title('Efficient Frontier — Portfolio Optimization (2019–2024)', fontsize=14)
plt.legend(fontsize=11)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEfficient Frontier saved!")