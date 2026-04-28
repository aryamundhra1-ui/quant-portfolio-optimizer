import numpy as np
import pandas as pd

# Your first financial calculation in Python!
apple_prices = [182.50, 184.20, 183.80, 186.10, 189.50]

prices = np.array(apple_prices)
daily_returns = np.diff(prices) / prices[:-1]

print("Apple daily returns:")
print(daily_returns)
print(f"Average daily return: {np.mean(daily_returns):.4f}")
print(f"Daily volatility:     {np.std(daily_returns):.4f}")