import yfinance as yf
import pandas as pd

# Download 5 years of Apple stock data
apple = yf.download("AAPL", start="2019-01-01", end="2024-01-01")

# Look at the first 5 rows
print(apple.head())

# How many rows and columns?
print(apple.shape)
# Output: (1258, 5) — 1258 trading days, 5 columns

# What are the columns?
print(apple.columns.tolist())
# ['Close', 'High', 'Low', 'Open', 'Volume']

# Last 5 rows — most recent data
print(apple.tail())

# Basic statistics for every column
print(apple.describe())
# Shows min, max, mean, std for Open, High, Low, Close, Volume

# Any missing data?
print(apple.isnull().sum())
# Should show 0 for all columns — yfinance data is very clean

import yfinance as yf
import pandas as pd

# Define your portfolio tickers
# SPY = S&P 500 index (your benchmark to beat)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]

# Download ALL of them at once — one line!
raw_data = yf.download(tickers, start="2019-01-01", end="2024-01-01")

# We only need the closing prices
prices = raw_data["Close"]

print(prices.head())
print(f"\nData shape: {prices.shape}")
print(f"\nAny missing values?\n{prices.isnull().sum()}")

# Drop any rows where ANY stock has missing data
prices = prices.dropna()

# Save to a CSV file so you never have to re-download
prices.to_csv("portfolio_prices.csv")

print(f"\nClean data shape: {prices.shape}")
print("Data saved to portfolio_prices.csv")

# In future, you can just load it instantly with:
# prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)