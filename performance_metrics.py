import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load data ───────────────────────────────────────────────
prices      = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
log_returns = np.log(prices / prices.shift(1)).dropna()

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05

# ── Portfolio weights ───────────────────────────────────────
weights = {
    "Optimized":   np.array([0.773, 0.000, 0.000, 0.227, 0.000]),
    "Equal Weight":np.array([0.200, 0.200, 0.200, 0.200, 0.200]),
    "SPY":         np.array([0.000, 0.000, 0.000, 0.000, 1.000]),
}

# ── Calculate daily portfolio returns ───────────────────────
portfolio_returns = {}
portfolio_values  = {}

for name, w in weights.items():
    daily_ret = log_returns.dot(w)
    portfolio_returns[name] = daily_ret
    portfolio_values[name]  = (1 + daily_ret).cumprod() * 10000

# ── Master metrics function ─────────────────────────────────
def calculate_metrics(daily_returns, name="Portfolio"):
    """
    Takes a Series of daily returns and returns
    a full dictionary of performance metrics.
    """
    r = daily_returns

    # Annualized return and volatility
    ann_return = r.mean() * TRADING_DAYS
    ann_vol    = r.std()  * np.sqrt(TRADING_DAYS)

    # Sharpe Ratio
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol

    # Sortino Ratio — only uses DOWNSIDE returns in denominator
    downside_returns = r[r < 0]
    downside_vol     = downside_returns.std() * np.sqrt(TRADING_DAYS)
    sortino          = (ann_return - RISK_FREE_RATE) / downside_vol

    # Max Drawdown
    cumulative   = (1 + r).cumprod()
    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio = Annual Return / |Max Drawdown|
    calmar = ann_return / abs(max_drawdown)

    # Win rate — % of days with positive returns
    win_rate = (r > 0).sum() / len(r)

    # Best and worst single day
    best_day  = r.max()
    worst_day = r.min()

    return {
        "Annual Return":    ann_return,
        "Annual Volatility":ann_vol,
        "Sharpe Ratio":     sharpe,
        "Sortino Ratio":    sortino,
        "Max Drawdown":     max_drawdown,
        "Calmar Ratio":     calmar,
        "Win Rate":         win_rate,
        "Best Day":         best_day,
        "Worst Day":        worst_day,
    }

# ── Run metrics on all three strategies ────────────────────
all_metrics = {}
for name, returns in portfolio_returns.items():
    all_metrics[name] = calculate_metrics(returns, name)

# ── Print professional tearsheet table ─────────────────────
metrics_df = pd.DataFrame(all_metrics).T

print("\n" + "="*65)
print("  PORTFOLIO PERFORMANCE REPORT")
print("="*65)
print(f"\n{'Metric':<22} {'Optimized':>14} {'Equal Weight':>14} {'SPY':>10}")
print("-"*65)

pct_metrics = ["Annual Return", "Annual Volatility", "Max Drawdown",
               "Win Rate", "Best Day", "Worst Day"]
raw_metrics = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]

for metric in pct_metrics + raw_metrics:
    row = f"{metric:<22}"
    for strat in ["Optimized", "Equal Weight", "SPY"]:
        val = all_metrics[strat][metric]
        if metric in pct_metrics:
            row += f"{val:>13.2%} "
        else:
            row += f"{val:>13.3f} "
    print(row)

print("="*65)

# ── Plot: Professional 4-panel tearsheet ───────────────────
fig = plt.figure(figsize=(15, 12))
fig.suptitle("Portfolio Performance Tearsheet (2019–2024)",
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colors = {"Optimized": "#E63946", "Equal Weight": "#FF9800", "SPY": "#2196F3"}

# ── Panel 1: Return distributions ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for name, ret in portfolio_returns.items():
    ax1.hist(ret, bins=80, alpha=0.5, color=colors[name],
             label=name, density=True)
ax1.axvline(0, color='black', linewidth=1, linestyle='--')
ax1.set_title("Daily Return Distribution", fontsize=12)
ax1.set_xlabel("Daily Return")
ax1.set_ylabel("Density")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ── Panel 2: Rolling 6-month Sharpe Ratio ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
window = 126   # ~6 months of trading days

for name, ret in portfolio_returns.items():
    rolling_sharpe = (
        ret.rolling(window).mean() * TRADING_DAYS -
        RISK_FREE_RATE
    ) / (ret.rolling(window).std() * np.sqrt(TRADING_DAYS))
    ax2.plot(rolling_sharpe.index, rolling_sharpe,
             color=colors[name], label=name, linewidth=1.5)

ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax2.axhline(1, color='green', linewidth=1, linestyle=':', alpha=0.5)
ax2.set_title("Rolling 6-Month Sharpe Ratio", fontsize=12)
ax2.set_ylabel("Sharpe Ratio")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── Panel 3: Drawdown chart ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name, ret in portfolio_returns.items():
    cumulative  = (1 + ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    ax3.fill_between(drawdown.index, drawdown, 0,
                     alpha=0.4, color=colors[name], label=name)
    ax3.plot(drawdown.index, drawdown,
             color=colors[name], linewidth=0.8)

ax3.set_title("Drawdown Over Time", fontsize=12)
ax3.set_ylabel("Drawdown")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# ── Panel 4: Monthly returns heatmap (Optimized only) ──────
ax4 = fig.add_subplot(gs[1, 1])
opt_returns = portfolio_returns["Optimized"].copy()
opt_returns.index = pd.to_datetime(opt_returns.index)

# Resample to monthly returns
monthly = opt_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
monthly_df = monthly.to_frame(name='Return')
monthly_df['Year']  = monthly_df.index.year
monthly_df['Month'] = monthly_df.index.month

pivot = monthly_df.pivot_table(
    values='Return', index='Year', columns='Month'
)
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']

sns.heatmap(
    pivot, annot=True, fmt='.1%', cmap='RdYlGn',
    center=0, linewidths=0.5, ax=ax4,
    cbar_kws={'label': 'Monthly Return'},
    annot_kws={'size': 8}
)
ax4.set_title("Optimized Portfolio — Monthly Returns", fontsize=12)
ax4.set_xlabel("")

plt.savefig('tearsheet.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTearsheet saved as tearsheet.png")