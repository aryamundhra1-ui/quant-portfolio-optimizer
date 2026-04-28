import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import minimize

# ════════════════════════════════════════════════════════════
# LOAD & PREPARE ALL DATA
# ════════════════════════════════════════════════════════════
prices      = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
log_returns = np.log(prices / prices.shift(1)).dropna()

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05
INITIAL_VALUE  = 10000
tickers        = list(prices.columns)

opt_weights  = np.array([0.773, 0.000, 0.000, 0.227, 0.000])
eq_weights   = np.array([0.200, 0.200, 0.200, 0.200, 0.200])
spy_weights  = np.array([0.000, 0.000, 0.000, 0.000, 1.000])

all_weights  = {"Optimized": opt_weights,
                "Equal Weight": eq_weights,
                "SPY": spy_weights}
colors       = {"Optimized": "#E63946",
                "Equal Weight": "#FF9800",
                "SPY": "#2196F3"}

# Portfolio daily returns & cumulative values
port_returns = {n: log_returns.dot(w) for n, w in all_weights.items()}
port_values  = {n: (1 + r).cumprod() * INITIAL_VALUE
                for n, r in port_returns.items()}

# ════════════════════════════════════════════════════════════
# METRICS HELPER
# ════════════════════════════════════════════════════════════
def get_metrics(daily_ret):
    ann_ret  = daily_ret.mean() * TRADING_DAYS
    ann_vol  = daily_ret.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - RISK_FREE_RATE) / ann_vol
    down_vol = daily_ret[daily_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino  = (ann_ret - RISK_FREE_RATE) / down_vol
    cum      = (1 + daily_ret).cumprod()
    mdd      = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar   = ann_ret / abs(mdd)
    var95    = np.percentile(daily_ret, 5)
    cvar95   = daily_ret[daily_ret <= var95].mean()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, mdd=mdd, calmar=calmar,
                var95=var95, cvar95=cvar95)

metrics = {n: get_metrics(r) for n, r in port_returns.items()}

# ════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER DATA
# ════════════════════════════════════════════════════════════
def port_perf(w):
    r   = np.sum(log_returns.mean() * w) * TRADING_DAYS
    cov = log_returns.cov() * TRADING_DAYS
    v   = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return r, v, (r - RISK_FREE_RATE) / v

np.random.seed(42)
n_sim = 5000
sim_r, sim_v, sim_s = [], [], []
for _ in range(n_sim):
    w = np.random.random(5); w /= w.sum()
    r, v, s = port_perf(w)
    sim_r.append(r); sim_v.append(v); sim_s.append(s)

# ════════════════════════════════════════════════════════════
# MONTE CARLO DATA
# ════════════════════════════════════════════════════════════
mu, sigma  = port_returns["Optimized"].mean(), port_returns["Optimized"].std()
n_paths, n_days = 500, 252
mc_paths   = np.zeros((n_days, n_paths))
mc_paths[0] = INITIAL_VALUE
for s in range(n_paths):
    p = INITIAL_VALUE
    for d in range(1, n_days):
        p = p * np.exp((mu - 0.5*sigma**2) + sigma*np.random.standard_normal())
        mc_paths[d, s] = p

# ════════════════════════════════════════════════════════════
# MASTER DASHBOARD — 6 PANELS
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0F1117')

fig.suptitle("Quantitative Portfolio Optimizer — Master Dashboard",
             fontsize=18, fontweight='bold', color='white', y=0.98)

gs  = gridspec.GridSpec(3, 3, figure=fig,
                         hspace=0.45, wspace=0.35,
                         top=0.93, bottom=0.05,
                         left=0.06, right=0.97)

panel_style = dict(facecolor='#1A1D27')
text_color  = 'white'
grid_kw     = dict(alpha=0.15, color='white')

# ── Panel 1: Efficient Frontier ─────────────────────────────
ax1 = fig.add_subplot(gs[0, 0], **panel_style)
sc  = ax1.scatter(sim_v, sim_r, c=sim_s, cmap='viridis',
                  alpha=0.4, s=6)
opt_r, opt_v, opt_s = port_perf(opt_weights)
ax1.scatter(opt_v, opt_r, marker='*', color='red',
            s=300, zorder=5, label=f'Max Sharpe')
ax1.set_title("Efficient Frontier", color=text_color, fontsize=11)
ax1.set_xlabel("Volatility", color=text_color, fontsize=9)
ax1.set_ylabel("Return",    color=text_color, fontsize=9)
ax1.tick_params(colors=text_color, labelsize=8)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax1.legend(fontsize=8, facecolor='#1A1D27', labelcolor=text_color)
ax1.grid(**grid_kw)
for sp in ax1.spines.values(): sp.set_color('#333')

# ── Panel 2: Backtest growth curves ────────────────────────
ax2 = fig.add_subplot(gs[0, 1:], **panel_style)
for name, vals in port_values.items():
    ax2.plot(vals.index, vals, color=colors[name],
             linewidth=2, label=name)
ax2.axhline(INITIAL_VALUE, color='white', linewidth=0.8,
            linestyle='--', alpha=0.4)
ax2.set_title("Portfolio Growth — $10,000 Initial (2019–2024)",
              color=text_color, fontsize=11)
ax2.set_ylabel("Portfolio Value ($)", color=text_color, fontsize=9)
ax2.tick_params(colors=text_color, labelsize=8)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax2.legend(fontsize=9, facecolor='#1A1D27', labelcolor=text_color)
ax2.grid(**grid_kw)
for sp in ax2.spines.values(): sp.set_color('#333')

# ── Panel 3: Correlation heatmap ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0], **panel_style)
corr = log_returns.corr()
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0, vmax=1, square=True, linewidths=0.3,
            ax=ax3, cbar=False, annot_kws={'size': 8},
            mask=mask)
ax3.set_title("Correlation Matrix", color=text_color, fontsize=11)
ax3.tick_params(colors=text_color, labelsize=8)

# ── Panel 4: Drawdown ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1], **panel_style)
for name, ret in port_returns.items():
    cum  = (1 + ret).cumprod()
    dd   = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(dd.index, dd, 0,
                     alpha=0.35, color=colors[name])
    ax4.plot(dd.index, dd, color=colors[name],
             linewidth=0.8, label=name)
ax4.set_title("Drawdown Over Time", color=text_color, fontsize=11)
ax4.set_ylabel("Drawdown",          color=text_color, fontsize=9)
ax4.tick_params(colors=text_color, labelsize=8)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax4.legend(fontsize=8, facecolor='#1A1D27', labelcolor=text_color)
ax4.grid(**grid_kw)
for sp in ax4.spines.values(): sp.set_color('#333')

# ── Panel 5: Monte Carlo ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2], **panel_style)
days_ax = np.arange(n_days)
for s in range(n_paths):
    ax5.plot(days_ax, mc_paths[:, s],
             color='steelblue', alpha=0.03, linewidth=0.4)
p5  = np.percentile(mc_paths, 5,  axis=1)
p50 = np.percentile(mc_paths, 50, axis=1)
p95 = np.percentile(mc_paths, 95, axis=1)
ax5.fill_between(days_ax, p5, p95,
                 alpha=0.2, color='steelblue')
ax5.plot(days_ax, p50, color='#E63946',
         linewidth=2, label='Median')
ax5.axhline(INITIAL_VALUE, color='white',
            linewidth=1, linestyle='--', alpha=0.5)
ax5.set_title("Monte Carlo (500 paths, 1yr)",
              color=text_color, fontsize=11)
ax5.set_xlabel("Trading Days", color=text_color, fontsize=9)
ax5.set_ylabel("Value ($)",    color=text_color, fontsize=9)
ax5.tick_params(colors=text_color, labelsize=8)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax5.legend(fontsize=8, facecolor='#1A1D27', labelcolor=text_color)
ax5.grid(**grid_kw)
for sp in ax5.spines.values(): sp.set_color('#333')

# ── Panel 6: Metrics summary table ────────────────────────
ax6 = fig.add_subplot(gs[2, :], **panel_style)
ax6.axis('off')

rows  = ["Annual Return", "Annual Volatility", "Sharpe Ratio",
         "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
         "95% VaR (daily)", "95% CVaR (daily)"]
keys  = ["ann_ret","ann_vol","sharpe","sortino",
         "mdd","calmar","var95","cvar95"]
pct   = {"ann_ret","ann_vol","mdd","var95","cvar95"}

table_data = []
for row, key in zip(rows, keys):
    r = [row]
    for strat in ["Optimized", "Equal Weight", "SPY"]:
        v = metrics[strat][key]
        r.append(f"{v:.2%}" if key in pct else f"{v:.3f}")
    table_data.append(r)

col_labels = ["Metric", "Optimized\n(77% AAPL / 23% MSFT)",
              "Equal Weight\n(20% each)", "SPY\n(Benchmark)"]
tbl = ax6.table(cellText=table_data,
                colLabels=col_labels,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.7)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#333333')
    if r == 0:
        cell.set_facecolor('#0C447C')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cell.set_facecolor('#1E2235')
        cell.set_text_props(color='#AAAAAA')
    elif c == 1:
        cell.set_facecolor('#2A1A1F')
        cell.set_text_props(color='#E63946', fontweight='bold')
    elif c == 2:
        cell.set_facecolor('#1A1D27')
        cell.set_text_props(color='#FF9800')
    else:
        cell.set_facecolor('#1A1D27')
        cell.set_text_props(color='#2196F3')

ax6.set_title("Complete Performance Metrics Summary",
              color=text_color, fontsize=11, pad=10)

plt.savefig('master_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("Master dashboard saved as master_dashboard.png")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY REPORT
# ════════════════════════════════════════════════════════════
final_vals = {n: v.iloc[-1] for n, v in port_values.items()}

print("\n" + "█"*60)
print("  QUANTITATIVE PORTFOLIO OPTIMIZER — FINAL REPORT")
print("█"*60)
print(f"\n  Strategy:    77.3% AAPL + 22.7% MSFT")
print(f"  Period:      January 2019 — January 2024  (5 years)")
print(f"  Universe:    AAPL, AMZN, GOOGL, MSFT, SPY")
print(f"  Method:      Markowitz Mean-Variance Optimization\n")

print(f"  {'─'*54}")
print(f"  {'RETURN METRICS'}")
print(f"  {'─'*54}")
m = metrics["Optimized"]
print(f"  Annual Return:          {m['ann_ret']:>10.2%}")
print(f"  Annual Volatility:      {m['ann_vol']:>10.2%}")
print(f"  $10k grew to:           ${final_vals['Optimized']:>10,.0f}")
print(f"  vs SPY ($10k grew to):  ${final_vals['SPY']:>10,.0f}")

print(f"\n  {'─'*54}")
print(f"  {'RISK-ADJUSTED METRICS'}")
print(f"  {'─'*54}")
print(f"  Sharpe Ratio:           {m['sharpe']:>10.3f}  (>1.0 = excellent)")
print(f"  Sortino Ratio:          {m['sortino']:>10.3f}  (>2.0 = excellent)")
print(f"  Calmar Ratio:           {m['calmar']:>10.3f}")

print(f"\n  {'─'*54}")
print(f"  {'DOWNSIDE RISK METRICS'}")
print(f"  {'─'*54}")
print(f"  Max Drawdown:           {m['mdd']:>10.2%}")
print(f"  95% VaR  (daily):       {m['var95']:>10.2%}")
print(f"  95% CVaR (daily):       {m['cvar95']:>10.2%}")

print(f"\n  {'─'*54}")
print(f"  {'MONTE CARLO (1-year forward)'}")
print(f"  {'─'*54}")
final_mc = mc_paths[-1, :]
print(f"  Probability of profit:  {(final_mc > INITIAL_VALUE).mean():>10.1%}")
print(f"  Median expected value:  ${np.median(final_mc):>10,.0f}")
print(f"  Worst case  (5th %ile): ${np.percentile(final_mc,5):>10,.0f}")
print(f"  Best case  (95th %ile): ${np.percentile(final_mc,95):>10,.0f}")

print(f"\n{'█'*60}")
print(f"  PROJECT COMPLETE — Built with Python, NumPy, Pandas,")
print(f"  Matplotlib, Seaborn, SciPy, yfinance")
print(f"{'█'*60}\n")