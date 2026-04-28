import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Load data ───────────────────────────────────────────────
prices      = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
log_returns = np.log(prices / prices.shift(1)).dropna()

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05
INITIAL_VALUE  = 10000

# ── Optimized portfolio daily returns ───────────────────────
opt_weights  = np.array([0.773, 0.000, 0.000, 0.227, 0.000])
opt_returns  = log_returns.dot(opt_weights)

# ── Key statistics from historical data ─────────────────────
mu      = opt_returns.mean()           # daily mean return
sigma   = opt_returns.std()            # daily volatility
ann_mu  = mu    * TRADING_DAYS
ann_sig = sigma * np.sqrt(TRADING_DAYS)

print("="*50)
print("PORTFOLIO RISK PARAMETERS")
print("="*50)
print(f"Daily mean return:       {mu:.6f}  ({mu*100:.4f}%)")
print(f"Daily volatility:        {sigma:.6f}  ({sigma*100:.4f}%)")
print(f"Annualized return:       {ann_mu:.2%}")
print(f"Annualized volatility:   {ann_sig:.2%}")

# ════════════════════════════════════════════════════════════
# PART 1 — VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR)
# ════════════════════════════════════════════════════════════

confidence_levels = [0.90, 0.95, 0.99]

print("\n" + "="*50)
print("VALUE AT RISK REPORT")
print("="*50)
print(f"\n{'Confidence':<14} {'Daily VaR':>12} {'Daily CVaR':>12} "
      f"{'$ Loss (10k)':>14}")
print("-"*55)

var_results = {}
for cl in confidence_levels:
    # Historical VaR — find the actual percentile from real data
    hist_var  = np.percentile(opt_returns, (1 - cl) * 100)

    # CVaR (Expected Shortfall) — average of all returns BELOW VaR
    cvar      = opt_returns[opt_returns <= hist_var].mean()

    # Dollar loss on $10,000 portfolio
    dollar_loss = abs(hist_var) * INITIAL_VALUE

    var_results[cl] = {'VaR': hist_var, 'CVaR': cvar}

    print(f"{cl:.0%}{'':8} {hist_var:>11.2%} {cvar:>11.2%} "
          f"${dollar_loss:>12,.0f}")

# Focus on 95% VaR for detailed explanation
var_95  = var_results[0.95]['VaR']
cvar_95 = var_results[0.95]['CVaR']

print(f"\n95% VaR Interpretation:")
print(f"  On 95% of trading days, the portfolio loses less than "
      f"{abs(var_95):.2%}")
print(f"  That's less than ${abs(var_95)*INITIAL_VALUE:,.0f} on a "
      f"$10,000 portfolio per day")
print(f"\n95% CVaR Interpretation:")
print(f"  On the worst 5% of days, the average loss is "
      f"{abs(cvar_95):.2%}")
print(f"  That's ${abs(cvar_95)*INITIAL_VALUE:,.0f} on a "
      f"$10,000 portfolio on a bad day")

# ════════════════════════════════════════════════════════════
# PART 2 — MONTE CARLO SIMULATION
# ════════════════════════════════════════════════════════════

n_simulations  = 1000
n_days         = 252          # 1 year forward

print(f"\n{'='*50}")
print(f"MONTE CARLO SIMULATION")
print(f"Running {n_simulations} simulations × {n_days} days...")
print(f"{'='*50}")

np.random.seed(42)

# Store all simulation paths
simulation_paths = np.zeros((n_days, n_simulations))
simulation_paths[0] = INITIAL_VALUE

# Geometric Brownian Motion formula:
# S(t) = S(t-1) × exp((μ - σ²/2) × dt + σ × √dt × Z)
# where Z ~ N(0,1) is a random shock each day
dt = 1  # 1 trading day

for sim in range(n_simulations):
    price = INITIAL_VALUE
    for day in range(1, n_days):
        Z     = np.random.standard_normal()     # random market shock
        drift = (mu - 0.5 * sigma**2) * dt      # expected drift
        shock = sigma * np.sqrt(dt) * Z          # random component
        price = price * np.exp(drift + shock)
        simulation_paths[day, sim] = price

# Final values across all simulations
final_values = simulation_paths[-1, :]

# ── Scenario analysis ───────────────────────────────────────
pct_5    = np.percentile(final_values, 5)
pct_25   = np.percentile(final_values, 25)
median   = np.percentile(final_values, 50)
pct_75   = np.percentile(final_values, 75)
pct_95   = np.percentile(final_values, 95)
prob_loss = (final_values < INITIAL_VALUE).mean()
mean_val  = final_values.mean()

print(f"\nScenario Analysis (starting from ${INITIAL_VALUE:,}):")
print(f"{'':4}{'Scenario':<28} {'Final Value':>12} {'Return':>10}")
print(f"  {'-'*52}")
print(f"  {'Worst case  (5th percentile)':<28} ${pct_5:>10,.0f}"
      f"  {(pct_5/INITIAL_VALUE-1):>9.1%}")
print(f"  {'Bad case    (25th percentile)':<28} ${pct_25:>10,.0f}"
      f"  {(pct_25/INITIAL_VALUE-1):>9.1%}")
print(f"  {'Median case (50th percentile)':<28} ${median:>10,.0f}"
      f"  {(median/INITIAL_VALUE-1):>9.1%}")
print(f"  {'Good case   (75th percentile)':<28} ${pct_75:>10,.0f}"
      f"  {(pct_75/INITIAL_VALUE-1):>9.1%}")
print(f"  {'Best case   (95th percentile)':<28} ${pct_95:>10,.0f}"
      f"  {(pct_95/INITIAL_VALUE-1):>9.1%}")
print(f"\n  Mean expected value:   ${mean_val:>10,.0f}")
print(f"  Probability of loss:    {prob_loss:.1%}")
print(f"  Probability of profit:  {1-prob_loss:.1%}")

# ════════════════════════════════════════════════════════════
# PART 3 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 11))
fig.suptitle("Risk Analysis & Monte Carlo Simulation",
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel 1: Monte Carlo fan chart ─────────────────────────
ax1 = fig.add_subplot(gs[0, :])    # spans full width

days_axis = np.arange(n_days)

# Plot all paths in light grey
for sim in range(n_simulations):
    ax1.plot(days_axis, simulation_paths[:, sim],
             color='steelblue', alpha=0.03, linewidth=0.5)

# Plot confidence interval bands
p5  = np.percentile(simulation_paths, 5,  axis=1)
p25 = np.percentile(simulation_paths, 25, axis=1)
p50 = np.percentile(simulation_paths, 50, axis=1)
p75 = np.percentile(simulation_paths, 75, axis=1)
p95 = np.percentile(simulation_paths, 95, axis=1)

ax1.fill_between(days_axis, p5,  p95, alpha=0.15,
                 color='steelblue', label='5th–95th percentile')
ax1.fill_between(days_axis, p25, p75, alpha=0.25,
                 color='steelblue', label='25th–75th percentile')
ax1.plot(days_axis, p50, color='#E63946',
         linewidth=2.5, label='Median path', zorder=5)
ax1.axhline(INITIAL_VALUE, color='black',
            linewidth=1.5, linestyle='--',
            alpha=0.6, label=f'Initial ${INITIAL_VALUE:,}')

ax1.set_title(f"Monte Carlo Simulation — {n_simulations} Paths "
              f"over {n_days} Trading Days (1 Year)",
              fontsize=13)
ax1.set_xlabel("Trading Days Forward")
ax1.set_ylabel("Portfolio Value ($)")
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(alpha=0.3)

# ── Panel 2: Final value distribution ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])

ax2.hist(final_values, bins=60, color='steelblue',
         edgecolor='white', alpha=0.8, density=True)

# Mark key percentiles
for val, label, color in [
    (pct_5,  '5th %ile',  '#E63946'),
    (median, 'Median',    '#1D9E75'),
    (pct_95, '95th %ile', '#FF9800'),
]:
    ax2.axvline(val, color=color, linewidth=2,
                linestyle='--', label=f'{label}: ${val:,.0f}')

ax2.axvline(INITIAL_VALUE, color='black',
            linewidth=2, linestyle='-',
            label=f'Initial: ${INITIAL_VALUE:,}')

ax2.set_title("Distribution of Final Portfolio Values", fontsize=12)
ax2.set_xlabel("Portfolio Value After 1 Year ($)")
ax2.set_ylabel("Density")
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Panel 3: VaR visualisation ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

ax3.hist(opt_returns, bins=80, color='#534AB7',
         edgecolor='white', alpha=0.8, density=True,
         label='Daily returns')

# Shade the VaR tail
tail_vals = opt_returns[opt_returns <= var_95]
ax3.hist(tail_vals, bins=40, color='#E63946',
         edgecolor='white', alpha=0.9, density=True,
         label=f'Worst 5% of days')

ax3.axvline(var_95,  color='#E63946', linewidth=2.5,
            linestyle='--',
            label=f'95% VaR: {var_95:.2%}')
ax3.axvline(cvar_95, color='#BA7517', linewidth=2.5,
            linestyle=':',
            label=f'95% CVaR: {cvar_95:.2%}')

ax3.set_title("Value at Risk — Daily Return Distribution",
              fontsize=12)
ax3.set_xlabel("Daily Return")
ax3.set_ylabel("Density")
ax3.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

plt.savefig('monte_carlo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nMonte Carlo chart saved as monte_carlo.png")