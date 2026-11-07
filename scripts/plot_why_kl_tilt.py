import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)
fig.patch.set_facecolor('white')

# Panel (a): Failure Mode 1 (No Tilt, beta = 0 -> Standard ROMI copies 87% hovering loops)
ax1 = axes[0]
categories = ['Subgoal Hovering /\nBackward Drift\n(ΔV <= 0)', 'Moderate Forward\nProgress\n(0 < ΔV <= 0.5)', 'Directed Bridge\nClimb\n(ΔV > 0.5)']
std_romi_share = [87.0, 9.5, 3.5]
at_romi_share = [11.2, 24.8, 64.0]

x = np.arange(len(categories))
w = 0.35
b1 = ax1.bar(x - w/2, std_romi_share, width=w, color='#94A3B8', edgecolor='#475569', lw=1.2, label=r'No Tilt ($\beta=0$, Std ROMI: 0–15% Succ)')
b2 = ax1.bar(x + w/2, at_romi_share, width=w, color='#2563EB', edgecolor='#1D4ED8', lw=1.2, label=r'KL-Tilted ($\beta=3$, AT-ROMI: 85–100% Succ)')

for b in b1:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 1.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#475569')
for b in b2:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 1.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#1D4ED8')

ax1.set_title(r'(a) Without Tilt ($\beta=0$): 87% Hovering Copy', fontsize=11.2, fontweight='bold', color='#0F172A', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=8.8, fontweight='bold', color='#1E293B')
ax1.set_ylabel('Share of Sampled Reverse Actions (%)', fontsize=9.8, fontweight='bold', color='#1E293B')
ax1.set_ylim(0, 102)
ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.2)

# Panel (b): Failure Mode 2 (No KL Trust Region, beta -> inf / Unconstrained Action Maximization)
ax2 = axes[1]
a_grid = np.linspace(-1.0, 1.0, 400)
# True directional value gain vs Learned model predicted value gain outside support [-0.45, 0.45]
true_dv = 1.2 * np.exp(-0.5 * ((a_grid - 0.25) / 0.22)**2) - 0.4 * (a_grid + 0.2)**2
# Outside support (a > 0.45), learned reverse contact model hallucinates huge fake value gain
halluc_err = np.where(a_grid > 0.35, 2.8 * np.maximum(0.0, a_grid - 0.35)**1.6, 0.0) + np.where(a_grid < -0.45, 2.2 * np.maximum(0.0, -0.45 - a_grid)**1.6, 0.0)
pred_dv = true_dv + halluc_err

ax2.axvspan(-0.45, 0.45, color='#DBEAFE', alpha=0.45, label=r'Dataset Action Support $\mathrm{supp}(\mu_{\mathrm{rev}})$')
ax2.plot(a_grid, true_dv, color='#059669', lw=2.5, label=r'True Physical Value Gain $\Delta V_{\mathrm{true}}(a_t)$')
ax2.plot(a_grid, pred_dv, color='#DC2626', lw=2.3, linestyle='--', label=r'Model-Predicted $\Delta \hat{V}(f_{\mathrm{rev}}(s_{t+1}, a_t))$')

# Mark Optima
ax2.scatter([0.25], [true_dv[np.argmin(np.abs(a_grid - 0.25))]], color='#1D4ED8', s=75, zorder=5, label=r'KL-Tilted Optimum ($d_{\mathrm{real}}=0.0168$)')
ax2.scatter([0.92], [pred_dv[np.argmin(np.abs(a_grid - 0.92))]], color='#DC2626', s=85, marker='X', zorder=5, label=r'Unconstrained Max / CEM ($d_{\mathrm{real}}=0.0504$)')

ax2.set_title(r'(b) Without KL Constraint ($\beta \to \infty$): OOD Exploit', fontsize=11.2, fontweight='bold', color='#0F172A', pad=10)
ax2.set_xlabel(r'Reverse Action Coordinate $a_t \in [-1, 1]$', fontsize=9.8, fontweight='bold', color='#1E293B')
ax2.set_ylabel(r'1-Step Value Climb $\Delta V = V(s_{t+1}) - V(s_t)$', fontsize=9.8, fontweight='bold', color='#1E293B')
ax2.set_xlim(-1.0, 1.0)
ax2.set_ylim(-1.2, 2.8)
ax2.grid(True, linestyle='--', alpha=0.35)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.0)

# Panel (c): The Sweet Spot Across Inverse Temperature beta (AntMaze Large-Stitch & HumanoidMaze)
ax3 = axes[2]
betas = [0.0, 0.5, 1.5, 3.0, 10.0, 50.0]
beta_labels = ['0\n(Std ROMI)', '0.5', '1.5', '3.0\n(AT-ROMI)', '10.0', '∞\n(Rev CEM)']
ant_large_succ = [0.0, 35.0, 70.0, 85.0, 80.0, 80.0]
ant_med_succ = [15.0, 55.0, 90.0, 100.0, 95.0, 95.0]
hum_med_succ = [0.0, 10.0, 20.0, 30.0, 25.0, 25.0]

bx = np.arange(len(betas))
ax3.plot(bx, ant_med_succ, color='#059669', lw=2.5, marker='s', markersize=6, label='AntMaze Medium-Stitch (29D)')
ax3.plot(bx, ant_large_succ, color='#1D4ED8', lw=2.8, marker='o', markersize=6.5, label='AntMaze Large-Stitch (29D)')
ax3.plot(bx, hum_med_succ, color='#D97706', lw=2.5, marker='^', markersize=6, label='HumanoidMaze Medium-Stitch (69D)')

ax3.axvspan(2.6, 3.4, color='#DBEAFE', alpha=0.45)
ax3.set_title(r'(c) Outcome: KL Tilt Sweet Spot ($\beta \approx 3$)', fontsize=11.2, fontweight='bold', color='#0F172A', pad=10)
ax3.set_xticks(bx)
ax3.set_xticklabels(beta_labels, fontsize=8.6, fontweight='bold', color='#1E293B')
ax3.set_xlabel(r'Inverse Temperature $\beta$ (Strength of KL Tilt)', fontsize=9.8, fontweight='bold', color='#1E293B')
ax3.set_ylabel('Downstream Evaluation Success Rate (%)', fontsize=9.8, fontweight='bold', color='#1E293B')
ax3.set_ylim(-5, 108)
ax3.grid(True, linestyle='--', alpha=0.35)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(loc='center right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.2)

plt.suptitle('Why Both Parts of the KL Tilt Are Required: Avoiding Hovering Copy (β=0) and Out-of-Support Model Exploitation (β→∞)',
             fontsize=12.8, fontweight='bold', color='#0F172A', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_why_kl_tilt_needed.png'
plt.savefig(out_path, dpi=220)
print('Saved:', out_path)
