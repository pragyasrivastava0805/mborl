import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)
fig.patch.set_facecolor('white')

# Panel (a): Empirical distribution of Delta V = V(s_{t+1}) - V(s_t) and Exponential Tilting
ax = axes[0]
np.random.seed(7)
dv = np.linspace(-2.5, 2.5, 400)
p_emp = 0.82 * np.exp(-0.5 * ((dv + 0.15) / 0.38)**2) + 0.18 * np.exp(-0.5 * ((dv - 0.85) / 0.45)**2)
p_emp = p_emp / np.trapezoid(p_emp, dv)

beta = 3.0
weights = np.clip(np.exp(beta * dv), 0.05, 20.0)
p_tilt = p_emp * weights
p_tilt = p_tilt / np.trapezoid(p_tilt, dv)

ax.fill_between(dv, 0, p_emp, color='#94A3B8', alpha=0.35, label=r'Standard ROMI $\mu_{\mathrm{rev}}(a_t|s_{t+1})$ ($\beta=0$)')
ax.plot(dv, p_emp, color='#475569', lw=2.2, linestyle='--')

ax.fill_between(dv, 0, p_tilt, color='#2563EB', alpha=0.30, label=r'AT-ROMI $\pi^*_{\mathrm{rev}}(a_t|s_{t+1}) \propto \mu_{\mathrm{rev}} e^{\beta \Delta V}$')
ax.plot(dv, p_tilt, color='#1D4ED8', lw=2.6)

ax.axvline(0.0, color='#DC2626', lw=1.4, linestyle=':')
ax.set_title('(a) 1. Cause: Exponential Tilting of Reverse Measure', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_xlabel(r'1-Step Forward Value Gain $\Delta V = V(s_{t+1}) - V(s_t)$', fontsize=10, fontweight='bold', color='#1E293B')
ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold', color='#1E293B')
ax.set_xlim(-2.2, 2.3)
ax.set_ylim(0, 1.35)
ax.grid(True, linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.8)

# Panel (b): Support Containment in 29D/69D Action Space (AT-ROMI vs Reverse CEM)
ax2 = axes[1]
theta = np.linspace(0, 2*np.pi, 200)
r_x = 0.55 * np.cos(theta) - 0.1 * np.sin(2*theta)
r_y = 0.38 * np.sin(theta) + 0.08 * np.cos(2*theta)
ax2.fill(r_x, r_y, color='#E2E8F0', alpha=0.7, edgecolor='#64748B', lw=1.8, linestyle='--', label=r'Behavioral Action Support $\mathrm{supp}(\mu_{\mathrm{rev}})$')

np.random.seed(12)
s_x = np.random.normal(-0.18, 0.14, 45)
s_y = np.random.normal(-0.08, 0.11, 45)
ax2.scatter(s_x, s_y, color='#64748B', s=28, alpha=0.75, label=r'Standard ROMI ($\Delta V \approx 0$ Hovering)')

a_x = np.random.normal(0.26, 0.10, 45)
a_y = np.random.normal(0.12, 0.08, 45)
ax2.scatter(a_x, a_y, color='#1D4ED8', s=36, marker='o', edgecolor='white', lw=0.6, label=r'AT-ROMI ($d_{\mathrm{real}}=0.0168$, In-Support)')

c_x = np.random.normal(0.72, 0.12, 30)
c_y = np.random.normal(0.58, 0.11, 30)
ax2.scatter(c_x, c_y, color='#DC2626', s=42, marker='x', lw=1.8, label=r'Reverse CEM ($d_{\mathrm{real}}=0.0504$, OOD Exploit)')

ax2.set_title('(b) 2. Mechanism: Strict In-Support KL Projection', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
ax2.set_xlabel(r'Action Dimension $a^{(1)}_t \in [-1, 1]$', fontsize=10, fontweight='bold', color='#1E293B')
ax2.set_ylabel(r'Action Dimension $a^{(2)}_t \in [-1, 1]$', fontsize=10, fontweight='bold', color='#1E293B')
ax2.set_xlim(-0.95, 1.05)
ax2.set_ylim(-0.75, 0.95)
ax2.grid(True, linestyle='--', alpha=0.35)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.3)

# Panel (c): Telescoping H-Step Value Bridge V(s_H) - V(s_0) = sum_{k=0}^{H-1} Delta V_k
ax3 = axes[2]
steps = np.arange(0, 16)
v_std = -12.0 - 0.02 * steps + np.array([0, -0.1, 0.05, -0.05, 0.02, -0.08, -0.04, 0.01, -0.06, -0.09, -0.05, -0.12, -0.08, -0.15, -0.18, -0.20])
v_cem = -12.0 - 0.52 * steps
v_at = -12.0 - 0.68 * steps

ax3.axhline(-12.0, color='#059669', lw=1.3, linestyle=':', label=r'Target Segment $\mathcal{D}_B$ Value ($V(s_H)$)')
ax3.axhline(-21.5, color='#D97706', lw=1.3, linestyle=':', label=r'Source Segment $\mathcal{D}_A$ Value ($V(s_{\mathrm{src}})$)')

ax3.plot(steps, v_std, color='#64748B', lw=2.2, marker='s', markersize=4.5, linestyle='--', label=r'Standard ROMI ($\sum \Delta V_k = +0.20$)')
ax3.plot(steps, v_cem, color='#DC2626', lw=2.2, marker='^', markersize=5.0, linestyle='-.', label=r'Reverse CEM Bridge (OOD Drift at $k>8$)')
ax3.plot(steps, v_at, color='#1D4ED8', lw=2.8, marker='o', markersize=5.5, label=r'AT-ROMI Telescoping Sum ($\sum \Delta V_k = +10.2$)')

ax3.set_title(r'(c) 3. Outcome: Telescoping $H$-Step Value Bridge', fontsize=11.5, fontweight='bold', color='#0F172A', pad=10)
ax3.set_xlabel(r'Reverse Rollout Step $H - k$ from Anchor $s_H$', fontsize=10, fontweight='bold', color='#1E293B')
ax3.set_ylabel(r'State Value $V(\hat{s}_k)$ Along Reverse Rollout', fontsize=10, fontweight='bold', color='#1E293B')
ax3.set_xlim(0, 15)
ax3.set_ylim(-24.5, -9.5)
ax3.grid(True, linestyle='--', alpha=0.35)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.2)

plt.suptitle('Mathematical Formulation of Idea 1 (Advantage-Tilted ROMI): KL-Regularized Reverse Bridge Sampling', fontsize=13.5, fontweight='bold', color='#0F172A', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_idea1_math_derivation.png'
plt.savefig(out_path, dpi=220)
print('Saved:', out_path)
