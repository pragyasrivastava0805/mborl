import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.5,
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

DATA_DIR = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/maze_stitch_results'
ENV_KEYS = ['pointmaze-medium', 'pointmaze-large', 'antmaze-medium', 'antmaze-large', 'humanoidmaze-medium']
ENV_LABELS = ['PointMaze\nMedium', 'PointMaze\nLarge', 'AntMaze\nMedium', 'AntMaze\nLarge', 'HumanoidMaze\nMedium']

results = {}
for k in ENV_KEYS:
    with open(os.path.join(DATA_DIR, f'{k}.json'), 'r') as f:
        results[k] = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.8, 5.4), gridspec_kw={'width_ratios': [1.15, 1.0]})
fig.subplots_adjust(wspace=0.22, top=0.82, bottom=0.15, left=0.04, right=0.98)

# Panel (a): Clean Block Flowchart of Standard ROMI vs. Target-Anchored ROMI Modifications
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('(a) Standard ROMI vs. Our 3 Modifications', pad=12)

def draw_box(ax, x, y, w, h, title, subtitle, bg, border, title_c='#0f172a'):
    box = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.18,rounding_size=0.25',
        facecolor=bg, edgecolor=border, linewidth=1.8
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center', fontsize=10.2, fontweight='bold', color=title_c)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha='center', va='center', fontsize=8.8, color='#334155')

# Row 1: Standard ROMI (Top)
ax1.text(0.2, 8.85, 'Standard ROMI (Wang et al., 2021)', fontsize=10.5, fontweight='bold', color='#b45309')
draw_box(ax1, 0.2, 6.2, 2.8, 2.1, '1. Single Anchor s_H', 'Unpaired state in D_B', '#fef3c7', '#f59e0b', '#92400e')
draw_box(ax1, 3.6, 6.2, 2.8, 2.1, '2. Reverse BC pi_rev', 'Copies 87% subgoal hover', '#fef3c7', '#f59e0b', '#92400e')
draw_box(ax1, 7.0, 6.2, 2.8, 2.1, '3. Unpinned Start s_0', '0% to 15% Task Success', '#fee2e2', '#ef4444', '#991b1b')

ax1.annotate('', xy=(3.55, 7.25), xytext=(3.05, 7.25), arrowprops=dict(arrowstyle='->', lw=2.0, color='#d97706'))
ax1.annotate('', xy=(6.95, 7.25), xytext=(6.45, 7.25), arrowprops=dict(arrowstyle='->', lw=2.0, color='#d97706'))

# Row 2: Our Target-Anchored ROMI (Bottom)
ax1.text(0.2, 4.35, 'Our Target-Anchored ROMI Bridge (H = 15)', fontsize=10.5, fontweight='bold', color='#15803d')
draw_box(ax1, 0.2, 1.7, 2.8, 2.1, 'Mod 1: Pair (s_0, s^+)', '33%/67% Manifold Check', '#dcfce7', '#16a34a', '#166534')
draw_box(ax1, 3.6, 1.7, 2.8, 2.1, 'Mod 2: Reverse CEM', 'Steer p_psi to Source s_0', '#dcfce7', '#16a34a', '#166534')
draw_box(ax1, 7.0, 1.7, 2.8, 2.1, 'Mod 3: Dual-Pin Gate', '80% to 100% (Point/Ant)', '#dcfce7', '#16a34a', '#166534')

ax1.annotate('', xy=(3.55, 2.75), xytext=(3.05, 2.75), arrowprops=dict(arrowstyle='->', lw=2.0, color='#16a34a'))
ax1.annotate('', xy=(6.95, 2.75), xytext=(6.45, 2.75), arrowprops=dict(arrowstyle='->', lw=2.0, color='#16a34a'))

# Vertical modification arrows
for cx in (1.6, 5.0, 8.4):
    ax1.annotate('', xy=(cx, 3.95), xytext=(cx, 6.05), arrowprops=dict(arrowstyle='->', lw=1.8, ls='--', color='#475569'))

# Panel (b): Direct Comparison of Standard ROMI vs. Target-Anchored ROMI Bridge Success Rate (%)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

x = np.arange(len(ENV_KEYS))
w = 0.34
std_vals = [results[k]['arm_results']['standard_romi_h15']['success_rate_pct'] for k in ENV_KEYS]
mod_vals = [results[k]['arm_results']['principled_romi_h15']['success_rate_pct'] for k in ENV_KEYS]

b1 = ax2.bar(x - w/2, std_vals, width=w * 0.92, label='Standard ROMI (Reverse BC, H = 15)', color='#f59e0b', edgecolor='white', zorder=3)
b2 = ax2.bar(x + w/2, mod_vals, width=w * 0.92, label='Target-Anchored ROMI (Mods 1-3, H = 15)', color='#16a34a', edgecolor='white', zorder=3)

for b, v in zip(b1, std_vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 1.8, f'{v:.0f}%', ha='center', va='bottom', fontsize=8.5, color='#92400e')
for b, v in zip(b2, mod_vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 1.8, f'{v:.0f}%', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#15803d')

ax2.set_title('(b) Task Success Gain From Modifications (%)', pad=12)
ax2.set_ylabel('Evaluation Success Rate (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(ENV_LABELS)
ax2.set_ylim(0, 115)
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=False)

out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_romi_modifications.png'
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
print('Saved clean figure to:', out_path)
