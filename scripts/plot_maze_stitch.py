import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication-grade style (zero callout boxes or prose annotations inside axes)
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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DATA_DIR = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/maze_stitch_results'
ENV_KEYS = [
    'pointmaze-medium',
    'pointmaze-large',
    'antmaze-medium',
    'antmaze-large',
    'humanoidmaze-medium',
]
ENV_LABELS = [
    'PointMaze\nMedium-Stitch',
    'PointMaze\nLarge-Stitch',
    'AntMaze\nMedium-Stitch',
    'AntMaze\nLarge-Stitch',
    'HumanoidMaze\nMedium-Stitch',
]

results = {}
for k in ENV_KEYS:
    p = os.path.join(DATA_DIR, f'{k}.json')
    with open(p, 'r') as f:
        results[k] = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.6, 5.7), gridspec_kw={'width_ratios': [1.32, 1.0]})
fig.subplots_adjust(wspace=0.22, top=0.80, bottom=0.16, left=0.055, right=0.985)

# Panel (a): 6-Arm Downstream Evaluation Success Rate (%)
arms = [
    ('model_free_h0', 'Model-Free Baseline (H = 0)', '#64748b'),
    ('short_dyna_h1', 'Short-Horizon Dyna (H = 1)', '#94a3b8'),
    ('medium_dyna_h5', 'Medium-Horizon Dyna (H = 5)', '#f59e0b'),
    ('td_lambda_mve_h15', 'TD(λ)-MVE Target-Only (H = 15)', '#8b5cf6'),
    ('unfiltered_dyna_h15', 'Unfiltered Long Dyna (H = 15)', '#ef4444'),
    ('principled_bridge_h15', 'Principled Bridge Dyna (H = 15)', '#16a34a'),
]

x = np.arange(len(ENV_KEYS))
width = 0.136
offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2.0) * width

bar_artists_a = []
for idx, (arm_key, arm_label, color) in enumerate(arms):
    vals = [results[k]['arm_results'][arm_key]['success_rate_pct'] for k in ENV_KEYS]
    bars = ax1.bar(
        x + offsets[idx],
        vals,
        width=width * 0.90,
        label=arm_label,
        color=color,
        edgecolor='white',
        linewidth=0.6,
        zorder=3,
    )
    bar_artists_a.append((arm_key, bars, vals))

# Place non-overlapping direct value labels per environment group
for env_i in range(len(ENV_KEYS)):
    prev_v = -999.0
    stagger = 0
    for idx, (arm_key, bars, vals) in enumerate(bar_artists_a):
        b = bars[env_i]
        v = vals[env_i]
        if v > 0 and abs(v - prev_v) < 4.0:
            stagger = (stagger + 1) % 2
        else:
            stagger = 0
        prev_v = v
        y_pos = v + 1.4 + (3.4 * stagger if v > 0 else 0.0)
        label_str = f'{v:.0f}%' if v > 0 else '0%'
        font_w = 'bold' if arm_key == 'principled_bridge_h15' else 'normal'
        text_c = '#15803d' if arm_key == 'principled_bridge_h15' else '#334155'
        fs = 8.2 if arm_key == 'principled_bridge_h15' else 7.2
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            y_pos,
            label_str,
            ha='center',
            va='bottom',
            fontsize=fs,
            fontweight=font_w,
            color=text_c,
        )

ax1.set_title('(a) Downstream Task Success Rate (%)', pad=12)
ax1.set_ylabel('Evaluation Success Rate (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(ENV_LABELS)
ax1.set_ylim(0, 116)
ax1.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax1.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.24),
    ncol=3,
    frameon=False,
    columnspacing=1.1,
    handletextpad=0.5,
)

# Panel (b): Test 2 In-Support Upward Cross-Episode Stitching Rate (%)
diag_arms = [
    ('short_h1', 'Short Rollouts (H = 1)', '#94a3b8'),
    ('medium_h5', 'Medium Rollouts (H = 5)', '#f59e0b'),
    ('unguided_h15', 'Unguided Rollouts (H = 15)', '#ef4444'),
    ('principled_bridge_h15', 'Principled CEM Bridges (H = 15)', '#16a34a'),
]

width2 = 0.192
offsets2 = (np.arange(len(diag_arms)) - (len(diag_arms) - 1) / 2.0) * width2

bar_artists_b = []
for idx, (d_key, d_label, color) in enumerate(diag_arms):
    vals = [results[k]['stitch_diagnostics'][d_key]['upward_stitch_pct'] for k in ENV_KEYS]
    bars = ax2.bar(
        x + offsets2[idx],
        vals,
        width=width2 * 0.90,
        label=d_label,
        color=color,
        edgecolor='white',
        linewidth=0.6,
        zorder=3,
    )
    bar_artists_b.append((d_key, bars, vals))

for env_i in range(len(ENV_KEYS)):
    prev_v = -999.0
    stagger = 0
    for idx, (d_key, bars, vals) in enumerate(bar_artists_b):
        b = bars[env_i]
        v = vals[env_i]
        if v > 1.0 and abs(v - prev_v) < 3.5:
            stagger = (stagger + 1) % 2
        else:
            stagger = 0
        prev_v = v
        y_pos = v + 1.3 + (3.2 * stagger if v > 1.0 else 0.0)
        font_w = 'bold' if d_key == 'principled_bridge_h15' else 'normal'
        text_c = '#15803d' if d_key == 'principled_bridge_h15' else '#334155'
        fs = 8.4 if d_key == 'principled_bridge_h15' else 7.5
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            y_pos,
            f'{v:.1f}%',
            ha='center',
            va='bottom',
            fontsize=fs,
            fontweight=font_w,
            color=text_c,
        )

ax2.set_title('(b) In-Support Upward Cross-Episode Stitching (%)', pad=12)
ax2.set_ylabel('Upward In-Support Stitch Rate (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(ENV_LABELS)
ax2.set_ylim(0, 96)
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.24),
    ncol=2,
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.5,
)

out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_maze_stitch_results.png'
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
print('Saved clean figure to:', out_path)
