import json
import os
import matplotlib.pyplot as plt
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
    'legend.fontsize': 9.6,
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
    with open(os.path.join(DATA_DIR, f'{k}.json'), 'r') as f:
        results[k] = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 5.7), gridspec_kw={'width_ratios': [1.3, 1.0]})
fig.subplots_adjust(wspace=0.22, top=0.80, bottom=0.16, left=0.055, right=0.985)

# Panel (a): Comparing Baselines vs. 3-Part Heuristic Pipeline vs. Clean Single-Principle Ideas
arms = [
    ('model_free_h0', 'Model-Free Baseline (H = 0)', '#64748b'),
    ('standard_romi_h15', 'Standard ROMI (Uniform Rev BC)', '#f59e0b'),
    ('principled_romi_h15', '3-Part Target-Anchored ROMI (CEM)', '#94a3b8'),
    ('at_romi_h15', 'Idea 1: Advantage-Tilted ROMI (No CEM)', '#2563eb'),
    ('real_to_real_model_stitch', 'Idea 2: 1-Step Real-to-Real Stitch (No Rollout)', '#16a34a'),
]

x = np.arange(len(ENV_KEYS))
width = 0.16
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

for env_i in range(len(ENV_KEYS)):
    prev_v = -999.0
    stagger = 0
    for idx, (arm_key, bars, vals) in enumerate(bar_artists_a):
        b = bars[env_i]
        v = vals[env_i]
        if v > 0 and abs(v - prev_v) < 4.5:
            stagger = (stagger + 1) % 2
        else:
            stagger = 0
        prev_v = v
        y_pos = v + 1.4 + (3.6 * stagger if v > 0 else 0.0)
        label_str = f'{v:.0f}%' if v > 0 else '0%'
        is_clean = arm_key in ('at_romi_h15', 'real_to_real_model_stitch')
        font_w = 'bold' if is_clean else 'normal'
        text_c = '#15803d' if arm_key == 'real_to_real_model_stitch' else ('#1d4ed8' if arm_key == 'at_romi_h15' else '#334155')
        fs = 8.2 if is_clean else 7.3
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

ax1.set_title('(a) Task Success: Clean Ideas vs. 3-Part CEM Pipeline (%)', pad=12)
ax1.set_ylabel('Evaluation Success Rate (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(ENV_LABELS)
ax1.set_ylim(0, 117)
ax1.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax1.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.24),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.5,
)

# Panel (b): Upward In-Support Stitching Rate (%) for Standard ROMI vs. Idea 1 (AT-ROMI) vs. Idea 2 (Real-to-Real)
diag_arms = [
    ('unguided_h15', 'Forward Unguided Rollouts', '#ef4444'),
    ('standard_romi_h15', 'Standard ROMI (Uniform BC)', '#f59e0b'),
    ('at_romi_h15', 'Idea 1: Advantage-Tilted ROMI', '#2563eb'),
    ('real_to_real_model_stitch', 'Idea 2: 1-Step Real-to-Real Stitch', '#16a34a'),
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
        if v > 1.0 and abs(v - prev_v) < 8.0:
            stagger = (stagger + 1) % 2
        else:
            stagger = 0
        prev_v = v
        y_pos = v + 1.3 + (3.6 * stagger if v > 1.0 else 0.0)
        is_clean = d_key in ('at_romi_h15', 'real_to_real_model_stitch')
        font_w = 'bold' if is_clean else 'normal'
        text_c = '#15803d' if d_key == 'real_to_real_model_stitch' else ('#1d4ed8' if d_key == 'at_romi_h15' else '#334155')
        fs = 8.2 if is_clean else 7.4
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

ax2.set_title('(b) Upward In-Support Cross-Episode Stitching (%)', pad=12)
ax2.set_ylabel('Upward In-Support Stitch Rate (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(ENV_LABELS)
ax2.set_ylim(0, 102)
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.24),
    ncol=2,
    frameon=False,
    columnspacing=1.1,
    handletextpad=0.5,
)

out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_single_principle_results.png'
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
print('Saved clean figure to:', out_path)
