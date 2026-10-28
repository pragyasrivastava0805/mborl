#!/usr/bin/env python3
"""Generate the two Comment 15 rollout fidelity and reward/success plots."""

import json
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA_PATH = Path(
    "/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/rollout_fidelity_initial.json"
)
OUT_DIR = Path(
    "/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch"
)

raw = json.loads(DATA_PATH.read_text())
ckpts = raw["checkpoints"]

depths = np.arange(1, 11)

SERIES = [
    (
        "exp14_e1_h5_100k_s1_1M",
        "Seed 1 at 1M Steps (Stable, 60% Eval Success)",
        "#1b9e77",
        "-",
        "o",
        2.4,
    ),
    (
        "exp14_e1_h5_100k_s2_1M",
        "Seed 2 at 1M Steps (Moderate, 13% Eval Success)",
        "#377eb8",
        "-",
        "s",
        2.2,
    ),
    (
        "s0_ckpt500k_precollapse",
        "Seed 0 at 500k Steps (Pre-Collapse Peak, 53% Eval / 87% Peak)",
        "#d95f02",
        "--",
        "^",
        2.4,
    ),
    (
        "exp14_e1_h5_100k_s0_1M",
        "Seed 0 at 1M Steps (Post-Collapse, 0% Eval Success)",
        "#e41a1c",
        ":",
        "D",
        2.2,
    ),
]


def get_curve(key, metric_prefix):
  rows = ckpts[key]["depths"][1:]  # depths 1..10
  mean = np.array([r[f"{metric_prefix}_mean"] for r in rows])
  se = np.array([r[f"{metric_prefix}_se"] for r in rows])
  return mean, se


# ==============================================================================
# Figure 1: State-Prediction Error vs. Rollout Depth (k = 1..10)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4), dpi=200)
fig.suptitle(
    "Comment 15 Plot 1: Normalized Trajectory State-Prediction Error vs. Rollout Depth (k = 1..10)\n"
    "200 Held-Out Simulator States × 5 Stochastic Ensemble Rollouts per State (Shaded Bands = ±1 SE)",
    y=0.98,
)

PANELS_FIG1 = [
    ("total_l2_sq", "A. Total Normalized State Error (All 40 Dims)", "Squared L2 Error ||φ(ŝ_k) - φ(s_k)||²"),
    ("proprio_l2_sq", "B. Robot Arm & Gripper Proprioception (Dims 0:19)", "Squared L2 Error (19 Dims)"),
    ("cube_l2_sq", "C. Cube Pose & Orientation (Dims 19:28)", "Squared L2 Error (9 Dims)"),
    ("buttons_l2_sq", "D. Button States & Positions (Dims 28:36)", "Squared L2 Error (8 Dims)"),
    ("drawer_l2_sq", "E. Drawer Position & Velocity (Dims 36:38)", "Squared L2 Error (2 Dims)"),
    ("window_l2_sq", "F. Window Position & Velocity (Dims 38:40)", "Squared L2 Error (2 Dims)"),
]

for idx, (metric, title, ylabel) in enumerate(PANELS_FIG1):
  ax = axes[idx // 3, idx % 3]
  for key, label, color, ls, marker, lw in SERIES:
    mean, se = get_curve(key, metric)
    ax.plot(
        depths,
        mean,
        label=label,
        color=color,
        linestyle=ls,
        marker=marker,
        linewidth=lw,
        markersize=5.5,
    )
    ax.fill_between(depths, np.maximum(0, mean - se), mean + se, color=color, alpha=0.15)

  means_1m = np.stack([
      get_curve(k, metric)[0]
      for k in [
          "exp14_e1_h5_100k_s0_1M",
          "exp14_e1_h5_100k_s1_1M",
          "exp14_e1_h5_100k_s2_1M",
      ]
  ])
  mean_3seed = means_1m.mean(axis=0)
  ax.plot(
      depths,
      mean_3seed,
      label="3-Seed Mean at 1M Steps (Seeds 0, 1, 2)",
      color="#222222",
      linestyle="-.",
      linewidth=2.0,
      alpha=0.85,
  )

  ax.axvline(5.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
  ymax = ax.get_ylim()[1]
  ax.text(
      5.15,
      ymax * 0.88 if ymax > 0 else 1.0,
      "H=5 Horizon",
      fontsize=8.5,
      color="#555555",
  )
  ax.set_title(title, pad=8)
  ax.set_xlabel("Prediction Rollout Depth k (Steps)")
  ax.set_ylabel(ylabel)
  ax.set_xticks(depths)
  ax.grid(True, linestyle=":", alpha=0.45)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    facecolor="#f8f9fa",
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.01),
)
plt.tight_layout(rect=[0, 0.075, 1, 0.93])
fig1_path = OUT_DIR / "fig1_comment15_state_error_vs_depth.png"
fig.savefig(fig1_path, dpi=200)
plt.close(fig)

# ==============================================================================
# Figure 2: Reward and Success Correctness vs. Rollout Depth (k = 1..10)
# Shared bottom legend so zero panels are occluded!
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.8), dpi=200)
fig.suptitle(
    "Comment 15 Plot 2: Reward & Goal-Completion Fidelity vs. Rollout Depth (k = 1..10)\n"
    "Solid Curves = Trajectory Evaluation (Includes State Drift)  |  Dotted/Dashed Curves = Same-State Head / Model Prediction",
    y=0.98,
)

# Panel A: Reward MAE (Trajectory vs Same-State Head)
ax = axes[0, 0]
for key, label, color, ls, marker, lw in SERIES:
  traj_m, traj_se = get_curve(key, "traj_rew_mae")
  head_m, head_se = get_curve(key, "valid_rew_mae")
  ax.plot(
      depths,
      traj_m,
      color=color,
      linestyle="-",
      marker=marker,
      linewidth=2.3,
      markersize=5.5,
  )
  ax.fill_between(depths, np.maximum(0, traj_m - traj_se), traj_m + traj_se, color=color, alpha=0.14)
  ax.plot(
      depths,
      head_m,
      color=color,
      linestyle=":",
      linewidth=2.0,
      alpha=0.9,
  )
ax.axvline(5.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(5.15, 0.15, "H=5 Horizon", fontsize=8.5, color="#555555")
ax.set_title("A. Reward Error (MAE): Trajectory vs. Same-State Head", pad=8)
ax.set_xlabel("Prediction Rollout Depth k (Steps)")
ax.set_ylabel("Mean Absolute Reward Error (Scale [-6, 0])")
ax.set_xticks(depths)
ax.grid(True, linestyle=":", alpha=0.45)

# Panel B: Signed Reward Error Bias
ax = axes[0, 1]
ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.6)
for key, label, color, ls, marker, lw in SERIES:
  traj_m, traj_se = get_curve(key, "traj_rew_bias")
  head_m, head_se = get_curve(key, "valid_rew_bias")
  ax.plot(
      depths,
      traj_m,
      color=color,
      linestyle="-",
      marker=marker,
      linewidth=2.3,
      markersize=5.5,
  )
  ax.fill_between(depths, traj_m - traj_se, traj_m + traj_se, color=color, alpha=0.14)
  ax.plot(
      depths,
      head_m,
      color=color,
      linestyle=":",
      linewidth=2.0,
      alpha=0.9,
  )
ax.axvline(5.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(5.15, 0.015, "H=5 Horizon", fontsize=8.5, color="#555555")
ax.set_title("B. Signed Reward Bias: Trajectory vs. Same-State Head", pad=8)
ax.set_xlabel("Prediction Rollout Depth k (Steps)")
ax.set_ylabel("Signed Reward Error (Positive = Optimistic)")
ax.set_xticks(depths)
ax.grid(True, linestyle=":", alpha=0.45)

# Panel C: True Simulator Completion Rate vs Model-Predicted Completion Rate (%)
ax = axes[1, 0]
for key, label, color, ls, marker, lw in SERIES:
  true_m, true_se = get_curve(key, "true_sim_succ")
  pred_m, pred_se = get_curve(key, "traj_pred_succ")
  ax.plot(
      depths,
      true_m * 100.0,
      color=color,
      linestyle="-",
      marker=marker,
      linewidth=2.3,
      markersize=5.5,
  )
  ax.fill_between(
      depths,
      np.maximum(0, (true_m - true_se) * 100.0),
      (true_m + true_se) * 100.0,
      color=color,
      alpha=0.12,
  )
  ax.plot(
      depths,
      pred_m * 100.0,
      color=color,
      linestyle=":",
      linewidth=2.0,
      alpha=0.9,
  )
ax.axvline(5.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(5.15, 1.50, "H=5 Horizon", fontsize=8.5, color="#555555")
ax.set_title("C. Goal Completion (%): True Simulator vs. Model Head", pad=8)
ax.set_xlabel("Prediction Rollout Depth k (Steps)")
ax.set_ylabel("Goal Completion Rate (%)")
ax.set_xticks(depths)
ax.grid(True, linestyle=":", alpha=0.45)

# Panel D: False-Positive Rate (Trajectory vs Same-State Head)
ax = axes[1, 1]
for key, label, color, ls, marker, lw in SERIES:
  fp_traj_m, fp_traj_se = get_curve(key, "traj_fp_succ")
  fp_head_m, _ = get_curve(key, "valid_fp_succ")
  ax.plot(
      depths,
      fp_traj_m * 100.0,
      color=color,
      linestyle="-",
      marker=marker,
      linewidth=2.3,
      markersize=5.5,
  )
  ax.fill_between(
      depths,
      np.maximum(0, (fp_traj_m - fp_traj_se) * 100.0),
      (fp_traj_m + fp_traj_se) * 100.0,
      color=color,
      alpha=0.12,
  )
  ax.plot(
      depths,
      fp_head_m * 100.0,
      color=color,
      linestyle=":",
      linewidth=2.0,
      alpha=0.9,
  )
ax.axvline(5.0, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(5.15, 0.38, "H=5 Horizon", fontsize=8.5, color="#555555")
ax.set_ylim(-0.02, 0.45)
ax.set_title("D. False-Positive Completion (%): Trajectory vs. Same-State Head", pad=8)
ax.set_xlabel("Prediction Rollout Depth k (Steps)")
ax.set_ylabel("Hallucinated Completion Rate (%)")
ax.set_xticks(depths)
ax.grid(True, linestyle=":", alpha=0.45)

legend_handles = [
    mlines.Line2D([], [], color=color, marker=marker, linestyle="-", linewidth=2.3, markersize=6, label=label)
    for _, label, color, _, marker, _ in SERIES
] + [
    mlines.Line2D([], [], color="#222222", linestyle="-", linewidth=2.3, label="Solid Line: Trajectory Evaluation (Includes State Drift) / True Simulator Rate"),
    mlines.Line2D([], [], color="#222222", linestyle=":", linewidth=2.0, label="Dotted Line: Same-State Valid Head Evaluation / Model-Predicted Rate"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=True,
    facecolor="#f8f9fa",
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.008),
)

plt.tight_layout(rect=[0, 0.095, 1, 0.93])
fig2_path = OUT_DIR / "fig2_comment15_reward_and_success_fidelity.png"
fig.savefig(fig2_path, dpi=200)
plt.close(fig)
print("Saved:", fig1_path, fig2_path)
