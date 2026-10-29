import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH_DIR = "/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/comment17"
REPO_OUT_DIR = os.path.expanduser("~/mborl/mborl/results/comment17")

with open(os.path.join(SCRATCH_DIR, "checkpoint_manifest.json")) as f:
    manifest_data = json.load(f)

with open(os.path.join(SCRATCH_DIR, "comment17_critic_diagnosis.json")) as f:
    diag_data = json.load(f)

ckpts = list(diag_data["checkpoints"].values()) if isinstance(diag_data["checkpoints"], dict) else diag_data["checkpoints"]
print(f"Loaded {len(ckpts)} evaluated checkpoints and {len(manifest_data['manifest'])} manifest rows.")

# Print clean summary table of the 8 evaluated checkpoints
print("\n=== SAVED CHECKPOINT EVALUATION SUMMARY ===")
for c in ckpts:
    sp = c["spec"]
    det = c["deterministic_eval"]
    sto = c["stochastic_eval"]
    sh = c["shared_set"]
    po = c["policy_set"]
    sh_ref_key = sh["candidate_names"][2]
    sh_alt_key = sh["candidate_names"][3]
    po_ref_key = po["candidate_names"][2]
    po_alt_key = po["candidate_names"][3]

    sh_cal_cur = sh["calibration"]["current_actor_actions"]
    sh_cal_alt = sh["calibration"]["alternative_actions"]
    po_cal_cur = po["calibration"]["current_actor_actions"]
    po_cal_alt = po["calibration"]["alternative_actions"]

    sh_rk_ref = sh["ranking_pairs"][sh_ref_key]
    sh_rk_dat = sh["ranking_pairs"][sh_alt_key]
    po_rk_ref = po["ranking_pairs"][po_ref_key]

    print(
        f"\n* {sp['label']} (cfg={sp['config_group']}, seed={sp['seed']}, step={sp['step']}, logged_succ={sp['logged_eval_success']*100:.1f}%)"
    )
    print(
        f"  Actor: alpha={sh['actor_stats']['lam_alpha']:.3f}, entropy={sh['actor_stats']['policy_entropy']:.2f}, std={sh['actor_stats']['action_std']:.3f}"
    )
    print(
        f"  Eval (15 ep): Det={det['success_rate']*100:.1f}% (J={det['discounted_return_mean']:.1f}±{det['discounted_return_se']:.1f}), "
        f"Stoch={sto['success_rate']*100:.1f}% (J={sto['discounted_return_mean']:.1f}±{sto['discounted_return_se']:.1f})"
    )
    print(
        f"  Shared Calib (cur a_eval/a_stoch): Q_actor={sh_cal_cur['q_actor_mean']:.1f}, Q_min={sh_cal_cur['q_min_mean']:.1f}, "
        f"Q_sim={sh_cal_cur['q_sim_trunc_mean']:.1f} [ep750_low={sh_cal_cur['q_sim_ep750_lower_mean']:.1f}], "
        f"Bias={sh_cal_cur['signed_bias_mean']:+.1f}±{sh_cal_cur['signed_bias_se']:.1f}, MAE={sh_cal_cur['mae_mean']:.1f}, "
        f"Q50/Q90/Q95={sh_cal_cur['bias_q50']:+.1f}/{sh_cal_cur['bias_q90']:+.1f}/{sh_cal_cur['bias_q95']:+.1f}"
    )
    print(
        f"  Shared Calib (alt {sh_ref_key}/{sh_alt_key}): Q_actor={sh_cal_alt['q_actor_mean']:.1f}, Q_min={sh_cal_alt['q_min_mean']:.1f}, "
        f"Q_sim={sh_cal_alt['q_sim_trunc_mean']:.1f}, Bias={sh_cal_alt['signed_bias_mean']:+.1f}±{sh_cal_alt['signed_bias_se']:.1f}, MAE={sh_cal_alt['mae_mean']:.1f}"
    )
    print(
        f"  Policy Calib (cur a_eval/a_stoch): Q_actor={po_cal_cur['q_actor_mean']:.1f}, Q_sim={po_cal_cur['q_sim_trunc_mean']:.1f} "
        f"[ep750_low={po_cal_cur['q_sim_ep750_lower_mean']:.1f}], Bias={po_cal_cur['signed_bias_mean']:+.1f}±{po_cal_cur['signed_bias_se']:.1f}, MAE={po_cal_cur['mae_mean']:.1f}, "
        f"Q50/Q90/Q95={po_cal_cur['bias_q50']:+.1f}/{po_cal_cur['bias_q90']:+.1f}/{po_cal_cur['bias_q95']:+.1f}"
    )
    print(
        f"  Shared Ranking vs {sh_ref_key}: Res={sh_rk_ref['num_resolved']}/50 (Inconc={sh_rk_ref['frac_inconclusive']*100:.1f}%), "
        f"WrongOpt(res)={sh_rk_ref['frac_confidently_wrong_optimistic_resolved']*100:.1f}% (all={sh_rk_ref['frac_confidently_wrong_optimistic_all']*100:.1f}%), "
        f"AnyWrong(res)={sh_rk_ref['frac_any_sign_error_resolved']*100:.1f}%, dQ={sh_rk_ref['mean_dq']:+.1f}, dSim={sh_rk_ref['mean_dsim']:+.1f}"
    )
    print(
        f"  Shared Ranking vs dataset_action: Res={sh_rk_dat['num_resolved']}/50 (Inconc={sh_rk_dat['frac_inconclusive']*100:.1f}%), "
        f"WrongOpt(res)={sh_rk_dat['frac_confidently_wrong_optimistic_resolved']*100:.1f}% (all={sh_rk_dat['frac_confidently_wrong_optimistic_all']*100:.1f}%), "
        f"AnyWrong(res)={sh_rk_dat['frac_any_sign_error_resolved']*100:.1f}%, dQ={sh_rk_dat['mean_dq']:+.1f}, dSim={sh_rk_dat['mean_dsim']:+.1f}"
    )
    print(
        f"  Policy Ranking vs {po_ref_key}: Res={po_rk_ref['num_resolved']}/25 (Inconc={po_rk_ref['frac_inconclusive']*100:.1f}%), "
        f"WrongOpt(res)={po_rk_ref['frac_confidently_wrong_optimistic_resolved']*100:.1f}% (all={po_rk_ref['frac_confidently_wrong_optimistic_all']*100:.1f}%), "
        f"AnyWrong(res)={po_rk_ref['frac_any_sign_error_resolved']*100:.1f}%, dQ={po_rk_ref['mean_dq']:+.1f}, dSim={po_rk_ref['mean_dsim']:+.1f}"
    )
    print(
        f"  Regret: Shared={sh['regret']['mean_regret']:.2f}±{sh['regret']['bootstrap_se']:.2f} (med={sh['regret']['median_regret']:.2f}, q90={sh['regret']['q90_regret']:.2f}, zero={sh['regret']['frac_zero_regret']*100:.1f}%), "
        f"Policy={po['regret']['mean_regret']:.2f}±{po['regret']['bootstrap_se']:.2f} (med={po['regret']['median_regret']:.2f}, q90={po['regret']['q90_regret']:.2f}, zero={po['regret']['frac_zero_regret']*100:.1f}%)"
    )

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

ts = manifest_data["timeseries"]
steps_k = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

cfg_styles = [
    ("Original_H5_50k", "Original H=5 (50k starts)", "#D93025", "-"),
    ("Exp1_H5_100k", "Exp 1 H=5 (100k starts)", "#E37400", "--"),
    ("Original_H10_50k", "Original H=10 (50k starts)", "#188038", "-"),
    ("CQL_f1.0", "CQL Reference (f=1.0)", "#1A73E8", ":"),
]

# ==============================================================================
# FIGURE 1: Policy Performance Over Training & Saved-Checkpoint Re-Evaluation
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=220)

# Panel A: Training trajectories (100k to 1M) across 4 primary configs (individual seeds + mean)
ax = axes[0]
for key, lbl, col, ls in cfg_styles:
    seed_dict = ts[key]["seeds"]
    all_s = []
    for s_str, s_data in sorted(seed_dict.items()):
        vals = [row["eval_success"] * 100.0 for row in s_data["steps"]]
        all_s.append(vals)
        ax.plot(steps_k, vals, color=col, linestyle=ls, alpha=0.28, linewidth=1.2)
    mean_vals = np.mean(all_s, axis=0)
    ax.plot(steps_k, mean_vals, color=col, linestyle=ls, linewidth=2.6, marker="o", markersize=4.5, label=f"{lbl} (mean)")

# Annotate the two 500k Seed 0 points explicitly!
ax.scatter([500], [86.7], color="#E37400", s=70, zorder=6, edgecolor="black", linewidth=1.2)
ax.annotate("Exp 1 H=5 (100k) s0 @ 500k: 86.7%\n(overwritten at 600k->1M)", xy=(500, 86.7), xytext=(530, 93),
            fontsize=8, fontweight="bold", color="#B06000",
            arrowprops=dict(arrowstyle="->", color="#B06000", lw=1.0))

ax.scatter([500], [53.3], color="#9334E6", s=70, zorder=6, marker="D", edgecolor="black", linewidth=1.2)
ax.annotate("Saved 500k Ref (b=0.5, H=10) s0:\n53.3% (collapsed to 0% @ 1M)", xy=(500, 53.3), xytext=(535, 56),
            fontsize=8, fontweight="bold", color="#9334E6",
            arrowprops=dict(arrowstyle="->", color="#9334E6", lw=1.0))

ax.set_title("A. Logged Success Over Training (Seeds 0, 1, 2 & Mean)")
ax.set_xlabel("Training Step (thousands)")
ax.set_ylabel("Evaluation Success Rate (%)")
ax.set_ylim(-4, 105)
ax.set_xlim(80, 1020)
ax.legend(loc="lower left", framealpha=0.92)

# Panel B: Deterministic vs Stochastic Success Rate on the 8 Saved Checkpoints
ax = axes[1]
short_labels = [
    "Ref 500k\n(b=0.5,H=10,s0)",
    "Exp1 H=5\n(100k, s0, 1M)",
    "Exp1 H=5\n(100k, s1, 1M)",
    "Exp1 H=5\n(100k, s2, 1M)",
    "Exp2A d1-5\n(Frozen, s0)",
    "Exp2B d6-10\n(Frozen, s0)",
    "Exp2A d1-5\n(Frozen, s1)",
    "Exp2B d6-10\n(Frozen, s1)",
]
x = np.arange(len(ckpts))
w = 0.27
logged_succ = [c["spec"]["logged_eval_success"] * 100.0 for c in ckpts]
det_succ = [c["deterministic_eval"]["success_rate"] * 100.0 for c in ckpts]
sto_succ = [c["stochastic_eval"]["success_rate"] * 100.0 for c in ckpts]

b1 = ax.bar(x - w, logged_succ, width=w, color="#5F6368", alpha=0.85, label="Logged Eval (train run)")
b2 = ax.bar(x, det_succ, width=w, color="#1A73E8", alpha=0.9, label="Re-Eval Deterministic (temp=0)")
b3 = ax.bar(x + w, sto_succ, width=w, color="#188038", alpha=0.9, label="Re-Eval Stochastic (temp=1)")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2)

ax.set_title("B. Saved Checkpoints: Deterministic vs. Stochastic Success")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Task 1 Success Rate (%, 15 episodes)")
ax.set_ylim(0, 102)
ax.legend(loc="upper right", framealpha=0.92)

# Panel C: Deterministic vs Stochastic Discounted Return (gamma=0.999)
ax = axes[2]
det_ret = [c["deterministic_eval"]["discounted_return_mean"] for c in ckpts]
det_se = [c["deterministic_eval"]["discounted_return_se"] for c in ckpts]
sto_ret = [c["stochastic_eval"]["discounted_return_mean"] for c in ckpts]
sto_se = [c["stochastic_eval"]["discounted_return_se"] for c in ckpts]

ax.bar(x - w/2, det_ret, width=w*1.2, yerr=det_se, capsize=3, color="#1A73E8", alpha=0.9, label="Deterministic (temp=0)")
ax.bar(x + w/2, sto_ret, width=w*1.2, yerr=sto_se, capsize=3, color="#188038", alpha=0.9, label="Stochastic (temp=1)")
ax.axhline(-1055.6, color="#D93025", linestyle="--", linewidth=1.2, label="0%-Success Floor (-1055.6)")

ax.set_title("C. Discounted Simulator Return (γ = 0.999)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Discounted Return J^π (higher / less negative is better)")
ax.set_ylim(-1180, -350)
ax.legend(loc="lower right", framealpha=0.92)

fig.tight_layout()
for out_d in [SCRATCH_DIR, REPO_OUT_DIR]:
    fig.savefig(os.path.join(out_d, "fig1_comment17_policy_performance.png"), dpi=220)
plt.close(fig)

# ==============================================================================
# FIGURE 2: Critic Calibration Over Training & Monte Carlo Return Bias
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=220)

# Panel A: Validation Critic Advantage dQ_val = Q_val(s, a_pi) - Q_val(s, a_data) over training
ax = axes[0]
ax.axhline(0.0, color="black", linestyle="-", linewidth=1.1)
ax.axhspan(0, 55, color="#FCE8E6", alpha=0.45, label="Optimistic OOD Preference (Q_π > Q_data)")
for key, lbl, col, ls in cfg_styles:
    seed_dict = ts[key]["seeds"]
    all_dq = []
    for s_str, s_data in sorted(seed_dict.items()):
        vals = [row["val_dq_pi_minus_data"] for row in s_data["steps"]]
        all_dq.append(vals)
        ax.plot(steps_k, vals, color=col, linestyle=ls, alpha=0.30, linewidth=1.2)
    mean_vals = np.mean(all_dq, axis=0)
    ax.plot(steps_k, mean_vals, color=col, linestyle=ls, linewidth=2.5, marker="o", markersize=4, label=f"{lbl} (mean)")

# Highlight Exp1 H=5 s0 spike at 400k (+21.8) and 700k (+40.6)
s0_exp1_dq = [row["val_dq_pi_minus_data"] for row in ts["Exp1_H5_100k"]["seeds"]["0"]["steps"]]
ax.plot(steps_k, s0_exp1_dq, color="#D93025", linestyle="-.", linewidth=2.0, marker="s", markersize=4.5, label="Exp 1 H=5 Seed 0 (collapses @ 600k)")
ax.annotate("s0 spikes to +21.8 @ 400k\n(right before 600k collapse)", xy=(400, 21.8), xytext=(430, 36),
            fontsize=8, fontweight="bold", color="#C5221F",
            arrowprops=dict(arrowstyle="->", color="#C5221F", lw=1.0))

ax.set_title("A. Validation Critic Advantage Q(s, a_π) - Q(s, a_data)")
ax.set_xlabel("Training Step (thousands)")
ax.set_ylabel("Validation ΔQ_val (Critic Units)")
ax.set_ylim(-115, 52)
ax.set_xlim(80, 1020)
ax.legend(loc="lower left", fontsize=7.8, framealpha=0.92)

# Panel B: Shared Set (50 states) Signed Calibration Bias (Q_actor - Q_sim)
ax = axes[1]
cur_bias_sh = [c["shared_set"]["calibration"]["current_actor_actions"]["signed_bias_mean"] for c in ckpts]
cur_se_sh = [c["shared_set"]["calibration"]["current_actor_actions"]["signed_bias_se"] for c in ckpts]
alt_bias_sh = [c["shared_set"]["calibration"]["alternative_actions"]["signed_bias_mean"] for c in ckpts]
alt_se_sh = [c["shared_set"]["calibration"]["alternative_actions"]["signed_bias_se"] for c in ckpts]
cur_q90_sh = [c["shared_set"]["calibration"]["current_actor_actions"]["bias_q90"] for c in ckpts]

ax.axhline(0.0, color="black", linewidth=1.0)
ax.bar(x - w/2, cur_bias_sh, width=w*1.2, yerr=cur_se_sh, capsize=3, color="#D93025", alpha=0.88, label="Current Actor Actions (a_eval, a_stoch)")
ax.bar(x + w/2, alt_bias_sh, width=w*1.2, yerr=alt_se_sh, capsize=3, color="#1A73E8", alpha=0.88, label="Alternative Actions (a_ref_good, a_data)")
ax.scatter(x - w/2, cur_q90_sh, color="#B06000", marker="^", s=42, zorder=5, label="Current Actor 90th Pct Error")

ax.set_title("B. Shared Set (50 States): Signed Bias (Q_actor - Q_sim)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Signed Calibration Bias ε_Q = Q_actor - Q_sim^π")
ax.set_ylim(-6600, 1600)
ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

# Panel C: Policy-Relevant Set (25 states) Signed Calibration Bias (Q_actor - Q_sim)
ax = axes[2]
cur_bias_po = [c["policy_set"]["calibration"]["current_actor_actions"]["signed_bias_mean"] for c in ckpts]
cur_se_po = [c["policy_set"]["calibration"]["current_actor_actions"]["signed_bias_se"] for c in ckpts]
alt_bias_po = [c["policy_set"]["calibration"]["alternative_actions"]["signed_bias_mean"] for c in ckpts]
alt_se_po = [c["policy_set"]["calibration"]["alternative_actions"]["signed_bias_se"] for c in ckpts]
cur_q90_po = [c["policy_set"]["calibration"]["current_actor_actions"]["bias_q90"] for c in ckpts]

ax.axhline(0.0, color="black", linewidth=1.0)
ax.bar(x - w/2, cur_bias_po, width=w*1.2, yerr=cur_se_po, capsize=3, color="#D93025", alpha=0.88, label="Current Actor Actions (a_eval, a_stoch)")
ax.bar(x + w/2, alt_bias_po, width=w*1.2, yerr=alt_se_po, capsize=3, color="#1A73E8", alpha=0.88, label="Alternative Actions (a_ref_good, a_pert)")
ax.scatter(x - w/2, cur_q90_po, color="#B06000", marker="^", s=42, zorder=5, label="Current Actor 90th Pct Error")

ax.set_title("C. Policy Set (25 States): Signed Bias (Q_actor - Q_sim)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Signed Calibration Bias ε_Q = Q_actor - Q_sim^π")
ax.set_ylim(-4000, 950)
ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

fig.tight_layout()
for out_d in [SCRATCH_DIR, REPO_OUT_DIR]:
    fig.savefig(os.path.join(out_d, "fig2_comment17_critic_calibration.png"), dpi=220)
plt.close(fig)

# ==============================================================================
# FIGURE 3: Action-Ranking Sign Errors & Candidate-Set Regret
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), dpi=220)

# Panel A: Confidently Wrong Optimistic Preference & Total Sign Errors (Resolved Pairs vs Reference Good Policy)
ax = axes[0]
w_opt_sh = [c["shared_set"]["ranking_pairs"][c["shared_set"]["candidate_names"][2]]["frac_confidently_wrong_optimistic_resolved"] * 100.0 for c in ckpts]
any_err_sh = [c["shared_set"]["ranking_pairs"][c["shared_set"]["candidate_names"][2]]["frac_any_sign_error_resolved"] * 100.0 for c in ckpts]
w_opt_po = [c["policy_set"]["ranking_pairs"][c["policy_set"]["candidate_names"][2]]["frac_confidently_wrong_optimistic_resolved"] * 100.0 for c in ckpts]
any_err_po = [c["policy_set"]["ranking_pairs"][c["policy_set"]["candidate_names"][2]]["frac_any_sign_error_resolved"] * 100.0 for c in ckpts]

ax.bar(x - 1.5*w/1.3, w_opt_sh, width=w*0.75, color="#D93025", alpha=0.92, label="Shared Set: Wrong Optimistic (ΔQ>0, Δsim<0)")
ax.bar(x - 0.5*w/1.3, any_err_sh, width=w*0.75, color="#F29900", alpha=0.85, label="Shared Set: Total Sign Error (Resolved)")
ax.bar(x + 0.5*w/1.3, w_opt_po, width=w*0.75, color="#9334E6", alpha=0.90, label="Policy Set: Wrong Optimistic (ΔQ>0, Δsim<0)")
ax.bar(x + 1.5*w/1.3, any_err_po, width=w*0.75, color="#1A73E8", alpha=0.80, label="Policy Set: Total Sign Error (Resolved)")

ax.axhline(50.0, color="#5F6368", linestyle="--", linewidth=1.1, label="Random Coin-Flip Error (50%)")
ax.set_title("A. Action-Ranking Sign Error Rate (a_eval vs. Verified Reference Action)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Sign Error Rate Among Resolved States (%)")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

# Panel B: Candidate-Set Regret (Discounted Return Lost per State Due to Critic Choice)
ax = axes[1]
reg_sh = [c["shared_set"]["regret"]["mean_regret"] for c in ckpts]
reg_sh_se = [c["shared_set"]["regret"]["bootstrap_se"] for c in ckpts]
reg_po = [c["policy_set"]["regret"]["mean_regret"] for c in ckpts]
reg_po_se = [c["policy_set"]["regret"]["bootstrap_se"] for c in ckpts]
reg_sh_q90 = [c["shared_set"]["regret"]["q90_regret"] for c in ckpts]
reg_po_q90 = [c["policy_set"]["regret"]["q90_regret"] for c in ckpts]

b_sh = ax.bar(x - w/2, reg_sh, width=w*1.2, yerr=reg_sh_se, capsize=3, color="#D93025", alpha=0.88, label="Shared Validation Set (50 states): Mean ± Boot SE")
b_po = ax.bar(x + w/2, reg_po, width=w*1.2, yerr=reg_po_se, capsize=3, color="#1A73E8", alpha=0.88, label="Policy-Relevant Set (25 states): Mean ± Boot SE")
ax.scatter(x - w/2, reg_sh_q90, color="#B06000", marker="D", s=36, zorder=5, label="Shared Set 90th Pct Regret")
ax.scatter(x + w/2, reg_po_q90, color="#0D47A1", marker="D", s=36, zorder=5, label="Policy Set 90th Pct Regret")

for i_b, bar in enumerate(b_sh):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + reg_sh_se[i_b] + 2.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#B31412", fontweight="bold")
for i_b, bar in enumerate(b_po):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + reg_po_se[i_b] + 2.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#0D47A1", fontweight="bold")

ax.set_title("B. Candidate-Set Decision Regret (Return Lost Under Critic argmax_a Q)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=28, ha="right")
ax.set_ylabel("Discounted Return Regret (lower is better; 0 = optimal choice)")
ax.set_ylim(0, max(max(reg_sh_q90), max(reg_po_q90)) * 1.18)
ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

fig.tight_layout()
for out_d in [SCRATCH_DIR, REPO_OUT_DIR]:
    fig.savefig(os.path.join(out_d, "fig3_comment17_decision_errors_and_regret.png"), dpi=220)
plt.close(fig)

# ==============================================================================
# FIGURE 4: Predicted Advantage (ΔQ) vs. Actual Simulator Advantage (Δ_sim) Scatter
# ==============================================================================
focus_indices = [0, 1, 2, 3, 4, 5]
clean_titles = [
    "Ref 500k (β=0.5, H=10, s0)",
    "Exp 1 H=5 (100k, s0, 1M — Collapsed)",
    "Exp 1 H=5 (100k, s1, 1M — Stable)",
    "Exp 1 H=5 (100k, s2, 1M)",
    "Exp 2A Frozen Depths 1–5 (s0, 1M)",
    "Exp 2B Frozen Depths 6–10 (s0, 1M)",
]
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=220)

for panel_idx, ck_idx in enumerate(focus_indices):
    r, col_i = divmod(panel_idx, 3)
    ax = axes[r, col_i]
    c = ckpts[ck_idx]
    sp = c["spec"]

    sh_ref_name = c["shared_set"]["candidate_names"][2]
    po_ref_name = c["policy_set"]["candidate_names"][2]
    sh_pair = c["shared_set"]["ranking_pairs"][sh_ref_name]
    po_pair = c["policy_set"]["ranking_pairs"][po_ref_name]

    dq_sh = np.array(sh_pair["per_state_dq"])
    ds_sh = np.array(sh_pair["per_state_dsim"])
    se_sh = np.array(sh_pair["per_state_dsim_se"])
    res_sh = np.array(sh_pair["per_state_resolved"], dtype=bool)

    dq_po = np.array(po_pair["per_state_dq"])
    ds_po = np.array(po_pair["per_state_dsim"])
    se_po = np.array(po_pair["per_state_dsim_se"])
    res_po = np.array(po_pair["per_state_resolved"], dtype=bool)

    all_ds = np.concatenate([ds_sh, ds_po])
    all_dq = np.concatenate([dq_sh, dq_po])
    x_lim = max(45.0, float(np.max(np.abs(all_ds)) * 1.15))
    y_lim = max(15.0, float(np.max(np.abs(all_dq)) * 1.18))

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    ax.axvline(0.0, color="black", linewidth=1.0, zorder=2)

    # Shade Upper-Left Quadrant (Confidently Wrong Optimistic: dQ > 0 but dSim < 0)
    ax.fill_between([-x_lim, 0], 0, y_lim, color="#FCE8E6", alpha=0.55, zorder=1, label="Wrong Optimistic (ΔQ>0, Δsim<0)")
    # Shade Lower-Right Quadrant (Wrong Pessimistic: dQ < 0 but dSim > 0)
    ax.fill_between([0, x_lim], -y_lim, 0, color="#FEF7E0", alpha=0.45, zorder=1, label="Wrong Pessimistic (ΔQ<0, Δsim>0)")

    # Plot Shared states
    ax.errorbar(ds_sh[res_sh], dq_sh[res_sh], xerr=se_sh[res_sh], fmt="o", markersize=5.0,
                color="#D93025", ecolor="#D93025", elinewidth=0.8, alpha=0.82, label="Shared Resolved")
    ax.scatter(ds_sh[~res_sh], dq_sh[~res_sh], marker="o", s=22, facecolors="none", edgecolors="#9AA0A6", alpha=0.7, label="Inconclusive Near-Tie")

    # Plot Policy-relevant states
    ax.errorbar(ds_po[res_po], dq_po[res_po], xerr=se_po[res_po], fmt="^", markersize=5.5,
                color="#1A73E8", ecolor="#1A73E8", elinewidth=0.8, alpha=0.88, label="Policy-Set Resolved")
    ax.scatter(ds_po[~res_po], dq_po[~res_po], marker="^", s=24, facecolors="none", edgecolors="#9AA0A6", alpha=0.7)

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    w_sh_pct = sh_pair["frac_confidently_wrong_optimistic_resolved"] * 100.0
    w_po_pct = po_pair["frac_confidently_wrong_optimistic_resolved"] * 100.0
    ax.set_title(f"{clean_titles[panel_idx]}\n(Wrong Optimistic: Shared {w_sh_pct:.0f}%, Policy {w_po_pct:.0f}%)")
    ax.set_xlabel("Actual Simulator Advantage Δ_sim(s) = Q_sim(a_eval) - Q_sim(a_ref)")
    ax.set_ylabel("Predicted Critic Advantage Δ_Q(s)")
    if panel_idx == 0:
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

fig.suptitle("Predicted Critic Advantage (ΔQ) vs. True Simulator Rollout Advantage (Δ_sim) Across States", fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout()
for out_d in [SCRATCH_DIR, REPO_OUT_DIR]:
    fig.savefig(os.path.join(out_d, "fig4_comment17_advantage_scatter.png"), dpi=220)
plt.close(fig)
print("\nSuccessfully generated all 4 Comment 17 figures!")
