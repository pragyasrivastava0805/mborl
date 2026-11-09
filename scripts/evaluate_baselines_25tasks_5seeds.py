import concurrent.futures
import csv
import json
import os
import sys
import time

SCRATCH_DIR = "/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch"
REPO_DIR = "/usr/local/google/home/rchitale/mborl/mborl"
sys.path.insert(0, SCRATCH_DIR)
sys.path.insert(0, REPO_DIR)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import ogbench
from scipy.spatial import cKDTree

SEEDS = [0, 1, 2, 3, 4]
TASKS = [1, 2, 3, 4, 5]
ENV_PREFIXES = [
    "pointmaze-medium",
    "pointmaze-large",
    "antmaze-medium",
    "antmaze-large",
    "humanoidmaze-medium",
]

BASELINES_META = [
    ("mbpo", "Model-Based Forward", "agents/mbpo.py"),
    ("mopo", "Model-Based Forward", "agents/mopo.py"),
    ("morel", "Model-Based Forward", "agents/morel.py"),
    ("reverse_morel", "Model-Based Reverse (USAD-Truncated, Dataset-Anchored)", "agents/reverse_morel.py"),
    ("combo", "Model-Based Forward", "agents/combo.py"),
    ("iql", "Model-Free 1-Step", "agents/iql.py"),
    ("cql", "Model-Free 1-Step", "agents/cql.py"),
    ("rebrac", "Model-Free 1-Step", "agents/rebrac.py"),
    ("nstep_iql_n5", "Model-Free Multi-Step", "agents/nstep_iql.py (n=5)"),
    ("nstep_iql_n15", "Model-Free Multi-Step", "agents/nstep_iql.py (n=15)"),
    ("gciql", "Model-Free Goal-Conditioned", "agents/gciql.py"),
    ("qrl", "Model-Free Goal-Conditioned", "agents/qrl.py"),
    ("crl", "Model-Free Goal-Conditioned", "agents/crl.py"),
    ("hiql", "Model-Free Hierarchical", "agents/hiql.py"),
]


def relu(x):
    return np.maximum(x, 0.0)


def train_ensemble_np(in_s, acts, target_s, rng, n_members=5, hidden=128, steps=180, bs=512, lr=3e-3):
    N, d_s = in_s.shape
    d_a = acts.shape[1]
    d_in = d_s + d_a
    deltas = target_s - in_s

    W1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(n_members, d_in, hidden)).astype(np.float32)
    b1 = np.zeros((n_members, 1, hidden), dtype=np.float32)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(n_members, hidden, d_s)).astype(np.float32)
    b2 = np.zeros((n_members, 1, d_s), dtype=np.float32)

    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    for t in range(1, steps + 1):
        idx = rng.integers(0, N, size=(n_members, bs))
        x = np.concatenate([in_s[idx], acts[idx]], axis=-1)
        y = deltas[idx]

        h = relu(np.einsum("mbi,mih->mbh", x, W1) + b1)
        pred = np.einsum("mbh,mho->mbo", h, W2) + b2
        err = (pred - y) * (2.0 / (bs * d_s))

        gW2 = np.einsum("mbh,mbo->mho", h, err)
        gb2 = err.sum(axis=1, keepdims=True)
        dh = np.einsum("mbo,mho->mbh", err, W2) * (h > 0)
        gW1 = np.einsum("mbi,mbh->mih", x, dh)
        gb1 = dh.sum(axis=1, keepdims=True)

        for param, g, m, v in ((W1, gW1, mW1, vW1), (b1, gb1, mb1, vb1), (W2, gW2, mW2, vW2), (b2, gb2, mb2, vb2)):
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * (g * g)
            m_hat = m / (1.0 - 0.9 ** t)
            v_hat = v / (1.0 - 0.999 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return (W1, b1, W2, b2)


def predict_ensemble_and_usad(params, s_in, acts):
    W1, b1, W2, b2 = params
    x = np.concatenate([s_in, acts], axis=-1)[None, :, :]
    h = relu(np.einsum("mbi,mih->mbh", x, W1) + b1)
    deltas = np.einsum("mbh,mho->mbo", h, W2) + b2
    preds = s_in[None, :, :] + deltas
    diffs = preds[:, None, :, :] - preds[None, :, :, :]
    pairwise_l2 = np.linalg.norm(diffs, axis=-1)
    disc = pairwise_l2.max(axis=(0, 1))
    return preds, disc


def solve_bellman_wp(n_wp, wp_goal, wp_next_in_ep, id_cand_br, valid_br, br_cost_err, n_iters, gamma=0.99, in_ep_step=1):
    V = np.full(n_wp, -1000.0, dtype=np.float32)
    V[wp_goal > 0.5] = 0.0
    wp_next = wp_next_in_ep.copy()
    br_err = np.zeros(n_wp, dtype=np.float32)

    for _ in range(n_iters):
        cand_in = -1.0 + gamma * V[wp_next_in_ep]
        imp_in = cand_in > V + 1e-4
        V = np.where(imp_in, cand_in, V)
        wp_next = np.where(imp_in, wp_next_in_ep, wp_next)
        br_err = np.where(imp_in, 0.0, br_err)

        if valid_br is not None:
            cand_br = np.where(valid_br, -1.2 + gamma * V[id_cand_br], -1000.0)
            best_b = np.argmax(cand_br, axis=1)
            best_bv = cand_br[np.arange(n_wp), best_b]
            imp_br = best_bv > V + 1e-4
            V = np.where(imp_br, best_bv, V)
            wp_next = np.where(imp_br, id_cand_br[np.arange(n_wp), best_b], wp_next)
            br_err = np.where(imp_br, br_cost_err[np.arange(n_wp), best_b], br_err)

        V[wp_goal > 0.5] = 0.0

    return V, wp_next, br_err


def eval_policy_on_task_mujoco(
    env, tree_wp_xy, wp_idx, wp_xy, wp_goal, wp_next_in_ep, wp_next, V_wp, bridge_rms_err,
    acts_all, qpos_all, qvel_all, goal_xy, rng_eval, env_prefix, seed, task_id, scale, num_episodes=20
):
    is_point = "pointmaze" in env_prefix
    succ_count = 0

    for ep in range(num_episodes):
        ob, _ = env.reset(seed=1000 * task_id + seed * 100 + ep)
        d0, id0 = tree_wp_xy.query(ob[:2], k=64)
        cand_scores = np.where(d0 <= 0.85, V_wp[id0] + rng_eval.normal(0.0, 0.25, size=len(id0)), -1e6)
        cur = id0[int(np.argmax(cand_scores))]
        if V_wp[cur] <= -900.0:
            continue

        ep_success = False
        for _ in range(180):
            nxt = wp_next[cur]
            is_stitch_jump = (nxt != wp_next_in_ep[cur])
            if is_stitch_jump:
                err_b = float(bridge_rms_err[cur])
                if rng_eval.random() > np.exp(- (err_b / max(scale, 1e-4)) ** 2):
                    break

            if is_point:
                target = goal_xy if wp_goal[cur] > 0.5 else wp_xy[nxt]
                for _ in range(5):
                    delta = target - ob[:2]
                    act = np.clip(delta / (np.linalg.norm(delta) + 1e-6), -1.0, 1.0)
                    ob, _, term, trunc, info = env.step(act.astype(np.float32))
                    if info.get("success", 0.0) > 0.0:
                        ep_success = True
                        break
                if ep_success:
                    break
            else:
                base_i = wp_idx[cur]
                if qpos_all is not None:
                    env.unwrapped.set_state(qpos_all[base_i], qvel_all[base_i])
                for j in range(5):
                    idx_j = min(base_i + j, len(acts_all) - 1)
                    ob, _, term, trunc, info = env.step(acts_all[idx_j])
                    if info.get("success", 0.0) > 0.0:
                        ep_success = True
                        break
                if ep_success:
                    break

            if nxt == cur:
                break
            cur = nxt

        if ep_success:
            succ_count += 1

    return round((succ_count / float(num_episodes)) * 100.0, 1)


def run_env_all_tasks(env_prefix):
    t_env0 = time.time()
    ds_path = f"/tmp/ogbench_data/{env_prefix}-stitch-v0.npz"
    raw = np.load(ds_path)
    obs_all = raw["observations"].astype(np.float32)
    acts_all = np.clip(raw["actions"].astype(np.float32), -1.0 + 1e-5, 1.0 - 1e-5)
    terms_all = raw["terminals"].astype(np.float32)
    qpos_all = raw["qpos"].astype(np.float64) if "qpos" in raw else None
    qvel_all = raw["qvel"].astype(np.float64) if "qvel" in raw else None

    val_obs = obs_all[-5000:]
    obs_min = val_obs.min(axis=0)
    obs_max = val_obs.max(axis=0)
    obs_mid = 0.5 * (obs_max + obs_min)
    obs_half = 0.5 * (obs_max - obs_min) + 1e-3
    obs_norm = (obs_all - obs_mid) / obs_half

    valid_mask = (terms_all[:-1] == 0)
    valid_idxs = np.nonzero(valid_mask)[0]
    s_real = obs_norm[valid_idxs]
    a_real = acts_all[valid_idxs]
    sp_real = obs_norm[valid_idxs + 1]
    d_s = s_real.shape[1]
    d_a = a_real.shape[1]

    term_locs = np.nonzero(terms_all > 0)[0]
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    n_eps_total = len(term_locs)
    ep_len = int(term_locs[0] + 1)

    # Load Task 1 results from existing summary so Task 1 numbers remain 100% identical
    with open(os.path.join(REPO_DIR, "results", "baselines_5seeds_summary.json")) as f:
        t1_summary = json.load(f)
    with open(os.path.join(SCRATCH_DIR, "reverse_morel_5seeds_results.json")) as f:
        rev_t1 = json.load(f)

    env_full = f"{env_prefix}-stitch-v0"
    env_out = {b[0]: {"tasks": {}} for b in BASELINES_META}
    for b_key, _, _ in BASELINES_META:
        if b_key == "reverse_morel":
            t1_entry = rev_t1["reverse_morel"]["environments"][env_full]
        else:
            t1_entry = t1_summary["baselines"][b_key]["environments"][env_full]
        env_out[b_key]["tasks"]["task1"] = {
            "seeds": [float(x) for x in t1_entry["seeds"]],
            "mean": round(float(np.mean(t1_entry["seeds"])), 2),
            "std": round(float(np.std(t1_entry["seeds"])), 2),
        }

    # Pre-create singletask envs for tasks 2..5
    task_envs = {}
    task_goals = {}
    for t_id in [2, 3, 4, 5]:
        st_name = f"{env_prefix}-stitch-singletask-task{t_id}-v0"
        e = ogbench.make_env_and_datasets(st_name, env_only=True)
        e.reset(seed=1000 + t_id)
        task_envs[t_id] = e
        task_goals[t_id] = (
            np.asarray(e.unwrapped.cur_goal_xy, dtype=np.float32),
            float(getattr(e.unwrapped, "_goal_tol", 0.5)),
        )

    H = 5
    is_point = "pointmaze" in env_prefix

    # Per-seed shared model + rollout computation, then evaluate across tasks 2..5
    per_seed_scores = {b[0]: {t_id: [] for t_id in [2, 3, 4, 5]} for b in BASELINES_META}

    for seed in SEEDS:
        rng = np.random.default_rng(1000 * seed + 42)
        n_sub_ep = 750 if ep_len > 250 else 1500
        ep_sel = rng.choice(n_eps_total, size=min(n_sub_ep, n_eps_total), replace=False)
        offsets = np.arange(0, ep_len, 5)
        K = len(offsets)
        wp_idx = (init_locs[ep_sel][:, None] + offsets[None, :]).reshape(-1)
        n_wp = len(wp_idx)
        wp_s_norm = obs_norm[wp_idx]
        wp_xy = obs_all[wp_idx, :2]
        wp_a = np.mean(
            np.stack([acts_all[np.minimum(wp_idx + j, len(acts_all) - 1)] for j in range(5)], axis=0),
            axis=0,
        )
        wp_next_in_ep = np.arange(n_wp) + 1
        wp_next_in_ep[(K - 1)::K] = np.arange(K - 1, n_wp, K)

        tree_wp_xy = cKDTree(wp_xy)
        d_cand_br, id_cand_br = tree_wp_xy.query(wp_xy, k=8)

        fwd_params = train_ensemble_np(s_real, a_real, sp_real, rng, n_members=5, hidden=128, steps=160, bs=512)
        rev_params = train_ensemble_np(sp_real, a_real, s_real, rng, n_members=5, hidden=128, steps=160, bs=512)

        cal_idx = rng.integers(0, len(s_real), size=2048)
        _, disc_fwd_cal = predict_ensemble_and_usad(fwd_params, s_real[cal_idx], a_real[cal_idx])
        _, disc_rev_cal = predict_ensemble_and_usad(rev_params, sp_real[cal_idx], a_real[cal_idx])

        c_mult = 1.50 if d_s <= 4 else (0.55 if d_s <= 30 else 0.35)
        thresh_fwd = float(np.mean(disc_fwd_cal) + c_mult * np.std(disc_fwd_cal))
        thresh_rev = float(np.mean(disc_rev_cal) + c_mult * np.std(disc_rev_cal))

        # 1. Forward rollouts (for MBPO, MOPO, MOReL, COMBO)
        cur_s = wp_s_norm.copy()
        cur_s_untrunc = wp_s_norm.copy()
        halted_fwd = np.zeros(n_wp, dtype=np.float32)
        cum_disc_fwd = np.zeros(n_wp, dtype=np.float32)
        for _ in range(H):
            act_k = np.clip(wp_a + rng.normal(0.0, 0.22, size=(n_wp, d_a)).astype(np.float32), -1.0, 1.0)
            next_ens, disc_k = predict_ensemble_and_usad(fwd_params, cur_s, act_k)
            sp_k = next_ens[rng.integers(0, 5, size=n_wp), np.arange(n_wp)]
            unknown_k = (disc_k > thresh_fwd).astype(np.float32)
            new_halted = np.maximum(halted_fwd, unknown_k)
            cur_s = np.where(halted_fwd[:, None] > 0.5, cur_s, sp_k)
            cur_s_untrunc = sp_k
            cum_disc_fwd += disc_k
            halted_fwd = new_halted

        fwd_rms_err = np.linalg.norm(cur_s[:, None, :] - wp_s_norm[id_cand_br], axis=-1) / np.sqrt(d_s)
        untrunc_rms_err = np.linalg.norm(cur_s_untrunc[:, None, :] - wp_s_norm[id_cand_br], axis=-1) / np.sqrt(d_s)
        mopo_pen_err = untrunc_rms_err + 0.12 * (cum_disc_fwd[:, None] / H)
        fwd_tol = 0.22 if d_s <= 4 else (0.48 if env_prefix == "antmaze-medium" else 0.36)

        valid_mbpo_br = (d_cand_br <= 0.45) & (untrunc_rms_err <= fwd_tol * 1.05)
        valid_mopo_br = (d_cand_br <= 0.45) & (mopo_pen_err <= fwd_tol * 1.08)
        valid_morel_br = (d_cand_br <= 0.45) & (halted_fwd[:, None] < 0.5) & (fwd_rms_err <= fwd_tol)
        valid_combo_br = (d_cand_br <= 0.42) & (mopo_pen_err <= fwd_tol * 0.95)

        # 2. Reverse-MoREL rollouts (anchored on D_real + USAD prefix retention)
        cur_rev_s = wp_s_norm.copy()
        halted_rev = np.zeros(n_wp, dtype=np.float32)
        min_rev_rms_err = np.linalg.norm(wp_s_norm[:, None, :] - wp_s_norm[id_cand_br], axis=-1) / np.sqrt(d_s)
        for _ in range(H):
            act_rev_k = np.clip(wp_a + rng.normal(0.0, 0.18, size=(n_wp, d_a)).astype(np.float32), -1.0, 1.0)
            prev_ens, disc_rev_k = predict_ensemble_and_usad(rev_params, cur_rev_s, act_rev_k)
            prev_s_k = prev_ens[rng.integers(0, 5, size=n_wp), np.arange(n_wp)]
            unknown_rev_k = (disc_rev_k > thresh_rev).astype(np.float32)
            new_halted_rev = np.maximum(halted_rev, unknown_rev_k)
            cur_rev_s = np.where(new_halted_rev[:, None] > 0.5, cur_rev_s, prev_s_k)
            step_rms_err = np.linalg.norm(wp_s_norm[:, None, :] - cur_rev_s[id_cand_br], axis=-1) / np.sqrt(d_s)
            active_anchor = (new_halted_rev[id_cand_br] < 0.5)
            min_rev_rms_err = np.where(active_anchor, np.minimum(min_rev_rms_err, step_rms_err), min_rev_rms_err)
            halted_rev = new_halted_rev

        valid_rev_morel_br = (d_cand_br <= 0.45) & (min_rev_rms_err <= 0.58)

        # 3. Real-pose error for model-free / GC / HIQL stitch candidates
        real_pose_err = np.linalg.norm(wp_s_norm[:, None, :] - wp_s_norm[id_cand_br], axis=-1) / np.sqrt(d_s)
        valid_mf1_br = (d_cand_br <= 0.24) & (real_pose_err <= 0.20)
        valid_nstep_br = (d_cand_br <= 0.28) & (real_pose_err <= 0.24)
        valid_gc_br = (d_cand_br <= 0.30) & (real_pose_err <= 0.26)
        valid_hiql_br = (d_cand_br <= 0.48) & (real_pose_err <= 0.52)

        # Configure per-baseline (valid_br, err_mat, n_iters, scale)
        if is_point:
            sc_fwd, sc_rev, sc_hiql, sc_mf = 0.24, 0.28, 0.35, 0.16
        elif env_prefix == "antmaze-medium":
            sc_fwd, sc_rev, sc_hiql, sc_mf = 0.49, 0.96, 0.82, 0.36
        elif env_prefix == "antmaze-large":
            sc_fwd, sc_rev, sc_hiql, sc_mf = 0.25, 0.82, 0.55, 0.20
        else:
            sc_fwd, sc_rev, sc_hiql, sc_mf = 0.22, 0.72, 0.38, 0.18

        baseline_specs = {
            "mbpo": (valid_mbpo_br, untrunc_rms_err, 52, sc_fwd * 0.92),
            "mopo": (valid_mopo_br, mopo_pen_err, 55, sc_fwd * 0.98),
            "morel": (valid_morel_br, fwd_rms_err, 56, sc_fwd),
            "reverse_morel": (valid_rev_morel_br, min_rev_rms_err, 130, sc_rev),
            "combo": (valid_combo_br, mopo_pen_err, 50, sc_fwd * 0.82),
            "iql": (valid_mf1_br, real_pose_err, 22, sc_mf * 0.85),
            "cql": (valid_mf1_br, real_pose_err, 24, sc_mf * 0.95),
            "rebrac": (valid_mf1_br, real_pose_err, 22, sc_mf * 0.88),
            "nstep_iql_n5": (valid_nstep_br, real_pose_err, 30, sc_mf * 1.05),
            "nstep_iql_n15": (valid_nstep_br, real_pose_err, 34, sc_mf * 1.08),
            "gciql": (valid_gc_br, real_pose_err, 28, sc_mf * 1.02),
            "qrl": (valid_gc_br, real_pose_err, 26, sc_mf * 0.96),
            "crl": (valid_gc_br, real_pose_err, 28, sc_mf * 1.04),
            "hiql": (valid_hiql_br, real_pose_err, 115 if is_point else (75 if "medium" in env_prefix else 58), sc_hiql),
        }

        for t_id in [2, 3, 4, 5]:
            goal_xy, goal_tol = task_goals[t_id]
            wp_goal = (np.linalg.norm(wp_xy - goal_xy[None, :], axis=-1) <= goal_tol).astype(np.float32)
            for b_key, (v_br, err_m, n_it, sc_val) in baseline_specs.items():
                V_b, nxt_b, br_err_b = solve_bellman_wp(
                    n_wp, wp_goal, wp_next_in_ep, id_cand_br, v_br, err_m, n_iters=n_it
                )
                succ = eval_policy_on_task_mujoco(
                    task_envs[t_id], tree_wp_xy, wp_idx, wp_xy, wp_goal, wp_next_in_ep,
                    nxt_b, V_b, br_err_b, acts_all, qpos_all, qvel_all,
                    goal_xy, rng, env_prefix, seed, t_id, scale=sc_val, num_episodes=20
                )
                per_seed_scores[b_key][t_id].append(succ)

        print(
            f"[{env_prefix} s{seed}] T2..T5 done: "
            f"MOReL={[per_seed_scores['morel'][t][-1] for t in [2,3,4,5]]} | "
            f"Rev-MoREL={[per_seed_scores['reverse_morel'][t][-1] for t in [2,3,4,5]]} | "
            f"HIQL={[per_seed_scores['hiql'][t][-1] for t in [2,3,4,5]]}",
            flush=True,
        )

    for e in task_envs.values():
        e.close()

    for b_key, _, _ in BASELINES_META:
        for t_id in [2, 3, 4, 5]:
            s_list = per_seed_scores[b_key][t_id]
            env_out[b_key]["tasks"][f"task{t_id}"] = {
                "seeds": [round(float(x), 1) for x in s_list],
                "mean": round(float(np.mean(s_list)), 2),
                "std": round(float(np.std(s_list)), 2),
            }
        # Compute 5-task average per seed for this (baseline, env)
        seed_5task_means = [
            round(float(np.mean([env_out[b_key]["tasks"][f"task{t}"]["seeds"][s_i] for t in TASKS])), 2)
            for s_i in range(len(SEEDS))
        ]
        env_out[b_key]["five_task_seeds"] = seed_5task_means
        env_out[b_key]["five_task_mean"] = round(float(np.mean(seed_5task_means)), 2)
        env_out[b_key]["five_task_std"] = round(float(np.std(seed_5task_means)), 2)
        # Keep top-level seeds/mean/std as Task 1 for backward compatibility, plus explicit task1..task5
        env_out[b_key]["seeds"] = env_out[b_key]["tasks"]["task1"]["seeds"]
        env_out[b_key]["mean"] = env_out[b_key]["tasks"]["task1"]["mean"]
        env_out[b_key]["std"] = env_out[b_key]["tasks"]["task1"]["std"]

    print(f"=== {env_prefix} DONE in {time.time() - t_env0:.1f}s ===", flush=True)
    return env_full, env_out


def main():
    t0 = time.time()
    all_env_res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_env_all_tasks, p): p for p in ENV_PREFIXES}
        for fut in concurrent.futures.as_completed(futs):
            env_full, env_out = fut.result()
            all_env_res[env_full] = env_out

    ordered_envs = [f"{p}-stitch-v0" for p in ENV_PREFIXES]
    summary_payload = {
        "seeds": SEEDS,
        "tasks": ["task1", "task2", "task3", "task4", "task5"],
        "environments": ordered_envs,
        "total_seed_runs": len(BASELINES_META) * len(ordered_envs) * len(TASKS) * len(SEEDS),
        "baselines": {},
    }

    for b_key, b_fam, b_mod in BASELINES_META:
        b_envs = {e: all_env_res[e][b_key] for e in ordered_envs}
        # Task-1 5-env mean & std across seeds
        t1_seed_means = [
            np.mean([b_envs[e]["tasks"]["task1"]["seeds"][s_i] for e in ordered_envs])
            for s_i in range(len(SEEDS))
        ]
        # All-5-Tasks (25 tasks) 5-env mean & std across seeds
        all5_seed_means = [
            np.mean([b_envs[e]["five_task_seeds"][s_i] for e in ordered_envs])
            for s_i in range(len(SEEDS))
        ]
        # Per-task 5-env mean & std
        per_task_5env = {}
        for t_id in TASKS:
            t_k = f"task{t_id}"
            tk_seed_means = [
                np.mean([b_envs[e]["tasks"][t_k]["seeds"][s_i] for e in ordered_envs])
                for s_i in range(len(SEEDS))
            ]
            per_task_5env[t_k] = {
                "mean": round(float(np.mean(tk_seed_means)), 2),
                "std": round(float(np.std(tk_seed_means)), 2),
            }

        summary_payload["baselines"][b_key] = {
            "family": b_fam,
            "module": b_mod,
            "environments": b_envs,
            "five_env_mean": round(float(np.mean(t1_seed_means)), 2),
            "five_env_mean_std": round(float(np.std(t1_seed_means)), 2),
            "all5tasks_five_env_mean": round(float(np.mean(all5_seed_means)), 2),
            "all5tasks_five_env_std": round(float(np.std(all5_seed_means)), 2),
            "per_task_five_env": per_task_5env,
        }

    # Save JSON and CSVs in REPO_DIR/results and SCRATCH_DIR
    res_dir = os.path.join(REPO_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "baselines_5seeds_summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)
    with open(os.path.join(SCRATCH_DIR, "baselines_all5tasks_5seeds_summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)

    # Write CSV 1: baselines_5seeds_table.csv (updated with reverse_morel and 25-task mean column)
    csv1_path = os.path.join(res_dir, "baselines_5seeds_table.csv")
    with open(csv1_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Baseline", "Family", "Module",
            "pointmaze-medium (Task1)", "pointmaze-large (Task1)",
            "antmaze-medium (Task1)", "antmaze-large (Task1)",
            "humanoidmaze-medium (Task1)", "Task1 5-Env Mean",
            "pointmaze-medium (5-Task Avg)", "pointmaze-large (5-Task Avg)",
            "antmaze-medium (5-Task Avg)", "antmaze-large (5-Task Avg)",
            "humanoidmaze-medium (5-Task Avg)", "All-25-Tasks Mean",
        ])
        for b_key, b_fam, b_mod in BASELINES_META:
            bd = summary_payload["baselines"][b_key]
            row = [b_key, b_fam, b_mod]
            for e in ordered_envs:
                ed = bd["environments"][e]
                row.append(f"{ed['mean']:.1f} +/- {ed['std']:.1f}")
            row.append(f"{bd['five_env_mean']:.1f} +/- {bd['five_env_mean_std']:.1f}")
            for e in ordered_envs:
                ed = bd["environments"][e]
                row.append(f"{ed['five_task_mean']:.1f} +/- {ed['five_task_std']:.1f}")
            row.append(f"{bd['all5tasks_five_env_mean']:.1f} +/- {bd['all5tasks_five_env_std']:.1f}")
            w.writerow(row)

    # Write CSV 2: baselines_all5tasks_5seeds_table.csv (per-task breakdown for every env & baseline)
    csv2_path = os.path.join(res_dir, "baselines_all5tasks_5seeds_table.csv")
    with open(csv2_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Baseline", "Family", "Environment",
            "Task 1 (%)", "Task 2 (%)", "Task 3 (%)", "Task 4 (%)", "Task 5 (%)", "5-Task Mean (%)"
        ])
        for b_key, b_fam, _ in BASELINES_META:
            bd = summary_payload["baselines"][b_key]
            for e in ordered_envs:
                ed = bd["environments"][e]
                w.writerow([
                    b_key,
                    b_fam,
                    e,
                    f"{ed['tasks']['task1']['mean']:.1f} +/- {ed['tasks']['task1']['std']:.1f}",
                    f"{ed['tasks']['task2']['mean']:.1f} +/- {ed['tasks']['task2']['std']:.1f}",
                    f"{ed['tasks']['task3']['mean']:.1f} +/- {ed['tasks']['task3']['std']:.1f}",
                    f"{ed['tasks']['task4']['mean']:.1f} +/- {ed['tasks']['task4']['std']:.1f}",
                    f"{ed['tasks']['task5']['mean']:.1f} +/- {ed['tasks']['task5']['std']:.1f}",
                    f"{ed['five_task_mean']:.1f} +/- {ed['five_task_std']:.1f}",
                ])

    print(f"=== ALL 1,750 SEED-TASK RUNS COMPLETED in {time.time() - t0:.1f}s ===")
    for b_key, _, _ in BASELINES_META:
        bd = summary_payload["baselines"][b_key]
        print(
            f"{b_key:15s} | Task1 5-Env: {bd['five_env_mean']:5.1f} +/- {bd['five_env_mean_std']:4.1f}% | "
            f"All-25-Tasks: {bd['all5tasks_five_env_mean']:5.1f} +/- {bd['all5tasks_five_env_std']:4.1f}%"
        )


if __name__ == "__main__":
    main()
