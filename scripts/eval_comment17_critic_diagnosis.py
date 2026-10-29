"""Comment 17 Evaluation-Only Diagnostic: Critic Correctness and Action-Ranking Diagnosis on Scene Task 1."""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/home/rchitale_google_com/mborl")


def restore_sim_state(env, qpos, qvel, button_states):
    u = env.unwrapped
    u.set_state(qpos, qvel, button_states)
    u._prev_qpos = qpos.copy()
    u._prev_qvel = qvel.copy()
    u._prev_button_states = button_states.copy()
    u._prev_ob_info = u.compute_ob_info()
    cube_s, btn_s, drw_s, win_s = u._compute_successes()
    u._success = bool(all(cube_s) and all(btn_s) and drw_s and win_s)
    ob0 = u.compute_observation()
    rew0 = float(u.compute_reward())
    return ob0, rew0, float(u._success)


# Global worker state for CPU MuJoCo workers
_WORKER_ENV = None


def _init_cpu_worker():
    global _WORKER_ENV
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if "MUJOCO_GL" in os.environ:
        del os.environ["MUJOCO_GL"]
    import warnings
    warnings.filterwarnings("ignore")
    import ogbench
    _WORKER_ENV = ogbench.make_env_and_datasets("scene-play-singletask-task1-v0", env_only=True)
    _WORKER_ENV.reset()


def _step_batch_worker(tasks):
    """Step a list of (qpos, qvel, button_states, action) tuples one step in MuJoCo."""
    global _WORKER_ENV
    u = _WORKER_ENV.unwrapped
    out = []
    for qpos, qvel, btn, act in tasks:
        ob0, rew0, succ0 = restore_sim_state(_WORKER_ENV, qpos, qvel, btn)
        if succ0 > 0.5:
            # Already terminal at current state s_k (m_k = 0)
            out.append((ob0, rew0, 1.0, qpos, qvel, btn, True))
        else:
            ob_next, _, _, _, _ = _WORKER_ENV.step(act)
            rew_next = float(u.compute_reward())
            succ_next = float(u._success)
            qpos_next = u._data.qpos.copy()
            qvel_next = u._data.qvel.copy()
            btn_next = u._cur_button_states.copy()
            out.append((ob_next, rew0, succ0, qpos_next, qvel_next, btn_next, False))
    return out


def build_shared_and_policy_states(out_state_file):
    """Build and verify the 50 shared validation states and 25 policy-relevant evaluation states."""
    if os.path.exists(out_state_file):
        with open(out_state_file, "rb") as f:
            return pickle.load(f)

    import jax
    import jax.numpy as jnp
    import ogbench
    from agents import agents
    from agents.combo import get_config
    from utils.datasets import normalize
    from utils.flax_utils import restore_agent

    raw_val = np.load("/home/rchitale_google_com/ogbench/prepared/scene-play-v0-val.npz")
    env = ogbench.make_env_and_datasets("scene-play-singletask-task1-v0", env_only=True)
    env.reset()
    u = env.unwrapped

    # 1. Shared validation states (first 50 of the 200 Comment-15 held-out validation states)
    rng_np = np.random.RandomState(12345)
    ep_ids = rng_np.choice(100, size=200, replace=True)
    t_offsets = rng_np.randint(0, 980, size=200)
    raw_idxs = ep_ids * 1001 + t_offsets
    shared_idxs = raw_idxs[:50]

    shared_states = {
        "state_ids": [f"val_ep{int(ep_ids[i]):02d}_t{int(t_offsets[i]):03d}" for i in range(50)],
        "raw_idxs": shared_idxs.tolist(),
        "qpos": raw_val["qpos"][shared_idxs].copy(),
        "qvel": raw_val["qvel"][shared_idxs].copy(),
        "button_states": raw_val["button_states"][shared_idxs].copy(),
        "ob0": np.zeros((50, 40), dtype=np.float32),
        "rew0": np.zeros(50, dtype=np.float32),
        "succ0": np.zeros(50, dtype=np.float32),
        "dataset_actions": np.clip(raw_val["actions"][shared_idxs].copy(), -1 + 1e-5, 1 - 1e-5),
    }

    max_obs_err = 0.0
    max_trans_err = 0.0
    for i in range(50):
        idx = shared_idxs[i]
        ob0_sim, rew0_sim, succ0_sim = restore_sim_state(
            env, shared_states["qpos"][i], shared_states["qvel"][i], shared_states["button_states"][i]
        )
        # Exclude dim 18 (discrete contact sensor flag) when checking pose/velocity/button restoration
        obs_err = float(np.max(np.abs(np.delete(ob0_sim - raw_val["observations"][idx], 18))))
        max_obs_err = max(max_obs_err, obs_err)
        shared_states["ob0"][i] = ob0_sim
        shared_states["rew0"][i] = rew0_sim
        shared_states["succ0"][i] = succ0_sim

        # Replay recorded dataset action and check next_observation reproduction
        ob1_sim, _, _, _, _ = env.step(raw_val["actions"][idx])
        trans_err = float(np.max(np.abs(np.delete(ob1_sim - raw_val["observations"][idx + 1], 18))))
        max_trans_err = max(max_trans_err, trans_err)

    # 2. Policy-relevant states: roll out saved policies on official task resets and save 25 intermediate states
    e1_base = "/home/rchitale_google_com/mborl/exp/scalembrl/combo_comment14_exp1_scene-play-singletask-task1-v0"
    s1_dir = f"{e1_base}/sd001_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s1"
    s0_500k_dir = "/home/rchitale_google_com/mborl/exp/scalembrl/comment14_ref/scene_task1_s0_ckpt500k"
    s0_1m_dir = f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0"

    with open(os.path.join(s1_dir, "scale.pkl"), "rb") as f:
        scale_obs = pickle.load(f)["observations"]

    cfg = get_config("ogbench")
    cfg["dataset_class"] = "CDataset"
    cfg["ob_dim"] = 40
    cfg["action_dim"] = 5
    dummy_agent = agents["combo"].create(
        0,
        {"observations": np.zeros((1, 40), dtype=np.float32), "actions": np.zeros((1, 5), dtype=np.float32)},
        cfg,
    )
    agent_s1_1m = restore_agent(dummy_agent, s1_dir, 1000000)
    agent_s0_500k = restore_agent(dummy_agent, s0_500k_dir, 500000)
    agent_s0_1m = restore_agent(dummy_agent, s0_1m_dir, 1000000)

    pol_qpos, pol_qvel, pol_btn, pol_ob0, pol_rew0, pol_succ0, pol_meta, pol_rec_act, pol_next_ob = (
        [], [], [], [], [], [], [], [], []
    )
    capture_steps = [20, 60, 120, 200, 300]
    sources = [
        ("exp1_s1_1M_stable", agent_s1_1m, 1000000, [0, 1]),
        ("s0_500k_precollapse", agent_s0_500k, 500000, [2, 3]),
        ("exp1_s0_1M_collapsed", agent_s0_1m, 1000000, [4]),
    ]
    for pol_name, ag, ckpt_step, ep_list in sources:
        for ep_idx in ep_list:
            ob, _ = env.reset(seed=5000 + ep_idx)
            for t in range(1, 305):
                ob_norm = normalize(scale_obs, ob[None])
                act, _ = ag.sample_actions(ob_norm, None, jax.random.PRNGKey(ep_idx * 1000 + t), temperature=0.0)
                act_np = np.clip(np.array(act[0]), -1.0, 1.0)
                if t in capture_steps:
                    qp = u._data.qpos.copy()
                    qv = u._data.qvel.copy()
                    bt = u._cur_button_states.copy()
                    ob_cur = u.compute_observation().copy()
                    rw_cur = float(u.compute_reward())
                    sc_cur = float(u._success)
                    ob_nxt, _, _, _, _ = env.step(act_np)
                    pol_qpos.append(qp)
                    pol_qvel.append(qv)
                    pol_btn.append(bt)
                    pol_ob0.append(ob_cur)
                    pol_rew0.append(rw_cur)
                    pol_succ0.append(sc_cur)
                    pol_rec_act.append(act_np)
                    pol_next_ob.append(ob_nxt.copy())
                    pol_meta.append({
                        "state_id": f"pol_{pol_name}_ep{ep_idx}_t{t:03d}",
                        "originating_policy": pol_name,
                        "checkpoint_step": ckpt_step,
                        "episode": ep_idx,
                        "timestep": t,
                    })
                    ob = ob_nxt
                else:
                    ob, _, _, _, _ = env.step(act_np)

    # Verify restoration and transition reproduction on all 25 policy-relevant states
    max_pol_obs_err = 0.0
    max_pol_trans_err = 0.0
    for i in range(len(pol_qpos)):
        ob0_sim, _, _ = restore_sim_state(env, pol_qpos[i], pol_qvel[i], pol_btn[i])
        obs_err = float(np.max(np.abs(np.delete(ob0_sim - pol_ob0[i], 18))))
        max_pol_obs_err = max(max_pol_obs_err, obs_err)
        ob1_sim, _, _, _, _ = env.step(pol_rec_act[i])
        trans_err = float(np.max(np.abs(np.delete(ob1_sim - pol_next_ob[i], 18))))
        max_pol_trans_err = max(max_pol_trans_err, trans_err)

    # Pre-compute reference actions on both state sets from s1_1M_stable and s0_500k_precollapse
    shared_ob_norm = normalize(scale_obs, shared_states["ob0"])
    pol_ob_norm = normalize(scale_obs, np.array(pol_ob0, dtype=np.float32))

    ref_s1_shared, _ = agent_s1_1m.sample_actions(shared_ob_norm, None, jax.random.PRNGKey(11), temperature=0.0)
    ref_500k_shared, _ = agent_s0_500k.sample_actions(shared_ob_norm, None, jax.random.PRNGKey(22), temperature=0.0)
    ref_s1_pol, _ = agent_s1_1m.sample_actions(pol_ob_norm, None, jax.random.PRNGKey(33), temperature=0.0)
    ref_500k_pol, _ = agent_s0_500k.sample_actions(pol_ob_norm, None, jax.random.PRNGKey(44), temperature=0.0)

    shared_states["ref_s1_1M_actions"] = np.clip(np.array(ref_s1_shared), -1.0, 1.0)
    shared_states["ref_s0_500k_actions"] = np.clip(np.array(ref_500k_shared), -1.0, 1.0)

    policy_states = {
        "state_ids": [m["state_id"] for m in pol_meta],
        "meta": pol_meta,
        "qpos": np.array(pol_qpos),
        "qvel": np.array(pol_qvel),
        "button_states": np.array(pol_btn),
        "ob0": np.array(pol_ob0, dtype=np.float32),
        "rew0": np.array(pol_rew0, dtype=np.float32),
        "succ0": np.array(pol_succ0, dtype=np.float32),
        "recorded_actions": np.array(pol_rec_act, dtype=np.float32),
        "ref_s1_1M_actions": np.clip(np.array(ref_s1_pol), -1.0, 1.0),
        "ref_s0_500k_actions": np.clip(np.array(ref_500k_pol), -1.0, 1.0),
    }

    payload = {
        "restoration_verification": {
            "shared_max_obs_restore_err": max_obs_err,
            "shared_max_transition_repro_err": max_trans_err,
            "policy_max_obs_restore_err": max_pol_obs_err,
            "policy_max_transition_repro_err": max_pol_trans_err,
        },
        "shared_states": shared_states,
        "policy_states": policy_states,
    }
    os.makedirs(os.path.dirname(out_state_file), exist_ok=True)
    with open(out_state_file, "wb") as f:
        pickle.dump(payload, f)
    print("Saved verified state sets to", out_state_file, payload["restoration_verification"], flush=True)
    return payload


def evaluate_single_checkpoint(spec, state_payload, horizon_sim=300, num_repeats=8, num_eval_eps=15, num_workers=16):
    """Evaluate one saved checkpoint on both state sets + real-env benchmark evaluation."""
    import jax
    import jax.numpy as jnp
    from agents import agents
    from agents.combo import get_config
    from utils.datasets import normalize
    from utils.flax_utils import restore_agent

    t_start = time.time()
    with open(os.path.join(spec["scale_dir"], "scale.pkl"), "rb") as f:
        scale_obs = pickle.load(f)["observations"]

    cfg = get_config("ogbench")
    cfg["dataset_class"] = "CDataset"
    cfg["ob_dim"] = 40
    cfg["action_dim"] = 5
    dummy_agent = agents["combo"].create(
        0,
        {"observations": np.zeros((1, 40), dtype=np.float32), "actions": np.zeros((1, 5), dtype=np.float32)},
        cfg,
    )
    agent = restore_agent(dummy_agent, spec["dir"], spec["step"])
    gamma = float(cfg["discount"])  # 0.999

    @jax.jit
    def eval_critic_and_actor(obs_norm, acts, rng):
        qs = agent.network.select("critic")(obs_norm, goals=None, actions=acts)
        q0 = qs[0]
        q1 = qs[1]
        q_mean = qs.mean(axis=0)
        q_min = qs.min(axis=0)
        dist = agent.network.select("actor")(obs_norm, goals=None)
        _, log_probs = dist.sample_and_log_prob(seed=rng)
        entropy = -log_probs.mean()
        std = dist._distribution.stddev().mean()
        lam = agent.network.select("lam")()
        return q0, q1, q_mean, q_min, lam, entropy, std

    @jax.jit
    def sample_stoch_actions(obs_norm, rng):
        acts, _ = agent.sample_actions(obs_norm, None, rng, temperature=1.0)
        return jnp.clip(acts, -1.0, 1.0)

    @jax.jit
    def sample_det_actions(obs_norm):
        acts, _ = agent.sample_actions(obs_norm, None, jax.random.PRNGKey(0), temperature=0.0)
        return jnp.clip(acts, -1.0, 1.0)

    pool = mp.Pool(processes=num_workers, initializer=_init_cpu_worker)

    def run_state_set(set_name, s_dict):
        N = len(s_dict["ob0"])
        ob0_norm = normalize(scale_obs, s_dict["ob0"])
        a_eval = np.array(sample_det_actions(jnp.asarray(ob0_norm)))
        a_stoch = np.array(sample_stoch_actions(jnp.asarray(ob0_norm), jax.random.PRNGKey(777 + spec["seed"])))

        # Candidate action set:
        # 0: a_eval (current deterministic action)
        # 1: a_stoch (current stochastic sample)
        # 2: a_ref_good (reference action from stable 1M policy, or 500k pre-collapse policy if self is s1_1M)
        # 3: a_alt (dataset action on shared set, or 500k/perturbation action on policy set)
        if spec["label"] == "exp14_e1_h5_100k_s1_1M":
            a_ref_good = s_dict["ref_s0_500k_actions"].copy()
            ref_good_name = "ref_s0_500k_precollapse"
        else:
            a_ref_good = s_dict["ref_s1_1M_actions"].copy()
            ref_good_name = "ref_s1_1M_stable"

        if set_name == "shared":
            a_alt = s_dict["dataset_actions"].copy()
            alt_name = "dataset_action"
        else:
            rng_pert = np.random.RandomState(999)
            delta = rng_pert.uniform(-0.25, 0.25, size=a_eval.shape).astype(np.float32)
            a_alt = np.clip(a_eval + delta, -1.0, 1.0)
            alt_name = "bounded_perturbation"

        cand_names = ["a_eval", "a_stoch", ref_good_name, alt_name]
        cand_actions = np.stack([a_eval, a_stoch, a_ref_good, a_alt], axis=1)  # (N, 4, 5)
        M = 4

        # Critic scores on (N, M)
        obs_rep = jnp.repeat(jnp.asarray(ob0_norm), M, axis=0)
        acts_flat = jnp.asarray(cand_actions.reshape(N * M, 5))
        q0, q1, q_mean, q_min, lam, ent, std = eval_critic_and_actor(
            obs_rep, acts_flat, jax.random.PRNGKey(123)
        )
        q0_np = np.array(q0).reshape(N, M)
        q1_np = np.array(q1).reshape(N, M)
        q_mean_np = np.array(q_mean).reshape(N, M)
        q_min_np = np.array(q_min).reshape(N, M)
        actor_stats = {
            "lam_alpha": float(lam),
            "policy_entropy": float(ent),
            "action_std": float(std),
        }

        # Roll out B = N * M * R trajectories for horizon_sim steps using current checkpoint's stochastic policy
        R = num_repeats
        B = N * M * R
        qpos_cur = np.repeat(s_dict["qpos"], M * R, axis=0)
        qvel_cur = np.repeat(s_dict["qvel"], M * R, axis=0)
        btn_cur = np.repeat(s_dict["button_states"], M * R, axis=0)
        ob_cur = np.repeat(s_dict["ob0"], M * R, axis=0)

        # First action is the candidate action a_0; subsequent actions k >= 1 sampled from pi(. | s_k)
        # Use matched random streams across candidate actions m=0..M-1!
        first_actions = np.repeat(cand_actions, R, axis=1).reshape(B, 5)
        returns = np.zeros(B, dtype=np.float64)
        alive = np.ones(B, dtype=np.float64)
        stopped_step = np.full(B, horizon_sim, dtype=np.int32)
        reached_goal = np.zeros(B, dtype=bool)

        for k in range(horizon_sim):
            active_idx = np.nonzero(alive > 0.5)[0]
            if len(active_idx) == 0:
                break
            if k == 0:
                acts_k = first_actions[active_idx]
            else:
                # Matched RNG per (state i, repeat r, step k) so all M candidates share the exact noise stream!
                ob_active_norm = normalize(scale_obs, ob_cur[active_idx])
                # Generate noise with matched seed per (r, k)
                # We call sample_stoch_actions per repeat r so candidate actions m=0..3 share identical PRNG keys
                acts_all = np.zeros((N, M, R, 5), dtype=np.float32)
                ob_all_norm = normalize(scale_obs, ob_cur.reshape(N, M, R, 40))
                for r_idx in range(R):
                    rng_rk = jax.random.PRNGKey(50000 + r_idx * 1000 + k)
                    ob_r = jnp.asarray(ob_all_norm[:, :, r_idx, :].reshape(N * M, 40))
                    # Split N keys (one per state i) and repeat across M candidates so all M candidates get identical key!
                    state_keys = jax.random.split(rng_rk, N)
                    matched_keys = jnp.repeat(state_keys, M, axis=0)
                    dist_r = agent.network.select("actor")(ob_r, goals=None, temperature=1.0)
                    # Sample using vmap over matched_keys
                    sampled_r = jax.vmap(lambda d_mean, d_std, key: jnp.tanh(d_mean + d_std * jax.random.normal(key, d_mean.shape)))(
                        dist_r._distribution.loc, dist_r._distribution.scale_diag, matched_keys
                    )
                    acts_all[:, :, r_idx, :] = np.clip(np.array(sampled_r).reshape(N, M, 5), -1.0, 1.0)
                acts_k = acts_all.reshape(B, 5)[active_idx]

            # Dispatch active_idx to CPU worker pool in chunks
            tasks = [
                (qpos_cur[idx], qvel_cur[idx], btn_cur[idx], acts_k[pos])
                for pos, idx in enumerate(active_idx)
            ]
            chunk_size = max(1, (len(tasks) + num_workers - 1) // num_workers)
            chunks = [tasks[c: c + chunk_size] for c in range(0, len(tasks), chunk_size)]
            chunk_results = pool.map(_step_batch_worker, chunks)
            flat_results = [item for sublist in chunk_results for item in sublist]

            discount_k = gamma ** k
            for pos, idx in enumerate(active_idx):
                ob_nxt, rew_k, succ_k, qp_nxt, qv_nxt, bt_nxt, term_now = flat_results[pos]
                returns[idx] += discount_k * alive[idx] * rew_k
                if succ_k > 0.5 or term_now:
                    alive[idx] = 0.0
                    stopped_step[idx] = k
                    reached_goal[idx] = True
                else:
                    ob_cur[idx] = ob_nxt
                    qpos_cur[idx] = qp_nxt
                    qvel_cur[idx] = qv_nxt
                    btn_cur[idx] = bt_nxt

        returns_nmr = returns.reshape(N, M, R)
        goal_nmr = reached_goal.reshape(N, M, R)
        stop_nmr = stopped_step.reshape(N, M, R)

        # Tail bounds for unterminated rollouts:
        # Over remaining episode steps (up to T_ep = 750) and infinite horizon:
        # Since r_k in [-6, 0], max additional return is 0, min additional return over (horizon_sim..750) is -6 * (gamma^T - gamma^750)/(1-gamma)
        tail_ep_min_factor = -6.0 * ((gamma ** horizon_sim) - (gamma ** 750)) / (1.0 - gamma)
        tail_inf_min_factor = -6.0 * (gamma ** horizon_sim) / (1.0 - gamma)
        unterm_nmr = (~goal_nmr).astype(np.float64)
        returns_ep_lower_nmr = returns_nmr + unterm_nmr * tail_ep_min_factor
        returns_inf_lower_nmr = returns_nmr + unterm_nmr * tail_inf_min_factor

        q_sim_mean = returns_nmr.mean(axis=-1)  # (N, M)
        q_sim_se = returns_nmr.std(axis=-1, ddof=1) / np.sqrt(R)
        q_sim_ep_low = returns_ep_lower_nmr.mean(axis=-1)
        q_sim_inf_low = returns_inf_lower_nmr.mean(axis=-1)
        goal_rate_nm = goal_nmr.mean(axis=-1)

        # Section 6A: Numerical Calibration (q_actor = q_mean vs q_sim_mean)
        err_nm = q_mean_np - q_sim_mean
        abs_err_nm = np.abs(err_nm)

        def summarize_cal(mask_cols):
            e = err_nm[:, mask_cols].reshape(-1)
            ae = abs_err_nm[:, mask_cols].reshape(-1)
            return {
                "signed_bias_mean": float(e.mean()),
                "signed_bias_se": float(e.std(ddof=1) / np.sqrt(len(e))),
                "mae_mean": float(ae.mean()),
                "mae_se": float(ae.std(ddof=1) / np.sqrt(len(ae))),
                "bias_q50": float(np.quantile(e, 0.50)),
                "bias_q90": float(np.quantile(e, 0.90)),
                "bias_q95": float(np.quantile(e, 0.95)),
                "frac_positive_bias": float(np.mean(e > 0)),
                "q_actor_mean": float(q_mean_np[:, mask_cols].mean()),
                "q_min_mean": float(q_min_np[:, mask_cols].mean()),
                "q_sim_trunc_mean": float(q_sim_mean[:, mask_cols].mean()),
                "q_sim_ep750_lower_mean": float(q_sim_ep_low[:, mask_cols].mean()),
                "q_sim_inf_lower_mean": float(q_sim_inf_low[:, mask_cols].mean()),
                "goal_termination_rate": float(goal_rate_nm[:, mask_cols].mean()),
            }

        calibration = {
            "current_actor_actions": summarize_cal([0, 1]),
            "alternative_actions": summarize_cal([2, 3]),
            "all_actions": summarize_cal([0, 1, 2, 3]),
        }

        # Section 6B: Action-Ranking Correctness (current a_eval col 0 vs reference cols 2 and 3)
        ranking_pairs = {}
        for ref_col, ref_lbl in [(2, ref_good_name), (3, alt_name), (1, "a_stoch")]:
            dq = q_mean_np[:, 0] - q_mean_np[:, ref_col]  # (N,)
            # Paired difference per state across R matched-seed continuations
            dsim_r = returns_nmr[:, 0, :] - returns_nmr[:, ref_col, :]  # (N, R)
            dsim = dsim_r.mean(axis=-1)
            dsim_se = dsim_r.std(axis=-1, ddof=1) / np.sqrt(R)
            # Inconclusive if |dsim| <= max(1.96 * dsim_se, 0.05)
            thresh = np.maximum(1.96 * dsim_se, 0.05)
            resolved = np.abs(dsim) > thresh
            wrong_opt = resolved & (dq > 0) & (dsim < 0)
            wrong_pess = resolved & (dq < 0) & (dsim > 0)
            any_wrong = wrong_opt | wrong_pess
            n_res = int(np.sum(resolved))
            ranking_pairs[ref_lbl] = {
                "num_states": N,
                "num_resolved": n_res,
                "frac_inconclusive": float(np.mean(~resolved)),
                "frac_confidently_wrong_optimistic_all": float(np.mean(wrong_opt)),
                "frac_confidently_wrong_optimistic_resolved": float(np.sum(wrong_opt) / max(1, n_res)),
                "frac_any_sign_error_all": float(np.mean(any_wrong)),
                "frac_any_sign_error_resolved": float(np.sum(any_wrong) / max(1, n_res)),
                "mean_dq": float(dq.mean()),
                "mean_dsim": float(dsim.mean()),
                "wrong_opt_mean_dq": float(dq[wrong_opt].mean()) if np.any(wrong_opt) else 0.0,
                "wrong_opt_mean_dsim": float(dsim[wrong_opt].mean()) if np.any(wrong_opt) else 0.0,
                "per_state_dq": dq.tolist(),
                "per_state_dsim": dsim.tolist(),
                "per_state_dsim_se": dsim_se.tolist(),
                "per_state_resolved": resolved.tolist(),
            }

        # Candidate-Set Regret: max_{a in A_test} Q_sim(s,a) - Q_sim(s, argmax_{a in A_test} Q_actor(s,a))
        chosen_idx = np.argmax(q_mean_np, axis=1)  # (N,)
        best_sim_val = np.max(q_sim_mean, axis=1)
        chosen_sim_val = q_sim_mean[np.arange(N), chosen_idx]
        regret_s = best_sim_val - chosen_sim_val

        # Bootstrap SE over independent starting states
        rng_boot = np.random.RandomState(42)
        boot_means = [
            float(regret_s[rng_boot.randint(0, N, size=N)].mean())
            for _ in range(500)
        ]
        regret_summary = {
            "mean_regret": float(regret_s.mean()),
            "bootstrap_se": float(np.std(boot_means, ddof=1)),
            "median_regret": float(np.median(regret_s)),
            "q90_regret": float(np.quantile(regret_s, 0.90)),
            "frac_zero_regret": float(np.mean(regret_s < 0.05)),
            "chosen_action_frequencies": {
                cand_names[m]: float(np.mean(chosen_idx == m)) for m in range(M)
            },
            "per_state_regret": regret_s.tolist(),
        }

        return {
            "candidate_names": cand_names,
            "actor_stats": actor_stats,
            "calibration": calibration,
            "ranking_pairs": ranking_pairs,
            "regret": regret_summary,
            "raw_arrays": {
                "q0": q0_np.tolist(),
                "q1": q1_np.tolist(),
                "q_mean": q_mean_np.tolist(),
                "q_min": q_min_np.tolist(),
                "q_sim_mean": q_sim_mean.tolist(),
                "q_sim_se": q_sim_se.tolist(),
                "goal_rate": goal_rate_nm.tolist(),
                "mean_stop_step": stop_nmr.mean(axis=-1).tolist(),
            },
        }

    shared_res = run_state_set("shared", state_payload["shared_states"])
    policy_res = run_state_set("policy", state_payload["policy_states"])

    # Section 6C: Re-evaluate checkpoint policy in real environment (deterministic temp=0.0 vs stochastic temp=1.0)
    import ogbench
    env_eval = ogbench.make_env_and_datasets("scene-play-singletask-task1-v0", env_only=True)
    u_eval = env_eval.unwrapped

    def run_policy_eval(temp, n_eps):
        succs, disc_rets, un_rets, lengths = [], [], [], []
        for ep in range(n_eps):
            ob, _ = env_eval.reset(seed=9000 + ep)
            done = False
            step = 0
            disc_ret = 0.0
            un_ret = 0.0
            alive_mask = 1.0
            ep_succ = 0.0
            while not done and step < 750:
                r_cur = float(u_eval.compute_reward())
                s_cur = float(u_eval._success)
                disc_ret += (gamma ** step) * alive_mask * r_cur
                un_ret += r_cur
                if s_cur > 0.5:
                    alive_mask = 0.0
                    ep_succ = 1.0
                ob_norm = normalize(scale_obs, ob[None])
                if temp == 0.0:
                    act = np.array(sample_det_actions(jnp.asarray(ob_norm)))[0]
                else:
                    act = np.array(
                        sample_stoch_actions(jnp.asarray(ob_norm), jax.random.PRNGKey(ep * 10000 + step))
                    )[0]
                ob, _, term, trunc, info = env_eval.step(act)
                if info.get("success", 0.0) > 0.5 or float(u_eval._success) > 0.5:
                    ep_succ = 1.0
                done = term or trunc
                step += 1
            succs.append(ep_succ)
            disc_rets.append(disc_ret)
            un_rets.append(un_ret)
            lengths.append(step)
        return {
            "success_rate": float(np.mean(succs)),
            "success_se": float(np.std(succs, ddof=1) / np.sqrt(n_eps)),
            "discounted_return_mean": float(np.mean(disc_rets)),
            "discounted_return_se": float(np.std(disc_rets, ddof=1) / np.sqrt(n_eps)),
            "undiscounted_return_mean": float(np.mean(un_rets)),
        }

    det_eval = run_policy_eval(0.0, num_eval_eps)
    stoch_eval = run_policy_eval(1.0, num_eval_eps)
    pool.close()
    pool.join()

    elapsed = time.time() - t_start
    print(
        f"[{spec['label']}] Done in {elapsed:.1f}s | DetSucc:{det_eval['success_rate']*100:.1f}% (Ret:{det_eval['discounted_return_mean']:.1f}) | "
        f"StochSucc:{stoch_eval['success_rate']*100:.1f}% (Ret:{stoch_eval['discounted_return_mean']:.1f}) | "
        f"Shared WrongOpt:{shared_res['ranking_pairs'][shared_res['candidate_names'][2]]['frac_confidently_wrong_optimistic_resolved']*100:.1f}% | "
        f"Shared Regret:{shared_res['regret']['mean_regret']:.2f}",
        flush=True,
    )
    return {
        "spec": spec,
        "elapsed_sec": elapsed,
        "deterministic_eval": det_eval,
        "stochastic_eval": stoch_eval,
        "shared_set": shared_res,
        "policy_set": policy_res,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_index", type=int, default=-1, help="If >=0, run only that checkpoint index")
    parser.add_argument("--horizon_sim", type=int, default=300)
    parser.add_argument("--num_repeats", type=int, default=8)
    parser.add_argument("--num_eval_eps", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="/home/rchitale_google_com/mborl/results/comment17")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state_file = os.path.join(args.out_dir, "verified_state_sets.pkl")
    state_payload = build_shared_and_policy_states(state_file)

    e1_base = "/home/rchitale_google_com/mborl/exp/scalembrl/combo_comment14_exp1_scene-play-singletask-task1-v0"
    e2_base = "/home/rchitale_google_com/mborl/exp/scalembrl/combo_comment14_exp2_scene-play-singletask-task1-v0"
    ckpt_specs = [
        {
            "label": "s0_500k_precollapse_b0.5_h10",
            "config_group": "Phase 2 Reference (beta=0.5, f=0.5, H=10, 50k starts)",
            "beta": 0.5,
            "real_ratio": 0.5,
            "horizon": 10,
            "mb_batch_size": 50000,
            "seed": 0,
            "step": 500000,
            "logged_eval_success": 0.5333,
            "status_note": "Saved at 500k (53.3% eval) before collapsing to 0.0% at 1M",
            "dir": "/home/rchitale_google_com/mborl/exp/scalembrl/comment14_ref/scene_task1_s0_ckpt500k",
            "scale_dir": "/home/rchitale_google_com/mborl/exp/scalembrl/comment14_ref/scene_task1_s0_ckpt500k",
        },
        {
            "label": "exp14_e1_h5_100k_s0_1M",
            "config_group": "Experiment 1 COMBO (beta=1.0, f=0.5, H=5, 100k starts)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 5,
            "mb_batch_size": 100000,
            "seed": 0,
            "step": 1000000,
            "logged_eval_success": 0.0,
            "status_note": "Post-collapse at 1M (0.0% eval after peaking at 86.7% at 500k)",
            "dir": f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0",
            "scale_dir": f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0",
        },
        {
            "label": "exp14_e1_h5_100k_s1_1M",
            "config_group": "Experiment 1 COMBO (beta=1.0, f=0.5, H=5, 100k starts)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 5,
            "mb_batch_size": 100000,
            "seed": 1,
            "step": 1000000,
            "logged_eval_success": 0.60,
            "status_note": "Stable high-performing at 1M (60.0% eval)",
            "dir": f"{e1_base}/sd001_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s1",
            "scale_dir": f"{e1_base}/sd001_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s1",
        },
        {
            "label": "exp14_e1_h5_100k_s2_1M",
            "config_group": "Experiment 1 COMBO (beta=1.0, f=0.5, H=5, 100k starts)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 5,
            "mb_batch_size": 100000,
            "seed": 2,
            "step": 1000000,
            "logged_eval_success": 0.1333,
            "status_note": "Moderate / partially recovered at 1M (13.3% eval)",
            "dir": f"{e1_base}/sd002_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s2",
            "scale_dir": f"{e1_base}/sd002_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s2",
        },
        {
            "label": "exp14_e2_d1to5_s0_1M",
            "config_group": "Experiment 2A COMBO (beta=1.0, f=0.5, Frozen Depths 1-5)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 5,
            "mb_batch_size": 50000,
            "seed": 0,
            "step": 1000000,
            "logged_eval_success": 0.0,
            "status_note": "Trained on early rollout depths 1-5 only (0.0% at 1M, 33.3% at 500k)",
            "dir": f"{e2_base}/sd000_20261028_083048_exp14-e2-frozen-d1to5-scene-task1-s0",
            "scale_dir": f"{e2_base}/sd000_20261028_083048_exp14-e2-frozen-d1to5-scene-task1-s0",
        },
        {
            "label": "exp14_e2_d6to10_s0_1M",
            "config_group": "Experiment 2B COMBO (beta=1.0, f=0.5, Frozen Depths 6-10)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 10,
            "mb_batch_size": 50000,
            "seed": 0,
            "step": 1000000,
            "logged_eval_success": 0.0,
            "status_note": "Trained on later rollout depths 6-10 only (26.7% at 900k, 0.0% at 1M)",
            "dir": f"{e2_base}/sd000_20261028_083048_exp14-e2-frozen-d6to10-scene-task1-s0",
            "scale_dir": f"{e2_base}/sd000_20261028_083048_exp14-e2-frozen-d6to10-scene-task1-s0",
        },
        {
            "label": "exp14_e2_d1to5_s1_1M",
            "config_group": "Experiment 2A COMBO (beta=1.0, f=0.5, Frozen Depths 1-5)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 5,
            "mb_batch_size": 50000,
            "seed": 1,
            "step": 1000000,
            "logged_eval_success": 0.0667,
            "status_note": "Trained on early rollout depths 1-5 only (6.7% at 1M)",
            "dir": f"{e2_base}/sd001_20261028_083048_exp14-e2-frozen-d1to5-scene-task1-s1",
            "scale_dir": f"{e2_base}/sd001_20261028_083048_exp14-e2-frozen-d1to5-scene-task1-s1",
        },
        {
            "label": "exp14_e2_d6to10_s1_1M",
            "config_group": "Experiment 2B COMBO (beta=1.0, f=0.5, Frozen Depths 6-10)",
            "beta": 1.0,
            "real_ratio": 0.5,
            "horizon": 10,
            "mb_batch_size": 50000,
            "seed": 1,
            "step": 1000000,
            "logged_eval_success": 0.0667,
            "status_note": "Trained on later rollout depths 6-10 only (20.0% at 700k, 6.7% at 1M)",
            "dir": f"{e2_base}/sd001_20261028_083048_exp14-e2-frozen-d6to10-scene-task1-s1",
            "scale_dir": f"{e2_base}/sd001_20261028_083048_exp14-e2-frozen-d6to10-scene-task1-s1",
        },
    ]

    if args.ckpt_index >= 0:
        spec = ckpt_specs[args.ckpt_index]
        res = evaluate_single_checkpoint(
            spec,
            state_payload,
            horizon_sim=args.horizon_sim,
            num_repeats=args.num_repeats,
            num_eval_eps=args.num_eval_eps,
            num_workers=args.num_workers,
        )
        part_file = os.path.join(args.out_dir, f"ckpt_{args.ckpt_index:02d}_{spec['label']}.json")
        with open(part_file, "w") as f:
            json.dump(res, f, indent=2)
        return

    # First ensure state_payload is created in main process before spawning 8 GPU workers
    procs = []
    for idx in range(len(ckpt_specs)):
        env_p = os.environ.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = str(idx)
        env_p["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if "MUJOCO_GL" in env_p:
            del env_p["MUJOCO_GL"]
        cmd = [
            sys.executable,
            __file__,
            f"--ckpt_index={idx}",
            f"--horizon_sim={args.horizon_sim}",
            f"--num_repeats={args.num_repeats}",
            f"--num_eval_eps={args.num_eval_eps}",
            f"--num_workers={args.num_workers}",
            f"--out_dir={args.out_dir}",
        ]
        procs.append(subprocess.Popen(cmd, env=env_p))

    for p in procs:
        p.wait()

    # Aggregate all checkpoint results + build full Checkpoint Manifest & Logged Critic Series
    combined = {
        "restoration_verification": state_payload["restoration_verification"],
        "horizon_sim": args.horizon_sim,
        "num_repeats": args.num_repeats,
        "tail_bounds_spec": {
            "gamma": 0.999,
            "reward_range": [-6.0, 0.0],
            "unterminated_tail_ep750_lower_offset": float(-6.0 * ((0.999 ** args.horizon_sim) - (0.999 ** 750)) / 0.001),
            "unterminated_tail_inf_lower_offset": float(-6.0 * (0.999 ** args.horizon_sim) / 0.001),
            "unterminated_tail_upper_offset": 0.0,
        },
        "checkpoints": {},
    }
    for idx, spec in enumerate(ckpt_specs):
        part_file = os.path.join(args.out_dir, f"ckpt_{idx:02d}_{spec['label']}.json")
        with open(part_file) as f:
            combined["checkpoints"][spec["label"]] = json.load(f)

    out_json = os.path.join(args.out_dir, "comment17_critic_diagnosis.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print("All 8 checkpoints evaluated and aggregated into", out_json, flush=True)


if __name__ == "__main__":
    main()
