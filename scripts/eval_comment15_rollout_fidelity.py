import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/home/rchitale_google_com/mborl")
import ogbench
from agents import agents
from utils.datasets import normalize
from utils.flax_utils import restore_agent
from agents.combo import get_config


def restore_sim_state(env, qpos, qvel, button_states):
    u = env.unwrapped
    u.set_state(qpos, qvel, button_states)
    u._prev_qpos = qpos.copy()
    u._prev_qvel = qvel.copy()
    u._prev_button_states = button_states.copy()
    u._prev_ob_info = u.compute_ob_info()
    cube_s, btn_s, drw_s, win_s = u._compute_successes()
    u._success = all(cube_s) and all(btn_s) and drw_s and win_s
    ob0 = u.compute_observation()
    rew0 = u.compute_reward()
    return ob0, float(rew0), float(u._success)


def evaluate_checkpoint(
    env,
    scale_obs,
    agent,
    init_states,
    horizon=10,
    num_repeats=5,
    rng_seed=42,
):
    N = len(init_states["ob0"])
    R = num_repeats
    B = N * R

    ob0_raw = np.repeat(init_states["ob0"], R, axis=0)
    ob0_norm = normalize(scale_obs, ob0_raw)

    @jax.jit
    def rollout_k(init_ob_norm, rng):
        def step_fn(carry, _):
            ob, rng = carry
            rng, act_rng, dyn_rng = jax.random.split(rng, 3)
            action, _ = agent.sample_actions(ob, None, act_rng, mode="train")
            next_ob = agent.sample_dynamics(ob, action, dyn_rng)
            succ_logit = agent.model.select("success")(ob, None, action)
            succ_prob = jax.nn.sigmoid(succ_logit)
            succ_pred = (succ_prob > 0.5).astype(jnp.float32)
            rew_pred = agent.model.select("reward")(ob, None, action)
            return (next_ob, rng), {
                "ob_norm": ob,
                "action": action,
                "next_ob_norm": next_ob,
                "rew_pred": rew_pred,
                "succ_prob": succ_prob,
                "succ_pred": succ_pred,
            }

        (final_ob, rng), scan_out = jax.lax.scan(
            step_fn, (jnp.asarray(init_ob_norm), rng), None, length=horizon
        )
        _, act_rng = jax.random.split(rng)
        final_act, _ = agent.sample_actions(final_ob, None, act_rng, mode="train")
        final_succ_logit = agent.model.select("success")(final_ob, None, final_act)
        final_succ_prob = jax.nn.sigmoid(final_succ_logit)
        final_succ_pred = (final_succ_prob > 0.5).astype(jnp.float32)
        final_rew_pred = agent.model.select("reward")(final_ob, None, final_act)
        return scan_out, {
            "ob_norm": final_ob,
            "action": final_act,
            "rew_pred": final_rew_pred,
            "succ_prob": final_succ_prob,
            "succ_pred": final_succ_pred,
        }

    scan_out, final_out = rollout_k(ob0_norm, jax.random.PRNGKey(rng_seed))
    s_hat_norm = np.concatenate(
        [np.array(scan_out["ob_norm"]), np.array(final_out["ob_norm"])[None]], axis=0
    )
    actions_all = np.concatenate(
        [np.array(scan_out["action"]), np.array(final_out["action"])[None]], axis=0
    )
    r_hat_traj = np.concatenate(
        [np.array(scan_out["rew_pred"]), np.array(final_out["rew_pred"])[None]], axis=0
    )
    succ_hat_traj = np.concatenate(
        [np.array(scan_out["succ_pred"]), np.array(final_out["succ_pred"])[None]], axis=0
    )

    s_env_raw = np.zeros((horizon + 1, B, 40), dtype=np.float32)
    r_env_true = np.zeros((horizon + 1, B), dtype=np.float32)
    succ_env_true = np.zeros((horizon + 1, B), dtype=np.float32)

    u = env.unwrapped
    for b in range(B):
        i_state = b // R
        ob0, rew0, succ0 = restore_sim_state(
            env,
            init_states["qpos"][i_state],
            init_states["qvel"][i_state],
            init_states["button_states"][i_state],
        )
        s_env_raw[0, b] = ob0
        r_env_true[0, b] = rew0
        succ_env_true[0, b] = succ0
        for k in range(1, horizon + 1):
            ob_k, _, _, _, _ = env.step(actions_all[k - 1, b])
            s_env_raw[k, b] = ob_k
            r_env_true[k, b] = float(u.compute_reward())
            succ_env_true[k, b] = float(u._success)

    s_env_norm = normalize(scale_obs, s_env_raw)

    @jax.jit
    def eval_heads_on_valid(obs_norm_flat, acts_flat):
        sl = agent.model.select("success")(obs_norm_flat, None, acts_flat)
        sp = jax.nn.sigmoid(sl)
        sd = (sp > 0.5).astype(jnp.float32)
        rw = agent.model.select("reward")(obs_norm_flat, None, acts_flat)
        return rw, sp, sd

    K1 = horizon + 1
    rw_head_valid, sp_head_valid, sd_head_valid = eval_heads_on_valid(
        jnp.asarray(s_env_norm.reshape(K1 * B, 40)),
        jnp.asarray(actions_all.reshape(K1 * B, -1)),
    )
    r_hat_valid = np.array(rw_head_valid).reshape(K1, B)
    succ_hat_valid = np.array(sd_head_valid).reshape(K1, B)

    sq_diff = np.square(s_hat_norm - s_env_norm).reshape(K1, N, R, 40)
    slices = {
        "total_l2_sq": (0, 40),
        "proprio_l2_sq": (0, 19),
        "cube_l2_sq": (19, 28),
        "buttons_l2_sq": (28, 36),
        "drawer_l2_sq": (36, 38),
        "window_l2_sq": (38, 40),
    }
    depth_rows = []
    for k in range(K1):
        row = {"depth": k}
        for name, (lo, hi) in slices.items():
            per_state = sq_diff[k, :, :, lo:hi].sum(axis=-1).mean(axis=-1)
            row[f"{name}_mean"] = float(per_state.mean())
            row[f"{name}_se"] = float(per_state.std(ddof=1) / np.sqrt(N))

        r_diff_traj = np.abs(r_hat_traj[k] - r_env_true[k]).reshape(N, R).mean(axis=-1)
        r_bias_traj = (r_hat_traj[k] - r_env_true[k]).reshape(N, R).mean(axis=-1)
        pred_succ_t = succ_hat_traj[k].reshape(N, R).mean(axis=-1)
        true_succ_t = succ_env_true[k].reshape(N, R).mean(axis=-1)
        fp_traj = ((succ_hat_traj[k] > 0.5) & (succ_env_true[k] < 0.5)).astype(np.float32).reshape(N, R).mean(axis=-1)
        fn_traj = ((succ_hat_traj[k] < 0.5) & (succ_env_true[k] > 0.5)).astype(np.float32).reshape(N, R).mean(axis=-1)

        r_diff_valid = np.abs(r_hat_valid[k] - r_env_true[k]).reshape(N, R).mean(axis=-1)
        r_bias_valid = (r_hat_valid[k] - r_env_true[k]).reshape(N, R).mean(axis=-1)
        pred_succ_v = succ_hat_valid[k].reshape(N, R).mean(axis=-1)
        fp_valid = ((succ_hat_valid[k] > 0.5) & (succ_env_true[k] < 0.5)).astype(np.float32).reshape(N, R).mean(axis=-1)
        fn_valid = ((succ_hat_valid[k] < 0.5) & (succ_env_true[k] > 0.5)).astype(np.float32).reshape(N, R).mean(axis=-1)

        for mname, arr in [
            ("traj_rew_mae", r_diff_traj),
            ("traj_rew_bias", r_bias_traj),
            ("traj_pred_succ", pred_succ_t),
            ("true_sim_succ", true_succ_t),
            ("traj_fp_succ", fp_traj),
            ("traj_fn_succ", fn_traj),
            ("valid_rew_mae", r_diff_valid),
            ("valid_rew_bias", r_bias_valid),
            ("valid_pred_succ", pred_succ_v),
            ("valid_fp_succ", fp_valid),
            ("valid_fn_succ", fn_valid),
        ]:
            row[f"{mname}_mean"] = float(arr.mean())
            row[f"{mname}_se"] = float(arr.std(ddof=1) / np.sqrt(N))
        depth_rows.append(row)

    return depth_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_states", type=int, default=200)
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--out_json", type=str, default="/home/rchitale_google_com/mborl/results/comment15/rollout_fidelity_initial.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    raw_val = np.load("/home/rchitale_google_com/ogbench/prepared/scene-play-v0-val.npz")
    env = ogbench.make_env_and_datasets("scene-play-singletask-task1-v0", env_only=True)
    env.reset()

    rng_np = np.random.RandomState(12345)
    ep_ids = rng_np.choice(100, size=args.num_states, replace=True)
    t_offsets = rng_np.randint(0, 980, size=args.num_states)
    raw_idxs = ep_ids * 1001 + t_offsets

    init_states = {
        "qpos": raw_val["qpos"][raw_idxs],
        "qvel": raw_val["qvel"][raw_idxs],
        "button_states": raw_val["button_states"][raw_idxs],
        "ob0": raw_val["observations"][raw_idxs].copy(),
    }

    max_restore_err = 0.0
    for i in range(args.num_states):
        ob0_sim, _, _ = restore_sim_state(
            env, init_states["qpos"][i], init_states["qvel"][i], init_states["button_states"][i]
        )
        err_non_contact = float(np.max(np.abs(np.delete(ob0_sim - init_states["ob0"][i], 18))))
        max_restore_err = max(max_restore_err, err_non_contact)
        init_states["ob0"][i] = ob0_sim
    print(f"Verified exact simulator restoration max error (39 pose/vel/button dims): {max_restore_err:.2e}", flush=True)

    e1_base = "/home/rchitale_google_com/mborl/exp/scalembrl/combo_comment14_exp1_scene-play-singletask-task1-v0"
    ckpt_specs = [
        {
            "label": "s0_ckpt500k_precollapse",
            "config": "online_H5_500k",
            "seed": 0,
            "step": 500000,
            "dir": "/home/rchitale_google_com/mborl/exp/scalembrl/comment14_ref/scene_task1_s0_ckpt500k",
            "scale_dir": f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0",
        },
        {
            "label": "exp14_e1_h5_100k_s0_1M",
            "config": "exp14_e1_h5_100k",
            "seed": 0,
            "step": 1000000,
            "dir": f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0",
            "scale_dir": f"{e1_base}/sd000_20261028_083047_exp14-e1-mb100k-h5-scene-task1-s0",
        },
        {
            "label": "exp14_e1_h5_100k_s1_1M",
            "config": "exp14_e1_h5_100k",
            "seed": 1,
            "step": 1000000,
            "dir": f"{e1_base}/sd001_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s1",
            "scale_dir": f"{e1_base}/sd001_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s1",
        },
        {
            "label": "exp14_e1_h5_100k_s2_1M",
            "config": "exp14_e1_h5_100k",
            "seed": 2,
            "step": 1000000,
            "dir": f"{e1_base}/sd002_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s2",
            "scale_dir": f"{e1_base}/sd002_20261028_083048_exp14-e1-mb100k-h5-scene-task1-s2",
        },
    ]

    cfg = get_config("ogbench")
    cfg["dataset_class"] = "CDataset"
    cfg["ob_dim"] = 40
    cfg["action_dim"] = 5

    dummy_agent = agents["combo"].create(
        0,
        {"observations": np.zeros((1, 40), dtype=np.float32), "actions": np.zeros((1, 5), dtype=np.float32)},
        cfg,
    )

    results = {
        "max_restore_err": max_restore_err,
        "num_states": args.num_states,
        "num_repeats": args.num_repeats,
        "horizon": args.horizon,
        "checkpoints": {},
    }

    for spec in ckpt_specs:
        with open(os.path.join(spec["scale_dir"], "scale.pkl"), "rb") as f:
            scale = pickle.load(f)
        agent = restore_agent(dummy_agent, spec["dir"], spec["step"])
        rows = evaluate_checkpoint(
            env=env,
            scale_obs=scale["observations"],
            agent=agent,
            init_states=init_states,
            horizon=args.horizon,
            num_repeats=args.num_repeats,
            rng_seed=100 + spec["seed"],
        )
        results["checkpoints"][spec["label"]] = {
            "meta": spec,
            "depths": rows,
        }
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

        e1 = rows[1]["total_l2_sq_mean"]
        e5 = rows[5]["total_l2_sq_mean"]
        e10 = rows[10]["total_l2_sq_mean"]
        lbl = spec["label"]
        p15 = np.mean([rows[k]["traj_pred_succ_mean"] for k in range(1, 6)]) * 100
        p610 = np.mean([rows[k]["traj_pred_succ_mean"] for k in range(6, 11)]) * 100
        t15 = np.mean([rows[k]["true_sim_succ_mean"] for k in range(1, 6)]) * 100
        t610 = np.mean([rows[k]["true_sim_succ_mean"] for k in range(6, 11)]) * 100
        fp15 = np.mean([rows[k]["traj_fp_succ_mean"] for k in range(1, 6)]) * 100
        fp610 = np.mean([rows[k]["traj_fp_succ_mean"] for k in range(6, 11)]) * 100
        vfp15 = np.mean([rows[k]["valid_fp_succ_mean"] for k in range(1, 6)]) * 100
        vfp610 = np.mean([rows[k]["valid_fp_succ_mean"] for k in range(6, 11)]) * 100
        print(
            f"[{lbl}] StateErr k=1:{e1:.4f} k=5:{e5:.4f} k=10:{e10:.4f} | "
            f"PredSucc 1-5:{p15:.2f}% 6-10:{p610:.2f}% | "
            f"TrueSimSucc 1-5:{t15:.2f}% 6-10:{t610:.2f}% | "
            f"TrajFP 1-5:{fp15:.2f}% 6-10:{fp610:.2f}% | "
            f"ValidHeadFP 1-5:{vfp15:.2f}% 6-10:{vfp610:.2f}%",
            flush=True,
        )

    print("Saved results to", args.out_json, flush=True)


if __name__ == "__main__":
    main()
