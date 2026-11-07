import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/home/rchitale_google_com/mborl")
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import ogbench
from utils.datasets import normalize

ENV_CONFIGS = {
    "pointmaze-medium": {
        "dname": "pointmaze-medium-stitch-v0",
        "ename": "pointmaze-medium-stitch-singletask-task1-v0",
        "goal_xy": [20.0, 20.0],
        "max_ep_steps": 1000,
        "lookahead_h": 15,
        "maze_scale": 1.0,
    },
    "pointmaze-large": {
        "dname": "pointmaze-large-stitch-v0",
        "ename": "pointmaze-large-stitch-singletask-task1-v0",
        "goal_xy": [36.0, 24.0],
        "max_ep_steps": 1000,
        "lookahead_h": 15,
        "maze_scale": 0.68,
    },
    "antmaze-medium": {
        "dname": "antmaze-medium-stitch-v0",
        "ename": "antmaze-medium-stitch-singletask-task1-v0",
        "goal_xy": [20.0, 20.0],
        "max_ep_steps": 1000,
        "lookahead_h": 15,
        "maze_scale": 1.0,
    },
    "antmaze-large": {
        "dname": "antmaze-large-stitch-v0",
        "ename": "antmaze-large-stitch-singletask-task1-v0",
        "goal_xy": [36.0, 24.0],
        "max_ep_steps": 1000,
        "lookahead_h": 15,
        "maze_scale": 0.68,
    },
    "humanoidmaze-medium": {
        "dname": "humanoidmaze-medium-stitch-v0",
        "ename": "humanoidmaze-medium-stitch-singletask-task1-v0",
        "goal_xy": [20.0, 20.0],
        "max_ep_steps": 2000,
        "lookahead_h": 30,
        "maze_scale": 1.0,
    },
}

def main():
    key = sys.argv[1]
    cfg = ENV_CONFIGS[key]
    dname = cfg["dname"]
    ename = cfg["ename"]
    goal_xy = np.array(cfg["goal_xy"], dtype=np.float32)
    max_ep_steps = cfg["max_ep_steps"]
    H_look = cfg["lookahead_h"]
    ms = cfg["maze_scale"]

    t0 = time.time()
    ddir = f"/home/rchitale_google_com/ogbench/prepared/{dname}"
    env, train_ds, val_ds = ogbench.make_env_and_datasets(
        ename, dataset_path=f"{ddir}/{dname}.npz", compact_dataset=False
    )

    terms = train_ds["terminals"].astype(np.int32)
    ep_ids_all = np.concatenate([[0], np.cumsum(terms)[:-1]])
    valid = np.nonzero(terms == 0)[0]

    scale_ob = {
        "min": val_ds["observations"].min(axis=0),
        "max": val_ds["observations"].max(axis=0),
    }
    def norm_ob(x):
        return normalize(scale_ob, x)

    ob_dim = train_ds["observations"].shape[1]
    act_dim = train_ds["actions"].shape[1]
    eps = 1e-5

    s_np = np.asarray(norm_ob(train_ds["observations"][valid]), dtype=np.float32)
    a_np = np.asarray(np.clip(train_ds["actions"][valid], -1 + eps, 1 - eps), dtype=np.float32)
    ns_np = np.asarray(norm_ob(train_ds["next_observations"][valid]), dtype=np.float32)
    r_np = np.asarray(train_ds["rewards"][valid], dtype=np.float32)
    m_np = np.asarray(train_ds["masks"][valid], dtype=np.float32)
    ep_np = ep_ids_all[valid]

    disp_xy = np.linalg.norm(ns_np[:, :2] - s_np[:, :2], axis=-1)
    p90_step = float(np.percentile(disp_xy, 90))
    print(f"[{key}] Loaded valid={len(valid)}, ob_dim={ob_dim}, act_dim={act_dim}, p90_step={p90_step:.5f}", flush=True)

    def make_lookahead(k_steps):
        idx_k = np.minimum(np.arange(len(valid)) + k_steps, len(valid) - 1)
        same_ep = (ep_np[idx_k] == ep_np)
        return np.where(same_ep[:, None], s_np[idx_k], ns_np)

    s_h5_np = make_lookahead(5)
    s_h15_np = make_lookahead(H_look)

    gpu_real = {
        "s": jnp.asarray(s_np),
        "a": jnp.asarray(a_np),
        "ns": jnp.asarray(ns_np),
        "s_h5": jnp.asarray(s_h5_np),
        "s_h15": jnp.asarray(s_h15_np),
        "r": jnp.asarray(r_np),
        "m": jnp.asarray(m_np),
    }

    # 1. Train 5-member Ensemble Dynamics Model
    class EnsembleDyn(nn.Module):
        num_ens: int = 5
        out_dim: int = ob_dim
        @nn.compact
        def __call__(self, x):
            h = jnp.broadcast_to(x[None, ...], (self.num_ens,) + x.shape)
            for i, dim in enumerate((256, 256, 256)):
                w = self.param(f"w_{i}", nn.initializers.lecun_normal(), (self.num_ens, h.shape[-1], dim))
                b = self.param(f"b_{i}", nn.initializers.zeros, (self.num_ens, 1, dim))
                h = nn.relu(jnp.einsum("ebd,edh->ebh", h, w) + b)
            w_out = self.param("w_out", nn.initializers.lecun_normal(), (self.num_ens, 256, self.out_dim))
            b_out = self.param("b_out", nn.initializers.zeros, (self.num_ens, 1, self.out_dim))
            return jnp.einsum("ebd,edh->ebh", h, w_out) + b_out

    dyn_mod = EnsembleDyn(num_ens=5, out_dim=ob_dim)
    rng = jax.random.PRNGKey(0)
    rng, dk = jax.random.split(rng)
    dyn_params = dyn_mod.init(dk, jnp.concatenate([gpu_real["s"][:2], gpu_real["a"][:2]], axis=-1))["params"]
    dyn_st = TrainState.create(apply_fn=dyn_mod.apply, params=dyn_params, tx=optax.adam(1e-3))

    @jax.jit
    def train_dyn_500(st, r, ds):
        N = ds["s"].shape[0]
        def step(carry, _):
            s_st, k = carry
            k, ik = jax.random.split(k)
            idx = jax.random.randint(ik, (512,), 0, N)
            x = jnp.concatenate([ds["s"][idx], ds["a"][idx]], axis=-1)
            target = ds["ns"][idx] - ds["s"][idx]
            def loss_fn(p):
                pred = dyn_mod.apply({"params": p}, x)
                return jnp.square(pred - target[None, ...]).mean()
            loss, grads = jax.value_and_grad(loss_fn)(s_st.params)
            return (s_st.apply_gradients(grads=grads), k), loss
        (st, r), losses = jax.lax.scan(step, (st, r), None, length=500)
        return st, r, losses[-1]

    for _ in range(30):
        dyn_st, rng, dloss = train_dyn_500(dyn_st, rng, gpu_real)
    jax.block_until_ready(dloss)
    print(f"[{key}] Trained EnsembleDyn MSE={float(dloss):.6f}", flush=True)

    # 2. Build Reference Anchor Pool (K_ref = 4096) & Compute Horizon-Specific Value Landscapes
    np_rng = np.random.default_rng(42)
    goal_pool = np.nonzero(r_np == 0.0)[0]
    non_goal_pool = np.nonzero(r_np < 0.0)[0]
    n_g = min(256, len(goal_pool))
    ref_idx = np.concatenate([
        np_rng.choice(goal_pool, size=n_g, replace=False),
        np_rng.choice(non_goal_pool, size=4096 - n_g, replace=False),
    ])
    ref_s = gpu_real["s"][ref_idx]
    ref_ns = gpu_real["ns"][ref_idx]
    ref_h5 = gpu_real["s_h5"][ref_idx]
    ref_ep = jnp.asarray(ep_np[ref_idx])
    ref_r = gpu_real["r"][ref_idx]

    supp_thresh = 0.05 * ms

    @jax.jit
    def compute_graph_for_mode(rs, rns, rr, rep, r_max, check_manifold, cross_ep_only, same_ep_strict):
        d_ij = jnp.linalg.norm(rs[:, None, :2] - rs[None, :, :2], axis=-1)
        d_ns_j = jnp.linalg.norm(rns[:, None, :2] - rs[None, :, :2], axis=-1)

        # For H=0 (Model-Free), only connect states within the same episode or immediate 1-step TD smoothing radius
        cand_h0 = ((rep[:, None] == rep[None, :]) & (d_ij <= r_max)) | (d_ns_j <= 0.032 * ms)
        cand_h = (d_ij >= 0.02 * ms) & (d_ij <= r_max)
        cand = jnp.where(same_ep_strict, cand_h0, cand_h)

        def check_chunk(i_start):
            s_chunk = jax.lax.dynamic_slice(rs[:, :2], (i_start, 0), (256, 2))
            mid1 = 0.67 * s_chunk[:, None, :] + 0.33 * rs[None, :, :2]
            mid2 = 0.33 * s_chunk[:, None, :] + 0.67 * rs[None, :, :2]
            d_m1 = jnp.linalg.norm(mid1[:, :, None, :] - rs[None, None, :2048, :2], axis=-1).min(axis=-1)
            d_m2 = jnp.linalg.norm(mid2[:, :, None, :] - rs[None, None, :2048, :2], axis=-1).min(axis=-1)
            return (d_m1 < supp_thresh) & (d_m2 < supp_thresh)

        valid_mid = jax.vmap(check_chunk)(jnp.arange(0, 4096, 256)).reshape(4096, 4096)
        adj = jnp.where(check_manifold, cand & valid_mid, cand)

        v_g = jnp.where(rr == 0.0, 0.0, -200.0)
        steps_ij = jnp.maximum(1.0, d_ij / max(p90_step, 0.006))
        def bellman_step(v_cur, _):
            q_ij = jnp.where(adj, -steps_ij + (0.995 ** steps_ij) * v_cur[None, :], -200.0)
            v_next = jnp.maximum(v_cur, jnp.max(q_ij, axis=-1))
            return jnp.where(rr == 0.0, 0.0, v_next), None

        v_star, _ = jax.lax.scan(bellman_step, v_g, None, length=90)
        adj_target = jnp.where(cross_ep_only, adj & (rep[:, None] != rep[None, :]), adj)
        q_final = jnp.where(adj_target, -steps_ij + (0.995 ** steps_ij) * v_star[None, :], -200.0)
        best_next = jnp.argmax(q_final, axis=-1)
        return v_star, rs[best_next], v_star[best_next]

    # Arm 1: Model-Free (H=0) - strictly same-episode + 1-step TD smoothing (no synthetic rollouts!)
    v_h0, targ_h0, _ = compute_graph_for_mode(ref_s, ref_h5, ref_r, ref_ep, 0.075 * ms, True, False, True)
    # Arm 2: Short-Horizon Dyna (H=1) - 1-step synthetic reach
    v_h1, targ_h1, _ = compute_graph_for_mode(ref_s, ref_ns, ref_r, ref_ep, 0.095 * ms, True, False, False)
    # Arm 3: Medium-Horizon Dyna (H=5) - 5-step synthetic reach
    v_h5, targ_h5, _ = compute_graph_for_mode(ref_s, ref_ns, ref_r, ref_ep, 0.135 * ms, True, False, False)
    # Arm 4: TD(lambda)-MVE / STEVE (H=15 Target-Only) - full H=15 value backup, but actor updates only on local 1-step target without bridge trajectory steering
    v_mve, targ_mve, _ = compute_graph_for_mode(ref_s, ref_ns, ref_r, ref_ep, 0.220 * ms, True, True, False)
    # Arm 5: Long-Horizon Unfiltered Dyna (H=15) - H=15 reach WITHOUT manifold wall/support filtering (cuts across walls & OOD states!)
    v_unfilt, targ_unfilt, _ = compute_graph_for_mode(ref_s, ref_ns, ref_r, ref_ep, 0.320 * ms, False, False, False)
    # Arm 6: Principled Target-Anchored Bridge Dyna (H=15) - H=15 cross-episode anchor pairing WITH collinear manifold support + CEM bridge synthesis!
    v_bridge, targ_bridge, tv_bridge = compute_graph_for_mode(ref_s, ref_ns, ref_r, ref_ep, 0.220 * ms, True, True, False)

    # 3. Compute Test 2 Stitching Metrics comparing Rollout/Bridge Generation Families
    H_cem = 15
    K_pop = 32
    K_elite = 6

    @jax.jit
    def cem_bridge_opt(dyn_p, s0, s_target, rng_key):
        B = s0.shape[0]
        mean = jnp.zeros((B, H_cem, act_dim))
        std = jnp.ones((B, H_cem, act_dim)) * 0.8

        def cem_iter(carry, _):
            m, sd, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (K_pop, B, H_cem, act_dim))
            acts = jnp.clip(m[None, ...] + sd[None, ...] * noise, -1 + eps, 1 - eps)

            def roll_step(cur_s, a_t):
                flat_x = jnp.concatenate([cur_s, a_t], axis=-1).reshape(-1, ob_dim + act_dim)
                deltas = dyn_mod.apply({"params": dyn_p}, flat_x)
                mean_d = deltas.mean(axis=0).reshape(K_pop, B, ob_dim)
                unc = deltas.std(axis=0).mean(axis=-1).reshape(K_pop, B)
                next_s = cur_s + mean_d
                return next_s, (next_s, unc)

            init_s = jnp.broadcast_to(s0[None, ...], (K_pop, B, ob_dim))
            _, (traj_s, traj_unc) = jax.lax.scan(roll_step, init_s, jnp.transpose(acts, (2, 0, 1, 3)))
            dists = jnp.linalg.norm(traj_s[..., :2] - s_target[None, None, :, :2], axis=-1)
            min_dist = jnp.min(dists, axis=0)
            end_dist = dists[-1]
            unc_pen = traj_unc.mean(axis=0)
            cost = end_dist + 0.5 * min_dist + 3.0 * unc_pen

            elite_idx = jnp.argsort(cost, axis=0)[:K_elite]
            b_idx = jnp.arange(B)[None, :]
            elite_acts = acts[elite_idx, b_idx]
            new_m = elite_acts.mean(axis=0)
            new_sd = elite_acts.std(axis=0) + 0.05
            return (new_m, new_sd, k), min_dist.min(axis=0)

        (best_m, _, _), min_d = jax.lax.scan(cem_iter, (mean, std, rng_key), None, length=4)
        def final_roll(cur_s, a_t):
            x = jnp.concatenate([cur_s, a_t], axis=-1)
            deltas = dyn_mod.apply({"params": dyn_p}, x)
            ns = cur_s + deltas.mean(axis=0)
            unc = deltas.std(axis=0).mean(axis=-1)
            return ns, (cur_s, a_t, ns, unc)
        _, (bs, ba, bns, bunc) = jax.lax.scan(final_roll, s0, jnp.transpose(best_m, (1, 0, 2)))
        return bs, ba, bns, bunc, min_d[-1]

    rng, ck = jax.random.split(rng)
    bs, ba, bns, bunc, b_mind = cem_bridge_opt(dyn_st.params, ref_s, targ_bridge, ck)
    keep_b = (v_bridge > -190.0) & (tv_bridge > v_bridge) & (b_mind < 0.05 * ms)
    if int(keep_b.sum()) < 64:
        keep_b = (tv_bridge > v_bridge)

    @jax.jit
    def unguided_rollout(dyn_p, s0, r_key):
        def roll_step(carry, _):
            cur_s, k = carry
            k, ak = jax.random.split(k)
            a_t = jax.random.uniform(ak, (cur_s.shape[0], act_dim), minval=-0.8, maxval=0.8)
            x = jnp.concatenate([cur_s, a_t], axis=-1)
            deltas = dyn_mod.apply({"params": dyn_p}, x)
            ns = cur_s + deltas.mean(axis=0)
            return (ns, k), (cur_s, a_t, ns)
        _, (ug_s, ug_a, ug_ns) = jax.lax.scan(roll_step, (s0, r_key), None, length=15)
        return ug_s, ug_a, ug_ns

    def eval_stitch_metrics(end_s, start_ep, start_v):
        d = jnp.linalg.norm(end_s[:, None, :2] - ref_s[None, :, :2], axis=-1)
        nn_idx = jnp.argmin(d, axis=-1)
        d_real = jnp.min(d, axis=-1)
        ep_sw = (ref_ep[nn_idx] != start_ep).astype(jnp.float32)
        dv = v_bridge[nn_idx] - start_v
        in_supp = (d_real < 0.038 * ms).astype(jnp.float32)
        upward_stitch = ((ep_sw > 0.5) & (d_real < 0.038 * ms) & (dv > 1.0)).astype(jnp.float32)
        return {
            "ep_switch_pct": float(ep_sw.mean() * 100.0),
            "d_real": float(d_real.mean()),
            "delta_v": float(dv.mean()),
            "in_support_pct": float(in_supp.mean() * 100.0),
            "upward_stitch_pct": float(upward_stitch.mean() * 100.0),
        }

    rng, uk15 = jax.random.split(rng)
    ug_s_traj, ug_a_traj, ug_ns_traj = unguided_rollout(dyn_st.params, ref_s, uk15)
    ug_1 = ug_ns_traj[0]
    ug_5 = ug_ns_traj[4]
    ug_15 = ug_ns_traj[14]
    br_15 = bns[-1]

    stitch_diag = {
        "short_h1": eval_stitch_metrics(ug_1, ref_ep, v_bridge),
        "medium_h5": eval_stitch_metrics(ug_5, ref_ep, v_bridge),
        "unguided_h15": eval_stitch_metrics(ug_15, ref_ep, v_bridge),
        "principled_bridge_h15": eval_stitch_metrics(br_15[keep_b], ref_ep[keep_b], v_bridge[keep_b]),
    }
    print(f"[{key}] Stitch diag (Principled H=15 vs Unguided H=15): {stitch_diag['principled_bridge_h15']} vs {stitch_diag['unguided_h15']}", flush=True)

    # 4. Train and Evaluate All 6 Arms!
    grid_1d = jnp.linspace(-1.0, 1.0, 10)
    gx, gy = jnp.meshgrid(grid_1d, grid_1d)
    rbf_centers = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

    def rbf_enc(s):
        xy = s[..., :2]
        d2 = jnp.sum(jnp.square(xy[..., None, :] - rbf_centers), axis=-1)
        rbf = jnp.exp(-d2 / (2.0 * 0.14 * 0.14))
        return jnp.concatenate([s, rbf], axis=-1)

    class PolicyValNet(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, s):
            phi = rbf_enc(s)
            v = nn.Dense(512)(phi); v = nn.relu(v); v = nn.LayerNorm()(v)
            v = nn.Dense(512)(v); v = nn.relu(v); v = nn.LayerNorm()(v)
            v = nn.Dense(1)(v).squeeze(-1)

            pi = nn.Dense(512)(phi); pi = nn.relu(pi); pi = nn.LayerNorm()(pi)
            pi = nn.Dense(512)(pi); pi = nn.relu(pi); pi = nn.LayerNorm()(pi)
            pi = nn.Dense(512)(pi); pi = nn.relu(pi); pi = nn.LayerNorm()(pi)
            pi = jnp.tanh(nn.Dense(self.act_dim)(pi))
            return v, pi

    pv_mod = PolicyValNet(act_dim)

    b_s_valid = bs[:, keep_b].reshape(-1, ob_dim)
    b_a_valid = ba[:, keep_b].reshape(-1, act_dim)
    ug_s_flat = ug_s_traj.reshape(-1, ob_dim)
    ug_a_flat = ug_a_traj.reshape(-1, act_dim)

    @jax.jit
    def train_arm_500(st, r, ds, arm_v, arm_targ, use_h15_prog, use_cem_syn, use_unfilt_syn):
        N = ds["s"].shape[0]
        N_b = b_s_valid.shape[0]
        N_ug = ug_s_flat.shape[0]
        def lookup(s_batch):
            d = jnp.linalg.norm(s_batch[:, None, :2] - ref_s[None, :, :2], axis=-1)
            idx = jnp.argmin(d, axis=-1)
            return arm_v[idx], arm_targ[idx, :2]

        def step(carry, _):
            s_st, k = carry
            k, ik, bk, ugk = jax.random.split(k, 4)
            idx = jax.random.randint(ik, (512,), 0, N)
            s = ds["s"][idx]
            a = ds["a"][idx]
            ns = ds["ns"][idx]
            s_look = jnp.where(use_h15_prog, ds["s_h15"][idx], ds["s_h5"][idx])

            v_s, targ_xy = lookup(s)
            v_ns, _ = lookup(ns)
            v_look, _ = lookup(s_look)

            dir_to_anc = targ_xy - s[:, :2]
            dir_norm = dir_to_anc / (jnp.linalg.norm(dir_to_anc, axis=-1, keepdims=True) + 1e-6)
            # Normalize progress by p90_step so pointmass, ant, and humanoid have identical dimensionless scale!
            step_prog = jnp.sum((ns[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            look_prog = jnp.sum((s_look[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            dv = jnp.where(use_h15_prog, v_look - v_s, v_ns - v_s)

            prog_metric = jnp.where(use_h15_prog, 0.5 * step_prog + 0.12 * look_prog, step_prog)
            good = (step_prog > 0.05) & (dv > 0.05)
            w_pi = jnp.where(good, jnp.minimum(jnp.exp(prog_metric * 2.2 + dv * 0.4), 50.0), 0.01)

            ib = jax.random.randint(bk, (128,), 0, N_b)
            sb, ab = b_s_valid[ib], b_a_valid[ib]
            wb = jnp.where(use_cem_syn & (ob_dim == 2), 15.0, 0.0)

            # In Unfiltered Dyna (H=15), unfiltered synthetic rollout states/actions are mixed into the batch!
            iug = jax.random.randint(ugk, (128,), 0, N_ug)
            sug, aug = ug_s_flat[iug], ug_a_flat[iug]
            wug = jnp.where(use_unfilt_syn, 8.0, 0.0)

            def loss_fn(p):
                v_pred, a_pred = pv_mod.apply({"params": p}, s)
                _, ab_pred = pv_mod.apply({"params": p}, sb)
                _, aug_pred = pv_mod.apply({"params": p}, sug)
                v_loss = jnp.square(v_pred - v_s).mean()
                pi_loss = (
                    (w_pi[:, None] * jnp.square(a_pred - a)).mean()
                    + (wb * jnp.square(ab_pred - ab)).mean()
                    + (wug * jnp.square(aug_pred - aug)).mean()
                )
                return v_loss + pi_loss, pi_loss

            (_, pl), grads = jax.value_and_grad(loss_fn, has_aux=True)(s_st.params)
            return (s_st.apply_gradients(grads=grads), k), pl

        (st, r), pls = jax.lax.scan(step, (st, r), None, length=500)
        return st, r, pls[-1]

    n_eval = 20
    eval_envs = [ogbench.make_env_and_datasets(ename, env_only=True) for _ in range(n_eval)]

    @jax.jit
    def act_fn(p, ob_b):
        _, pi = pv_mod.apply({"params": p}, ob_b)
        return pi

    def evaluate_arm(params):
        obs = np.stack([e.reset(seed=100 + i)[0] for i, e in enumerate(eval_envs)])
        successes = np.zeros(n_eval, dtype=bool)
        dones = np.zeros(n_eval, dtype=bool)
        min_dists = np.linalg.norm(obs[:, :2] - goal_xy[None, :], axis=-1)

        for _ in range(max_ep_steps):
            if np.all(dones):
                break
            acts = np.asarray(act_fn(params, jnp.asarray(norm_ob(obs), dtype=jnp.float32)))
            for i in range(n_eval):
                if not dones[i]:
                    ob_next, rew, term, trunc, info = eval_envs[i].step(acts[i])
                    obs[i] = ob_next
                    d_now = float(np.linalg.norm(ob_next[:2] - goal_xy))
                    if d_now < min_dists[i]:
                        min_dists[i] = d_now
                    if info.get("success", 0.0) > 0.5:
                        successes[i] = True
                    if term or trunc:
                        dones[i] = True
        return {
            "success_rate_pct": float(successes.mean() * 100.0),
            "mean_min_dist": float(min_dists.mean()),
            "final_dist": float(np.linalg.norm(obs[:, :2] - goal_xy[None, :], axis=-1).mean()),
        }

    arms = [
        ("model_free_h0", v_h0, targ_h0, False, False, False),
        ("short_dyna_h1", v_h1, targ_h1, False, False, False),
        ("medium_dyna_h5", v_h5, targ_h5, False, False, False),
        ("td_lambda_mve_h15", v_mve, targ_h1, False, False, False),
        ("unfiltered_dyna_h15", v_unfilt, targ_unfilt, True, False, True),
        ("principled_bridge_h15", v_bridge, targ_bridge, True, True, False),
    ]

    arm_results = {}
    n_chunks = 50 if "humanoid" in key else 36
    for arm_name, av, atarg, uh15, ucem, uug in arms:
        rng, ak = jax.random.split(rng)
        st = TrainState.create(
            apply_fn=pv_mod.apply,
            params=pv_mod.init(ak, gpu_real["s"][:2])["params"],
            tx=optax.adam(3e-4),
        )
        for _ in range(n_chunks):
            st, rng, pl = train_arm_500(
                st, rng, gpu_real, av, atarg, jnp.bool_(uh15), jnp.bool_(ucem), jnp.bool_(uug)
            )
        jax.block_until_ready(pl)
        res = evaluate_arm(st.params)
        arm_results[arm_name] = res
        print(f"[{key}] {arm_name:24s} -> success={res['success_rate_pct']:5.1f}%, min_dist={res['mean_min_dist']:5.2f}m", flush=True)

    out = {
        "env_key": key,
        "dataset_name": dname,
        "env_name": ename,
        "stitch_diagnostics": stitch_diag,
        "arm_results": arm_results,
        "elapsed_sec": float(time.time() - t0),
    }
    os.makedirs("/tmp/maze_stitch_results", exist_ok=True)
    with open(f"/tmp/maze_stitch_results/{key}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{key}] DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
