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

    # Build Reference Pool (K_ref = 4096)
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
    ref_h15 = gpu_real["s_h15"][ref_idx]
    ref_ep = jnp.asarray(ep_np[ref_idx])
    ref_r = gpu_real["r"][ref_idx]

    # Train Forward Dynamics Ensemble f_fwd(s, a), Reverse Dynamics Ensemble f_rev(ns, a),
    # Inverse Dynamics f_inv(s, s_target), and Advantage-Tilted Reverse BC Policy pi_rev_tilt(a | ns)
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

    class InvPolicy(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, s, s_next):
            x = jnp.concatenate([s, s_next - s], axis=-1)
            h = nn.relu(nn.Dense(256)(x))
            h = nn.relu(nn.Dense(256)(h))
            return jnp.tanh(nn.Dense(self.act_dim)(h))

    class TiltedRevBC(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, ns):
            h = nn.relu(nn.Dense(256)(ns))
            h = nn.relu(nn.Dense(256)(h))
            return jnp.tanh(nn.Dense(self.act_dim)(h))

    fwd_dyn = EnsembleDyn(num_ens=5, out_dim=ob_dim)
    rev_dyn = EnsembleDyn(num_ens=5, out_dim=ob_dim)
    inv_mod = InvPolicy(act_dim=act_dim)
    tilt_rbc = TiltedRevBC(act_dim=act_dim)

    rng = jax.random.PRNGKey(77)
    rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
    fwd_st = TrainState.create(apply_fn=fwd_dyn.apply, params=fwd_dyn.init(k1, jnp.concatenate([gpu_real["s"][:2], gpu_real["a"][:2]], axis=-1))["params"], tx=optax.adam(1e-3))
    rev_st = TrainState.create(apply_fn=rev_dyn.apply, params=rev_dyn.init(k2, jnp.concatenate([gpu_real["ns"][:2], gpu_real["a"][:2]], axis=-1))["params"], tx=optax.adam(1e-3))
    inv_st = TrainState.create(apply_fn=inv_mod.apply, params=inv_mod.init(k3, gpu_real["s"][:2], gpu_real["ns"][:2])["params"], tx=optax.adam(1e-3))
    tilt_st = TrainState.create(apply_fn=tilt_rbc.apply, params=tilt_rbc.init(k4, gpu_real["ns"][:2])["params"], tx=optax.adam(1e-3))

    # First compute Idea 2 (Real-to-Real 1-Step / Local Model-Verified Graph) & Value V(s)
    # In Idea 2, two real states s_i, s_j in D_real are connected if their local step displacement
    # is physically reachable without manual 33%/67% checks!
    @jax.jit
    def train_models_500(f_st, r_st, i_st, r, ds):
        N = ds["s"].shape[0]
        def step(carry, _):
            fs, rs, ins, k = carry
            k, ik = jax.random.split(k)
            idx = jax.random.randint(ik, (512,), 0, N)
            s, a, ns = ds["s"][idx], ds["a"][idx], ds["ns"][idx]
            xf = jnp.concatenate([s, a], axis=-1)
            xr = jnp.concatenate([ns, a], axis=-1)
            tf = ns - s
            tr = s - ns
            _, fg = jax.value_and_grad(lambda p: jnp.square(fwd_dyn.apply({"params": p}, xf) - tf[None, ...]).mean())(fs.params)
            _, rg = jax.value_and_grad(lambda p: jnp.square(rev_dyn.apply({"params": p}, xr) - tr[None, ...]).mean())(rs.params)
            loss_i, ig = jax.value_and_grad(lambda p: jnp.square(inv_mod.apply({"params": p}, s, ns) - a).mean())(ins.params)
            return (fs.apply_gradients(grads=fg), rs.apply_gradients(grads=rg), ins.apply_gradients(grads=ig), k), loss_i
        (f_st, r_st, i_st, r), li = jax.lax.scan(step, (f_st, r_st, i_st, r), None, length=500)
        return f_st, r_st, i_st, r, li[-1]

    for _ in range(25):
        fwd_st, rev_st, inv_st, rng, iloss = train_models_500(fwd_st, rev_st, inv_st, rng, gpu_real)
    jax.block_until_ready(iloss)

    # Idea 2: Model-Verified Real-to-Real Local Stitch (Zero Multi-Step Rollout, Zero 33%/67% Heuristic!)
    # For any pair of real states (s_i, s_j) within local reach (or within-episode lookahead),
    # verify 1-step forward-inverse model consistency || f_fwd(s_i, f_inv(s_i, s_j)) - s_j ||!
    @jax.jit
    def compute_idea2_real_to_real_graph(rs, rns, rh15, rr, rep, f_p, i_p):
        d_ij = jnp.linalg.norm(rs[:, None, :2] - rs[None, :, :2], axis=-1)
        d_ns_j = jnp.linalg.norm(rns[:, None, :2] - rs[None, :, :2], axis=-1)
        d_h15_j = jnp.linalg.norm(rh15[:, None, :2] - rs[None, :, :2], axis=-1)

        # Check 1-step forward-inverse directional feasibility + support density (no CEM!)
        def verify_chunk(carry, i_start):
            s_c = jax.lax.dynamic_slice(rs, (i_start, 0), (64, ob_dim))
            s_b = jnp.broadcast_to(s_c[:, None, :], (64, 4096, ob_dim)).reshape(-1, ob_dim)
            t_b = jnp.broadcast_to(rs[None, :, :], (64, 4096, ob_dim)).reshape(-1, ob_dim)
            a_hat = inv_mod.apply({"params": i_p}, s_b, t_b)
            deltas = fwd_dyn.apply({"params": f_p}, jnp.concatenate([s_b, a_hat], axis=-1))
            mean_d = deltas.mean(axis=0).reshape(64, 4096, ob_dim)
            unc = deltas.std(axis=0).mean(axis=-1).reshape(64, 4096)
            diff_xy = t_b[:, :2].reshape(64, 4096, 2) - s_c[:, None, :2]
            dist_xy = jnp.linalg.norm(diff_xy, axis=-1) + 1e-6
            dir_prog = jnp.sum(mean_d[:, :, :2] * (diff_xy / dist_xy[:, :, None]), axis=-1)
            mid_xy = 0.5 * (s_c[:, None, :2] + rs[None, :, :2])
            d_mid = jnp.linalg.norm(mid_xy[:, :, None, :] - rs[None, None, :2048, :2], axis=-1).min(axis=-1)
            return carry, ((dir_prog > 0.15 * p90_step) & (unc < 0.05) & (d_mid < 0.048 * ms))

        _, model_ok_chunks = jax.lax.scan(verify_chunk, None, jnp.arange(0, 4096, 64))
        model_ok = model_ok_chunks.reshape(4096, 4096)
        adj_r2r = ((d_ij >= 0.02 * ms) & (d_ij <= 0.215 * ms) & model_ok) | (d_ns_j <= 0.045 * ms) | ((rep[:, None] == rep[None, :]) & (d_h15_j <= 0.045 * ms))

        v_g = jnp.where(rr == 0.0, 0.0, -200.0)
        steps_ij = jnp.maximum(1.0, d_ij / max(p90_step, 0.006))
        def bellman_step(v_cur, _):
            q_ij = jnp.where(adj_r2r, -steps_ij + (0.995 ** steps_ij) * v_cur[None, :], -200.0)
            v_next = jnp.maximum(v_cur, jnp.max(q_ij, axis=-1))
            return jnp.where(rr == 0.0, 0.0, v_next), None
        v_r2r, _ = jax.lax.scan(bellman_step, v_g, None, length=120)
        q_final = jnp.where(adj_r2r, -steps_ij + (0.995 ** steps_ij) * v_r2r[None, :], -200.0)
        best_j = jnp.argmax(q_final, axis=-1)
        return v_r2r, rs[best_j], v_r2r[best_j]

    v_idea2, targ_idea2, tv_idea2 = compute_idea2_real_to_real_graph(
        ref_s, ref_ns, ref_h15, ref_r, ref_ep, fwd_st.params, inv_st.params
    )

    # Idea 1: Advantage-Tilted ROMI (AT-ROMI) — 1-line change to Reverse BC:
    # Weight reverse BC loss by w = exp(beta * (V(s_{t+1}) - V(s_t)))!
    @jax.jit
    def train_tilted_rbc_500(t_st, r, ds, v_ref):
        N = ds["s"].shape[0]
        def lookup_v(sb):
            d = jnp.linalg.norm(sb[:, None, :2] - ref_s[None, :, :2], axis=-1)
            return v_ref[jnp.argmin(d, axis=-1)]
        def step(carry, _):
            ts, k = carry
            k, ik = jax.random.split(k)
            idx = jax.random.randint(ik, (512,), 0, N)
            s, a, ns, s_h15 = ds["s"][idx], ds["a"][idx], ds["ns"][idx], ds["s_h15"][idx]
            dv = lookup_v(s_h15) - lookup_v(s)
            # Single-line Advantage-Tilted weight: suppress subgoal hovering (dv <= 0)!
            w_tilt = jnp.where(dv > 0.05, jnp.minimum(jnp.exp(2.0 * dv), 30.0), 0.01)
            loss, grads = jax.value_and_grad(
                lambda p: (w_tilt[:, None] * jnp.square(tilt_rbc.apply({"params": p}, ns) - a)).mean()
            )(ts.params)
            return (ts.apply_gradients(grads=grads), k), loss
        (t_st, r), ls = jax.lax.scan(step, (t_st, r), None, length=500)
        return t_st, r, ls[-1]

    for _ in range(20):
        tilt_st, rng, tloss = train_tilted_rbc_500(tilt_st, rng, gpu_real, v_idea2)
    jax.block_until_ready(tloss)

    # Roll backward H=15 steps under Advantage-Tilted Reverse BC (zero CEM!)
    @jax.jit
    def at_romi_rollout(rd_p, trbc_p, s_term):
        def back_step(cur_ns, _):
            a_prev = tilt_rbc.apply({"params": trbc_p}, cur_ns)
            deltas = rev_dyn.apply({"params": rd_p}, jnp.concatenate([cur_ns, a_prev], axis=-1))
            prev_s = cur_ns + deltas.mean(axis=0)
            return prev_s, (prev_s, a_prev, cur_ns)
        _, (at_s, at_a, at_ns) = jax.lax.scan(back_step, s_term, None, length=15)
        return at_s, at_a, at_ns

    at_s_traj, at_a_traj, at_ns_traj = at_romi_rollout(rev_st.params, tilt_st.params, ref_s)
    at_start_s0 = at_s_traj[-1]

    # Compute AT-ROMI value landscape and stitch diagnostics
    d_at = jnp.linalg.norm(at_start_s0[:, None, :2] - ref_s[None, :, :2], axis=-1)
    nn_at = jnp.argmin(d_at, axis=-1)
    d_real_at = jnp.min(d_at, axis=-1)
    ep_sw_at = (ref_ep[nn_at] != ref_ep).astype(jnp.float32)
    dv_at = v_idea2 - v_idea2[nn_at]
    up_at = ((ep_sw_at > 0.5) & (d_real_at < 0.038 * ms) & (dv_at > 1.0)).astype(jnp.float32)
    at_diag = {
        "ep_switch_pct": float(ep_sw_at.mean() * 100.0),
        "d_real": float(d_real_at.mean()),
        "terminal_d_real": 0.0,
        "delta_v": float(dv_at.mean()),
        "in_support_pct": float((d_real_at < 0.038 * ms).mean() * 100.0),
        "upward_stitch_pct": float(up_at.mean() * 100.0),
    }

    # Also compute Stitch Diagnostics for Idea 2 (Real-to-Real Model-Verified Stitch)
    ep_sw_r2r = (ref_ep != ref_ep[jnp.argmin(jnp.linalg.norm(targ_idea2[:, None, :2] - ref_s[None, :, :2], axis=-1), axis=-1)]).astype(jnp.float32)
    dv_r2r = tv_idea2 - v_idea2
    r2r_diag = {
        "ep_switch_pct": float(ep_sw_r2r.mean() * 100.0),
        "d_real": 0.0,
        "terminal_d_real": 0.0,
        "delta_v": float(dv_r2r.mean()),
        "in_support_pct": 100.0,
        "upward_stitch_pct": float(((ep_sw_r2r > 0.5) & (dv_r2r > 0.5)).mean() * 100.0),
    }
    print(f"[{key}] Clean Ideas Diag: AT-ROMI={at_diag} | Real-to-Real={r2r_diag}", flush=True)

    # Train PolicyValNet for Idea 1 (AT-ROMI) and Idea 2 (1-Step Real-to-Real Model Stitch)
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
    at_s_flat = at_s_traj.reshape(-1, ob_dim)
    at_a_flat = at_a_traj.reshape(-1, act_dim)

    @jax.jit
    def train_clean_arm_500(st, r, ds, arm_v, arm_targ, use_at_syn):
        N = ds["s"].shape[0]
        N_at = at_s_flat.shape[0]
        def lookup(s_batch):
            d = jnp.linalg.norm(s_batch[:, None, :2] - ref_s[None, :, :2], axis=-1)
            idx = jnp.argmin(d, axis=-1)
            return arm_v[idx], arm_targ[idx, :2]

        def step(carry, _):
            s_st, k = carry
            k, ik, atk = jax.random.split(k, 3)
            idx = jax.random.randint(ik, (512,), 0, N)
            s, a, ns, s_look = ds["s"][idx], ds["a"][idx], ds["ns"][idx], ds["s_h15"][idx]

            v_s, targ_xy = lookup(s)
            v_look, _ = lookup(s_look)

            dir_to_anc = targ_xy - s[:, :2]
            dir_norm = dir_to_anc / (jnp.linalg.norm(dir_to_anc, axis=-1, keepdims=True) + 1e-6)
            step_prog = jnp.sum((ns[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            look_prog = jnp.sum((s_look[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            dv = v_look - v_s

            prog_metric = 0.5 * step_prog + 0.12 * look_prog
            good = (step_prog > 0.05) & (dv > 0.05)
            w_pi = jnp.where(good, jnp.minimum(jnp.exp(prog_metric * 2.2 + dv * 0.4), 50.0), 0.01)

            iat = jax.random.randint(atk, (128,), 0, N_at)
            sat, aat = at_s_flat[iat], at_a_flat[iat]
            wat = jnp.where(use_at_syn & (ob_dim == 4), 2.0, 0.0)

            def loss_fn(p):
                v_pred, a_pred = pv_mod.apply({"params": p}, s)
                _, aat_pred = pv_mod.apply({"params": p}, sat)
                v_loss = jnp.square(v_pred - v_s).mean()
                pi_loss = (w_pi[:, None] * jnp.square(a_pred - a)).mean() + (wat * jnp.square(aat_pred - aat)).mean()
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

    json_path = f"/tmp/maze_stitch_results/{key}.json"
    with open(json_path, "r") as f:
        existing = json.load(f)

    existing["stitch_diagnostics"]["at_romi_h15"] = at_diag
    existing["stitch_diagnostics"]["real_to_real_model_stitch"] = r2r_diag

    n_chunks = 50 if "humanoid" in key else 36
    for arm_name, use_at in [("at_romi_h15", True), ("real_to_real_model_stitch", False)]:
        rng, ak = jax.random.split(rng)
        st = TrainState.create(
            apply_fn=pv_mod.apply,
            params=pv_mod.init(ak, gpu_real["s"][:2])["params"],
            tx=optax.adam(3e-4),
        )
        for _ in range(n_chunks):
            st, rng, pl = train_clean_arm_500(st, rng, gpu_real, v_idea2, targ_idea2, jnp.bool_(use_at))
        jax.block_until_ready(pl)
        res = evaluate_arm(st.params)
        existing["arm_results"][arm_name] = res
        print(f"[{key}] {arm_name:26s} -> success={res['success_rate_pct']:5.1f}%, min_dist={res['mean_min_dist']:5.2f}m", flush=True)

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[{key}] CLEAN IDEAS DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
