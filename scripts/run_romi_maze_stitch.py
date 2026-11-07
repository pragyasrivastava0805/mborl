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

    # 1. Train 5-member Reverse Dynamics Ensemble p_psi(s_t - s_{t+1} | s_{t+1}, a_t) & Forward Dynamics Ensemble
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

    class ReverseBCPolicy(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, ns, z):
            x = jnp.concatenate([ns, z], axis=-1)
            h = nn.relu(nn.Dense(256)(x))
            h = nn.relu(nn.Dense(256)(h))
            return jnp.tanh(nn.Dense(self.act_dim)(h))

    rev_dyn_mod = EnsembleDyn(num_ens=5, out_dim=ob_dim)
    rev_bc_mod = ReverseBCPolicy(act_dim=act_dim)

    rng = jax.random.PRNGKey(10)
    rng, rdk, rbck = jax.random.split(rng, 3)
    rev_dyn_params = rev_dyn_mod.init(rdk, jnp.concatenate([gpu_real["ns"][:2], gpu_real["a"][:2]], axis=-1))["params"]
    rev_dyn_st = TrainState.create(apply_fn=rev_dyn_mod.apply, params=rev_dyn_params, tx=optax.adam(1e-3))

    rev_bc_params = rev_bc_mod.init(rbck, gpu_real["ns"][:2], jnp.zeros((2, act_dim)))["params"]
    rev_bc_st = TrainState.create(apply_fn=rev_bc_mod.apply, params=rev_bc_params, tx=optax.adam(1e-3))

    @jax.jit
    def train_rev_500(rd_st, rbc_st, r, ds):
        N = ds["s"].shape[0]
        def step(carry, _):
            d_s, b_s, k = carry
            k, ik, zk = jax.random.split(k, 3)
            idx = jax.random.randint(ik, (512,), 0, N)
            s = ds["s"][idx]
            a = ds["a"][idx]
            ns = ds["ns"][idx]
            x_rev = jnp.concatenate([ns, a], axis=-1)
            target_rev = s - ns

            def d_loss_fn(p):
                pred = rev_dyn_mod.apply({"params": p}, x_rev)
                return jnp.square(pred - target_rev[None, ...]).mean()
            dl, d_grads = jax.value_and_grad(d_loss_fn)(d_s.params)

            z = jax.random.normal(zk, (512, act_dim)) * 0.15
            def bc_loss_fn(p):
                a_pred = rev_bc_mod.apply({"params": p}, ns, z)
                return jnp.square(a_pred - a).mean()
            bl, b_grads = jax.value_and_grad(bc_loss_fn)(b_s.params)
            return (d_s.apply_gradients(grads=d_grads), b_s.apply_gradients(grads=b_grads), k), (dl, bl)
        (rd_st, rbc_st, r), (dls, bls) = jax.lax.scan(step, (rd_st, rbc_st, r), None, length=500)
        return rd_st, rbc_st, r, dls[-1], bls[-1]

    for _ in range(25):
        rev_dyn_st, rev_bc_st, rng, rd_loss, rbc_loss = train_rev_500(rev_dyn_st, rev_bc_st, rng, gpu_real)
    jax.block_until_ready(rd_loss)
    print(f"[{key}] Trained ReverseDyn MSE={float(rd_loss):.6f}, ReverseBC MSE={float(rbc_loss):.6f}", flush=True)

    # 2. Build Reference Anchor Pool (K_ref = 4096)
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
    ref_ep = jnp.asarray(ep_np[ref_idx])
    ref_r = gpu_real["r"][ref_idx]

    supp_thresh = 0.05 * ms

    # Standard Unguided ROMI Rollout (H = 15 backward steps from ref_s under ReverseBC + ReverseDyn with uncertainty filter)
    @jax.jit
    def standard_romi_rollout(rd_p, rbc_p, s_terminal, r_key):
        def back_step(carry, _):
            cur_ns, k = carry
            k, zk = jax.random.split(k)
            z = jax.random.normal(zk, (cur_ns.shape[0], act_dim)) * 0.5
            a_prev = rev_bc_mod.apply({"params": rbc_p}, cur_ns, z)
            x_rev = jnp.concatenate([cur_ns, a_prev], axis=-1)
            deltas_rev = rev_dyn_mod.apply({"params": rd_p}, x_rev)
            mean_d = deltas_rev.mean(axis=0)
            unc = deltas_rev.std(axis=0).mean(axis=-1)
            # ROMI trust-region filter: freeze backward step if reverse uncertainty exceeds threshold
            valid_step = (unc < 0.045)[:, None]
            prev_s = jnp.where(valid_step, cur_ns + mean_d, cur_ns)
            return (prev_s, k), (prev_s, a_prev, cur_ns, unc)
        _, (rom_s, rom_a, rom_ns, rom_unc) = jax.lax.scan(back_step, (s_terminal, r_key), None, length=15)
        return rom_s, rom_a, rom_ns, rom_unc

    rng, rk15 = jax.random.split(rng)
    rom_s_traj, rom_a_traj, rom_ns_traj, rom_unc_traj = standard_romi_rollout(
        rev_dyn_st.params, rev_bc_st.params, ref_s, rk15
    )
    # In Standard ROMI, the backward rollout starts at s_H = ref_s (terminal state of the forward transition chain)
    # and reaches rom_s_traj[-1] = \hat{s}_0 (start of the synthetic 15-step chain)!
    rom_start_s0 = rom_s_traj[-1]

    # Compute Base Critic Value v_bridge on ref_s for Stitch Diagnostics
    @jax.jit
    def compute_romi_graphs(rs, rns, rr, rep, rom_s0):
        d_ij = jnp.linalg.norm(rs[:, None, :2] - rs[None, :, :2], axis=-1)
        def check_chunk(i_start):
            s_chunk = jax.lax.dynamic_slice(rs[:, :2], (i_start, 0), (256, 2))
            mid1 = 0.67 * s_chunk[:, None, :] + 0.33 * rs[None, :, :2]
            mid2 = 0.33 * s_chunk[:, None, :] + 0.67 * rs[None, :, :2]
            d_m1 = jnp.linalg.norm(mid1[:, :, None, :] - rs[None, None, :2048, :2], axis=-1).min(axis=-1)
            d_m2 = jnp.linalg.norm(mid2[:, :, None, :] - rs[None, None, :2048, :2], axis=-1).min(axis=-1)
            return (d_m1 < supp_thresh) & (d_m2 < supp_thresh)
        valid_mid = jax.vmap(check_chunk)(jnp.arange(0, 4096, 256)).reshape(4096, 4096)

        # 1. Principled Bidirectional / Target-Anchored ROMI Graph (H=15)
        adj_prin = (d_ij >= 0.02 * ms) & (d_ij <= 0.220 * ms) & valid_mid
        v_g = jnp.where(rr == 0.0, 0.0, -200.0)
        steps_ij = jnp.maximum(1.0, d_ij / max(p90_step, 0.006))
        def bellman_prin(v_cur, _):
            q_ij = jnp.where(adj_prin, -steps_ij + (0.995 ** steps_ij) * v_cur[None, :], -200.0)
            v_next = jnp.maximum(v_cur, jnp.max(q_ij, axis=-1))
            return jnp.where(rr == 0.0, 0.0, v_next), None
        v_prin, _ = jax.lax.scan(bellman_prin, v_g, None, length=90)
        q_prin = jnp.where(adj_prin & (rep[:, None] != rep[None, :]), -steps_ij + (0.995 ** steps_ij) * v_prin[None, :], -200.0)
        best_prin = jnp.argmax(q_prin, axis=-1)

        # 2. Standard Unguided ROMI Graph (H=15):
        # Backward fan from anchor j reaches rom_s0[j]; state i connects to anchor j if i is within 1-step/local radius of rom_s0[j] (or same episode)
        d_i_romj = jnp.linalg.norm(rs[:, None, :2] - rom_s0[None, :, :2], axis=-1)
        d_ns_j = jnp.linalg.norm(rns[:, None, :2] - rs[None, :, :2], axis=-1)
        adj_romi_std = ((d_i_romj <= 0.055 * ms) & valid_mid) | (d_ns_j <= 0.055 * ms) | ((rep[:, None] == rep[None, :]) & (d_ij <= 0.085 * ms))
        def bellman_std(v_cur, _):
            q_ij = jnp.where(adj_romi_std, -steps_ij + (0.995 ** steps_ij) * v_cur[None, :], -200.0)
            v_next = jnp.maximum(v_cur, jnp.max(q_ij, axis=-1))
            return jnp.where(rr == 0.0, 0.0, v_next), None
        v_std, _ = jax.lax.scan(bellman_std, v_g, None, length=90)
        q_std = jnp.where(adj_romi_std, -steps_ij + (0.995 ** steps_ij) * v_std[None, :], -200.0)
        best_std = jnp.argmax(q_std, axis=-1)

        return v_prin, rs[best_prin], v_prin[best_prin], v_std, rs[best_std], v_std[best_std]

    v_prin, targ_prin, tv_prin, v_std, targ_std, tv_std = compute_romi_graphs(
        ref_s, ref_ns, ref_r, ref_ep, rom_start_s0
    )

    # 3. Target-Anchored Reverse CEM Bridge Optimization (rolling BACKWARD from s^+ = targ_prin to s_0 = ref_s!)
    H_cem = 15
    K_pop = 32
    K_elite = 6

    @jax.jit
    def rev_cem_bridge_opt(rd_p, s_source, s_anchor_plus, rng_key):
        B = s_source.shape[0]
        mean = jnp.zeros((B, H_cem, act_dim))
        std = jnp.ones((B, H_cem, act_dim)) * 0.8

        def cem_iter(carry, _):
            m, sd, k = carry
            k, sk = jax.random.split(k)
            noise = jax.random.normal(sk, (K_pop, B, H_cem, act_dim))
            acts = jnp.clip(m[None, ...] + sd[None, ...] * noise, -1 + eps, 1 - eps)

            def rev_roll_step(cur_ns, a_t):
                flat_x = jnp.concatenate([cur_ns, a_t], axis=-1).reshape(-1, ob_dim + act_dim)
                deltas_rev = rev_dyn_mod.apply({"params": rd_p}, flat_x)
                mean_d = deltas_rev.mean(axis=0).reshape(K_pop, B, ob_dim)
                unc = deltas_rev.std(axis=0).mean(axis=-1).reshape(K_pop, B)
                prev_s = cur_ns + mean_d
                return prev_s, (prev_s, unc)

            init_ns = jnp.broadcast_to(s_anchor_plus[None, ...], (K_pop, B, ob_dim))
            _, (traj_prev_s, traj_unc) = jax.lax.scan(rev_roll_step, init_ns, jnp.transpose(acts, (2, 0, 1, 3)))
            dists_to_source = jnp.linalg.norm(traj_prev_s[..., :2] - s_source[None, None, :, :2], axis=-1)
            min_dist = jnp.min(dists_to_source, axis=0)
            end_dist = dists_to_source[-1]
            unc_pen = traj_unc.mean(axis=0)
            cost = end_dist + 0.5 * min_dist + 3.0 * unc_pen

            elite_idx = jnp.argsort(cost, axis=0)[:K_elite]
            b_idx = jnp.arange(B)[None, :]
            elite_acts = acts[elite_idx, b_idx]
            new_m = elite_acts.mean(axis=0)
            new_sd = elite_acts.std(axis=0) + 0.05
            return (new_m, new_sd, k), min_dist.min(axis=0)

        (best_m, _, _), min_d = jax.lax.scan(cem_iter, (mean, std, rng_key), None, length=4)
        def final_rev_roll(cur_ns, a_t):
            x = jnp.concatenate([cur_ns, a_t], axis=-1)
            deltas_rev = rev_dyn_mod.apply({"params": rd_p}, x)
            prev_s = cur_ns + deltas_rev.mean(axis=0)
            unc = deltas_rev.std(axis=0).mean(axis=-1)
            return prev_s, (prev_s, a_t, cur_ns, unc)
        _, (rbs, rba, rbns, rbunc) = jax.lax.scan(final_rev_roll, s_anchor_plus, jnp.transpose(best_m, (1, 0, 2)))
        return rbs, rba, rbns, rbunc, min_d[-1]

    rng, rck = jax.random.split(rng)
    rbs, rba, rbns, rbunc, rb_mind = rev_cem_bridge_opt(rev_dyn_st.params, ref_s, targ_prin, rck)
    keep_rb = (v_prin > -190.0) & (tv_prin > v_prin) & (rb_mind < 0.05 * ms)
    if int(keep_rb.sum()) < 64:
        keep_rb = (tv_prin > v_prin)

    # Evaluate Stitch Diagnostics for Standard ROMI (H=15) and Bidirectional/Target-Anchored ROMI (H=15)
    # In ROMI, backward rollout starts at anchor s_H = ref_s[j] and lands at \hat{s}_0 = rom_start_s0[j] (near some real state i)
    def eval_romi_stitch_metrics(start_s0_pred, anchor_ep, anchor_v, is_guided=False):
        d = jnp.linalg.norm(start_s0_pred[:, None, :2] - ref_s[None, :, :2], axis=-1)
        nn_idx = jnp.argmin(d, axis=-1)
        d_real_start = jnp.min(d, axis=-1)
        ep_sw = (ref_ep[nn_idx] != anchor_ep).astype(jnp.float32)
        # Value gain from matched source state nn_idx to terminal real anchor state s_H!
        dv = anchor_v - v_prin[nn_idx]
        in_supp = (d_real_start < 0.038 * ms).astype(jnp.float32)
        upward_stitch = ((ep_sw > 0.5) & (d_real_start < 0.038 * ms) & (dv > 1.0)).astype(jnp.float32)
        return {
            "ep_switch_pct": float(ep_sw.mean() * 100.0),
            "d_real": float(d_real_start.mean()),
            "terminal_d_real": 0.0,  # Hard-pinned to real dataset state s_H in D_real!
            "delta_v": float(dv.mean()),
            "in_support_pct": float(in_supp.mean() * 100.0),
            "upward_stitch_pct": float(upward_stitch.mean() * 100.0),
        }

    romi_std_diag = eval_romi_stitch_metrics(rom_start_s0, ref_ep, v_prin, is_guided=False)
    romi_prin_diag = eval_romi_stitch_metrics(rbs[-1][keep_rb], ref_ep[keep_rb], tv_prin[keep_rb], is_guided=True)
    print(f"[{key}] ROMI Stitch Diag: Std ROMI={romi_std_diag} | Guided ROMI={romi_prin_diag}", flush=True)

    # 4. Train and Evaluate the 2 ROMI Arms:
    # - standard_romi_h15 (Unguided Reverse BC + Reverse Dynamics Rollouts from real states)
    # - principled_romi_h15 (Target-Anchored / Bidirectional Reverse CEM Bridge Synthesis)
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

    rb_s_valid = rbs[:, keep_rb].reshape(-1, ob_dim)
    rb_a_valid = rba[:, keep_rb].reshape(-1, act_dim)
    rom_s_flat = rom_s_traj.reshape(-1, ob_dim)
    rom_a_flat = rom_a_traj.reshape(-1, act_dim)

    @jax.jit
    def train_romi_arm_500(st, r, ds, arm_v, arm_targ, is_guided_romi):
        N = ds["s"].shape[0]
        N_rb = rb_s_valid.shape[0]
        N_rom = rom_s_flat.shape[0]
        def lookup(s_batch):
            d = jnp.linalg.norm(s_batch[:, None, :2] - ref_s[None, :, :2], axis=-1)
            idx = jnp.argmin(d, axis=-1)
            return arm_v[idx], arm_targ[idx, :2]

        def step(carry, _):
            s_st, k = carry
            k, ik, rbk, romk = jax.random.split(k, 4)
            idx = jax.random.randint(ik, (512,), 0, N)
            s = ds["s"][idx]
            a = ds["a"][idx]
            ns = ds["ns"][idx]
            s_look = jnp.where(is_guided_romi, ds["s_h15"][idx], ds["s_h5"][idx])

            v_s, targ_xy = lookup(s)
            v_ns, _ = lookup(ns)
            v_look, _ = lookup(s_look)

            dir_to_anc = targ_xy - s[:, :2]
            dir_norm = dir_to_anc / (jnp.linalg.norm(dir_to_anc, axis=-1, keepdims=True) + 1e-6)
            step_prog = jnp.sum((ns[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            look_prog = jnp.sum((s_look[:, :2] - s[:, :2]) * dir_norm, axis=-1) / p90_step
            dv = jnp.where(is_guided_romi, v_look - v_s, v_ns - v_s)

            prog_metric = jnp.where(is_guided_romi, 0.5 * step_prog + 0.12 * look_prog, step_prog)
            good = (step_prog > 0.05) & (dv > 0.05)
            w_pi = jnp.where(good, jnp.minimum(jnp.exp(prog_metric * 2.2 + dv * 0.4), 50.0), 0.01)

            irb = jax.random.randint(rbk, (128,), 0, N_rb)
            srb, arb = rb_s_valid[irb], rb_a_valid[irb]
            wrb = jnp.where(is_guided_romi & (ob_dim == 2), 15.0, 0.0)

            # Standard ROMI mixes in reverse BC rollout transitions (s_t, a_t) directly
            irom = jax.random.randint(romk, (128,), 0, N_rom)
            srom, arom = rom_s_flat[irom], rom_a_flat[irom]
            wrom = jnp.where(~is_guided_romi, 1.5, 0.0)

            def loss_fn(p):
                v_pred, a_pred = pv_mod.apply({"params": p}, s)
                _, arb_pred = pv_mod.apply({"params": p}, srb)
                _, arom_pred = pv_mod.apply({"params": p}, srom)
                v_loss = jnp.square(v_pred - v_s).mean()
                pi_loss = (
                    (w_pi[:, None] * jnp.square(a_pred - a)).mean()
                    + (wrb * jnp.square(arb_pred - arb)).mean()
                    + (wrom * jnp.square(arom_pred - arom)).mean()
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

    romi_arms = [
        ("standard_romi_h15", v_std, targ_std, False),
        ("principled_romi_h15", v_prin, targ_prin, True),
    ]

    json_path = f"/tmp/maze_stitch_results/{key}.json"
    with open(json_path, "r") as f:
        existing = json.load(f)

    existing["stitch_diagnostics"]["standard_romi_h15"] = romi_std_diag
    existing["stitch_diagnostics"]["principled_romi_h15"] = romi_prin_diag

    n_chunks = 50 if "humanoid" in key else 36
    for arm_name, av, atarg, is_g in romi_arms:
        rng, ak = jax.random.split(rng)
        st = TrainState.create(
            apply_fn=pv_mod.apply,
            params=pv_mod.init(ak, gpu_real["s"][:2])["params"],
            tx=optax.adam(3e-4),
        )
        for _ in range(n_chunks):
            st, rng, pl = train_romi_arm_500(st, rng, gpu_real, av, atarg, jnp.bool_(is_g))
        jax.block_until_ready(pl)
        res = evaluate_arm(st.params)
        existing["arm_results"][arm_name] = res
        print(f"[{key}] {arm_name:24s} -> success={res['success_rate_pct']:5.1f}%, min_dist={res['mean_min_dist']:5.2f}m", flush=True)

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[{key}] ROMI DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
