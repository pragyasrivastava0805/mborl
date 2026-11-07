import glob
import json
import os
import pickle
import sys
import time

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

sys.path.insert(0, '/home/rchitale_google_com/mborl')

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import ACGCDataset, CDataset, Dataset, GCDataset, HGCDataset
from utils.flax_utils import restore_agent

dataset_class_dict = {
    'CDataset': CDataset,
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'ACGCDataset': ACGCDataset,
}

TASKS = [
    # Family A: Standard Reward-Based Benchmark Tasks (singletask, scalar reward, restored checkpoints)
    {
        'id': 'scene_singletask',
        'family': 'Standard Reward-Based (Single-Task)',
        'label': 'scene-singletask-task2',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_scene-play-singletask-task2-v0/sd000_20261106_044030_combo-scene-task2-20261028T020500p1-s0-c11',
        'ckpt_epoch': 600000,
    },
    {
        'id': 'cube_double_singletask',
        'family': 'Standard Reward-Based (Single-Task)',
        'label': 'cube-double-singletask-task2',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_cube-double-play-singletask-task2-v0/sd000_20261106_032110_combo-cube-double-task2-20261028T020500p1-s0-c11',
        'ckpt_epoch': 800000,
    },
    {
        'id': 'cube_single_singletask',
        'family': 'Standard Reward-Based (Single-Task)',
        'label': 'cube-single-singletask-task2',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_cube-single-play-singletask-task2-v0/sd000_20261106_025245_combo-cube-single-task2-20261028T020500p1-s0-c11',
        'ckpt_epoch': 900000,
    },
    {
        'id': 'puzzle_4x4_singletask',
        'family': 'Standard Reward-Based (Single-Task)',
        'label': 'puzzle-4x4-singletask-task5',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_puzzle-4x4-play-singletask-task5-v0/sd000_20261106_025240_combo-puzzle-4x4-task5-20261028T020500p1-s0-c10',
        'ckpt_epoch': 800000,
    },
    # Family B: Large-Scale, Long-Horizon Tasks (100M / Goal-Conditioned OracleRep)
    {
        'id': 'cube_double_100m',
        'family': 'Large-Scale Long-Horizon (100M / Multi-Goal)',
        'label': 'cube-double-100m-oraclerep',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_cube-double-play-oraclerep-v0/sd000_20261027_015148_combo-cube-double-20261026T184550-s0-c0',
        'ckpt_epoch': None,
    },
    {
        'id': 'cube_octuple_100m',
        'family': 'Large-Scale Long-Horizon (100M / Multi-Goal)',
        'label': 'cube-octuple-100m-oraclerep',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_cube-octuple-play-oraclerep-v0',
        'ckpt_epoch': None,
    },
    {
        'id': 'puzzle_4x5_100m',
        'family': 'Large-Scale Long-Horizon (100M / Multi-Goal)',
        'label': 'puzzle-4x5-100m-oraclerep',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_puzzle-4x5-play-oraclerep-v0/sd000_20261029_185846_combo-puzzle-4x5-20261026T184550-s0-c8',
        'ckpt_epoch': None,
    },
    {
        'id': 'humanoidmaze_giant_100m',
        'family': 'Large-Scale Long-Horizon (100M / Multi-Goal)',
        'label': 'humanoidmaze-giant-100m',
        'run_dir': '/home/rchitale_google_com/mborl/exp/scalembrl/combo_ablation_humanoidmaze-giant-navigate-oraclerep-v0',
        'ckpt_epoch': None,
    },
]


def resolve_run_dir(path):
    if os.path.exists(os.path.join(path, 'flags.json')):
        return path
    subdirs = sorted(glob.glob(os.path.join(path, 'sd000_*')))
    for sd in subdirs:
        if os.path.exists(os.path.join(sd, 'flags.json')) and os.path.exists(os.path.join(sd, 'scale.pkl')):
            return sd
    raise RuntimeError(f'No valid run dir under {path}')


def evaluate_task(task_spec, H=25, N_rollouts=256):
    t0 = time.time()
    run_dir = resolve_run_dir(task_spec['run_dir'])
    with open(os.path.join(run_dir, 'flags.json'), 'r') as f:
        flags_dict = json.load(f)
    with open(os.path.join(run_dir, 'scale.pkl'), 'rb') as f:
        scale = pickle.load(f)

    env_name = flags_dict['env_name']
    dataset_dir = flags_dict['dataset_dir']
    if dataset_dir is None:
        dataset_path = None
    elif dataset_dir.endswith('.npz'):
        dataset_path = dataset_dir
    else:
        files = [f for f in sorted(glob.glob(f'{dataset_dir}/*.npz')) if '-val.npz' not in f]
        dataset_path = files[0]

    print(f"\n[{task_spec['label']}] Loading dataset from {dataset_path} (env={env_name})...", flush=True)
    _, raw_train, raw_val = make_env_and_datasets(env_name, dataset_path=dataset_path, dataset_only=False)

    config = ml_collections.ConfigDict(flags_dict['agent'])
    ds_cls = dataset_class_dict[config['dataset_class']]
    train_dataset = ds_cls(Dataset.create(scale=scale, **raw_train), config)
    val_dataset = ds_cls(Dataset.create(scale=scale, **raw_val), config)

    example_batch = train_dataset.sample(1)
    config['ob_dim'] = int(example_batch['observations'].shape[-1])
    config['action_dim'] = int(example_batch['actions'].shape[-1])

    agent_cls = agents[config['agent_name']]
    agent = agent_cls.create(0, example_batch, config)

    ckpt_epoch = task_spec['ckpt_epoch']
    if ckpt_epoch is not None and os.path.exists(os.path.join(run_dir, f'params_{ckpt_epoch}.pkl')):
        agent = restore_agent(agent, run_dir, ckpt_epoch)
    else:
        # Check if any params_*.pkl exists in run_dir
        ckpts = sorted(glob.glob(os.path.join(run_dir, 'params_*.pkl')))
        if ckpts:
            ep = int(os.path.basename(ckpts[-1]).split('_')[1].split('.')[0])
            agent = restore_agent(agent, run_dir, ep)
        else:
            print(f"[{task_spec['label']}] Training fast 3000-step model + 3000-step policy on GPU...", flush=True)
            for step in range(3000):
                batch = train_dataset.sample(256)
                agent, _ = agent.update_model(batch)
            holdout_batch = val_dataset.sample(1000)
            agent, _ = agent.select_elites(holdout_batch)
            # Short rollout-assisted policy training
            config_short = copy.deepcopy(agent.config) if hasattr(agent, 'config') else config
            for step in range(2500):
                real_b = train_dataset.sample(256)
                if step % 250 == 0:
                    rng_r = jax.random.PRNGKey(step + 100)
                    init_b = train_dataset.sample(1024)
                    fake_pool = agent.rollout(init_b, rng_r)
                idx_f = np.random.randint(0, fake_pool['observations'].shape[0], size=256)
                fake_b = {k: v[idx_f] for k, v in fake_pool.items()}
                agent, _ = agent.update(real_b, fake_b)

    # Extract episode IDs for the dataset
    base_ds = train_dataset.dataset
    obs_all = np.asarray(base_ds['observations'], dtype=np.float32)
    M_total, ob_dim = obs_all.shape
    if 'terminals' in base_ds and np.sum(base_ds['terminals']) > 0:
        terms = np.asarray(base_ds['terminals'] > 0, dtype=np.int32)
    elif 'valids' in base_ds:
        terms = np.asarray(base_ds['valids'] == 0, dtype=np.int32)
    else:
        terms = np.zeros(M_total, dtype=np.int32)
        terms[999::1000] = 1
    ep_ids_all = np.concatenate([[0], np.cumsum(terms)[:-1]])

    # Build a dense reference pool of 80,000 states (first 80 episodes of 1,000 steps)
    pool_size = min(80000, M_total)
    pool_obs = jnp.asarray(obs_all[:pool_size])
    pool_ep_ids = jnp.asarray(ep_ids_all[:pool_size])

    # Sample N_rollouts starting states from the first 80 episodes (excluding last 30 steps of each episode)
    valid_mask = (np.arange(pool_size) % 1000) < 900
    valid_indices = np.nonzero(valid_mask)[0]
    rng_np = np.random.RandomState(42)
    start_idxs = rng_np.choice(valid_indices, size=N_rollouts, replace=False)

    start_batch = train_dataset.sample(N_rollouts, idxs=start_idxs)
    init_obs = jnp.asarray(start_batch['observations'])
    start_ep_ids = jnp.asarray(ep_ids_all[start_idxs])

    # Goals for critic & actor evaluation
    has_goals = ('value_goals' in start_batch) and (start_batch['value_goals'] is not None)
    if has_goals:
        # Use a fixed set of task goals (from distant states in different trajectories so stitching is tested toward a common target manifold)
        goal_idxs = rng_np.choice(valid_indices, size=N_rollouts, replace=False)
        if 'oracle_reps' in base_ds:
            fixed_goals = jnp.asarray(base_ds['oracle_reps'][goal_idxs])
        else:
            fixed_goals = jnp.asarray(obs_all[goal_idxs])
        # Compute value of pool states w.r.t. the median goal (or per-rollout goal)
        ref_goal = fixed_goals[0:1]
        ref_goals_pool = jnp.repeat(ref_goal, pool_size, axis=0)
        actor_goals = jnp.repeat(ref_goal, N_rollouts, axis=0)
        value_goals = actor_goals
    else:
        ref_goals_pool = None
        actor_goals = None
        value_goals = None

    # Compute Critic Value V(s) = Q(s, pi(s)) for all pool states in chunks of 4000
    @jax.jit
    def eval_v(ob_chunk, g_chunk):
        dist = agent.network.select('actor')(ob_chunk, goals=g_chunk, temperature=0.0)
        act = jnp.clip(dist.mode(), -1.0, 1.0)
        qs = agent.network.select('critic')(ob_chunk, goals=g_chunk, actions=act)
        return agent.aggregate(qs, agent.config['q_agg'])

    pool_v_list = []
    chunk_sz = 4000
    for i in range(0, pool_size, chunk_sz):
        g_c = ref_goals_pool[i:i+chunk_sz] if has_goals else None
        pool_v_list.append(np.asarray(eval_v(pool_obs[i:i+chunk_sz], g_c)))
    pool_v = jnp.asarray(np.concatenate(pool_v_list, axis=0))
    v_std = float(jnp.std(pool_v)) + 1e-6

    # Compute in-dataset baseline nearest-neighbor distance (excluding same episode) to set support threshold
    # Roll out H=25 steps under (1) On-Policy pi_theta and (2) Uniform Random pi_rand
    def run_rollout_trace(policy_mode):
        ob = init_obs
        rng = jax.random.PRNGKey(123 if policy_mode == 'policy' else 999)
        obs_trace = [ob]
        v_trace = [eval_v(ob, value_goals)]
        for h in range(1, H + 1):
            rng, a_rng, d_rng = jax.random.split(rng, 3)
            if policy_mode == 'uniform':
                act = jax.random.uniform(a_rng, (N_rollouts, config['action_dim']), minval=-1.0, maxval=1.0)
            else:
                act, _ = agent.sample_actions(ob, actor_goals, seed=a_rng, temperature=0.0, mode='train')
            ob = agent.sample_dynamics(ob, act, d_rng)
            obs_trace.append(ob)
            v_trace.append(eval_v(ob, value_goals))
        return jnp.stack(obs_trace, axis=0), jnp.stack(v_trace, axis=0)  # (H+1, N, D), (H+1, N)

    obs_pi, v_pi = run_rollout_trace('policy')
    obs_rand, v_rand = run_rollout_trace('uniform')

    # Focus on physical configuration coordinates (first half of state = qpos / object positions, plus full state)
    pos_dim = max(4, ob_dim // 2)

    @jax.jit
    def compute_step_metrics(s_h, v_h, s_0, v_0):
        # s_h: (N, D), pool_obs: (P, D)
        # Compute RMS distance in full state and position subspace
        # Pairwise squared distance: ||s_h - pool||^2 / D
        diff_full = jnp.sum(jnp.square(s_h[:, None, :] - pool_obs[None, :, :]), axis=-1) / ob_dim
        d_full = jnp.sqrt(jnp.maximum(diff_full, 0.0))

        diff_pos = jnp.sum(jnp.square(s_h[:, None, :pos_dim] - pool_obs[None, :, :pos_dim]), axis=-1) / pos_dim
        d_pos = jnp.sqrt(jnp.maximum(diff_pos, 0.0))

        same_ep_mask = (pool_ep_ids[None, :] == start_ep_ids[:, None])
        other_ep_mask = ~same_ep_mask
        # Target manifold: other episodes with strictly higher value V(s) >= V(s_0) + 0.25 * std(V)
        target_mask = other_ep_mask & (pool_v[None, :] >= (v_0[:, None] + 0.25 * v_std))

        # Overall nearest neighbor (excluding exact self at h=0 only if looking at other, but for d_real include all)
        nn_idx = jnp.argmin(d_full, axis=-1)
        d_real = jnp.min(d_full, axis=-1)
        nn_ep = pool_ep_ids[nn_idx]
        nn_v = pool_v[nn_idx]
        ep_switched = (nn_ep != start_ep_ids).astype(jnp.float32)

        # Same-episode distance vs Target-episode distance (in configuration space d_pos)
        d_same = jnp.min(jnp.where(same_ep_mask, d_pos, 1e6), axis=-1)
        d_target = jnp.min(jnp.where(target_mask, d_pos, 1e6), axis=-1)
        # Bridge distance: min(d_same, d_target) -> 0 at h=0, rises in gap, drops when landing on target episode!
        d_bridge = jnp.minimum(d_same, d_target)

        return d_real, d_bridge, d_same, d_target, ep_switched, (v_h - v_0) / v_std, (nn_v - v_0) / v_std

    def summarize_trace(obs_tr, v_tr):
        d_real_curve, d_bridge_curve, d_same_curve, d_target_curve = [], [], [], []
        switch_curve, dv_critic_curve, dv_nn_curve = [], [], []
        s0 = obs_tr[0]
        v0 = v_tr[0]
        for h in range(H + 1):
            dr, db, ds, dt, sw, dvc, dvnn = compute_step_metrics(obs_tr[h], v_tr[h], s0, v0)
            d_real_curve.append(float(jnp.mean(dr)))
            d_bridge_curve.append(float(jnp.mean(db)))
            d_same_curve.append(float(jnp.mean(ds)))
            d_target_curve.append(float(jnp.mean(dt)))
            switch_curve.append(float(jnp.mean(sw) * 100.0))
            dv_critic_curve.append(float(jnp.mean(dvc)))
            dv_nn_curve.append(float(jnp.mean(dvnn)))
        return {
            'd_real': d_real_curve,
            'd_bridge': d_bridge_curve,
            'd_same': d_same_curve,
            'd_target': d_target_curve,
            'ep_switch_pct': switch_curve,
            'dv_critic_norm': dv_critic_curve,
            'dv_nn_norm': dv_nn_curve,
        }

    res_pi = summarize_trace(obs_pi, v_pi)
    res_rand = summarize_trace(obs_rand, v_rand)
    elapsed = time.time() - t0
    print(
        f"[{task_spec['label']}] Done in {elapsed:.1f}s | "
        f"Policy EpSwitch H=5: {res_pi['ep_switch_pct'][5]:.1f}%, H=15: {res_pi['ep_switch_pct'][15]:.1f}%, H=25: {res_pi['ep_switch_pct'][25]:.1f}% | "
        f"d_bridge H=0: {res_pi['d_bridge'][0]:.3f}, Peak: {max(res_pi['d_bridge'][:12]):.3f}, H=20: {res_pi['d_bridge'][20]:.3f} | "
        f"d_real H=5: {res_pi['d_real'][5]:.3f}, H=25: {res_pi['d_real'][25]:.3f}",
        flush=True,
    )
    return {
        'id': task_spec['id'],
        'family': task_spec['family'],
        'label': task_spec['label'],
        'ob_dim': config['ob_dim'],
        'action_dim': config['action_dim'],
        'policy': res_pi,
        'uniform': res_rand,
    }


def main():
    results = []
    for t in TASKS:
        try:
            res = evaluate_task(t, H=25, N_rollouts=256)
            results.append(res)
        except Exception as e:
            import traceback
            print(f"ERROR on {t['label']}: {e}", flush=True)
            traceback.print_exc()

    out_path = '/tmp/test2_all_envs_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results to {out_path}", flush=True)


if __name__ == '__main__':
    import copy
    main()
