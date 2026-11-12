import jax
import jax.numpy as jnp
import numpy as np


def compute_spatial_kinematic_log_density(obs, grid_bins=64, kinematic_density_weight=0.1):
    """Compute empirical state-space log-density log d_D(s) via 2-D spatial kernel + kinematic typicality."""
    ob_dim = obs.shape[-1]
    xy = obs[:, :2]
    xy_min = jnp.min(xy, axis=0, keepdims=True)
    xy_max = jnp.max(xy, axis=0, keepdims=True)
    xy_norm = (xy - xy_min) / jnp.maximum(xy_max - xy_min, 1e-5)
    coords = jnp.clip(jnp.floor(xy_norm * grid_bins).astype(jnp.int32), 0, grid_bins - 1)
    gx, gy = coords[:, 0], coords[:, 1]
    flat_cell = gx * grid_bins + gy

    counts_flat = jnp.bincount(flat_cell, length=grid_bins * grid_bins).astype(jnp.float32)
    counts_grid = counts_flat.reshape((grid_bins, grid_bins))
    padded = jnp.pad(counts_grid, ((1, 1), (1, 1)), mode='edge')
    smoothed_grid = (
        4.0 * padded[1:-1, 1:-1]
        + 2.0 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        + 1.0 * (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
    ) / 16.0
    log_d_spatial = jnp.log(jnp.maximum(smoothed_grid[gx, gy], 1e-5))

    if ob_dim > 2 and kinematic_density_weight > 0.0:
        rest = obs[:, 2:]
        rest_mean = jnp.mean(rest, axis=0, keepdims=True)
        rest_std = jnp.maximum(jnp.std(rest, axis=0, keepdims=True), 1e-4)
        z_sq = jnp.mean(jnp.square((rest - rest_mean) / rest_std), axis=-1)
        return log_d_spatial - kinematic_density_weight * z_sq
    return log_d_spatial


def select_anchor_indices(obs, v_scores, step, config, rng):
    """Select in-dataset anchor indices according to config['sampling_mode'] in:
    {'uniform', 'exp', 'qbeta', 'argmax', 'high_freq'}.
    """
    batch_size = obs.shape[0]
    mode = config.get('sampling_mode', 'uniform')
    if mode == 'uniform':
        return jnp.arange(batch_size, dtype=jnp.int32)

    anchor_fraction = float(config.get('anchor_fraction', 0.25))
    num_anchors = max(1, int(batch_size * anchor_fraction))

    if config.get('full_buffer_sampling', True):
        # Batch is already pre-selected from the full replay buffer via sample_full_buffer_batch
        return jnp.arange(num_anchors, dtype=jnp.int32)

    if mode == 'high_freq':
        log_d = compute_spatial_kinematic_log_density(
            obs,
            grid_bins=int(config.get('high_freq_grid_bins', 64)),
            kinematic_density_weight=float(config.get('kinematic_density_weight', 0.1)),
        )
        return jnp.argsort(-log_d)[:num_anchors]

    v_mean = jnp.mean(v_scores)
    v_std = jnp.maximum(jnp.std(v_scores), 1e-5)
    v_scaled = jnp.clip((v_scores - v_mean) / v_std, -5.0, 5.0)

    if mode == 'exp':
        tau_exp = float(config.get('tau_exp', 3.0))
        logits = jnp.where(step > 0, tau_exp * v_scaled, jnp.zeros_like(v_scaled))
        return jax.random.categorical(rng, logits=logits, shape=(num_anchors,))

    if mode == 'qbeta':
        log_d = compute_spatial_kinematic_log_density(
            obs,
            grid_bins=int(config.get('high_freq_grid_bins', 64)),
            kinematic_density_weight=float(config.get('kinematic_density_weight', 0.1)),
        )
        beta_v = float(config.get('q_beta', 3.0)) * v_scaled
        log_Z = jax.scipy.special.logsumexp(beta_v) - jnp.log(float(batch_size))
        log_tilt = jnp.where(step > 0, beta_v - log_Z, jnp.zeros_like(beta_v))
        return jax.random.categorical(rng, logits=log_d + log_tilt, shape=(num_anchors,))

    if mode == 'argmax':
        top_frac = float(config.get('argmax_top_fraction', 0.01))
        top_k = max(1, int(batch_size * top_frac))
        top_pool = jnp.argsort(-v_scores)[:top_k]
        repeats = (num_anchors + top_k - 1) // top_k
        return jnp.tile(top_pool, (repeats,))[:num_anchors]

    raise ValueError(f'Unknown sampling_mode: {mode}')


def sample_full_buffer_anchors_generic(agent, train_dataset, mb_batch_size, use_next_obs=True):
    """Evaluate V_pi(s) across ALL states s in the replay buffer and sample in-buffer transitions."""
    mode = agent.config.get('sampling_mode', 'uniform')
    if mode == 'uniform':
        return train_dataset.sample(mb_batch_size)

    ds_dict = train_dataset.dataset._dict
    valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
    if use_next_obs:
        if 'next_observations' in ds_dict:
            all_obs = ds_dict['next_observations'][valid_idxs]
        else:
            next_idxs = np.minimum(valid_idxs + 1, train_dataset.size - 1)
            all_obs = ds_dict['observations'][next_idxs]
    else:
        all_obs = ds_dict['observations'][valid_idxs]

    total_n, ob_dim = all_obs.shape

    def compute_np_log_density():
        grid_bins = int(agent.config.get('high_freq_grid_bins', 64))
        kin_w = float(agent.config.get('kinematic_density_weight', 0.1))
        xy = all_obs[:, :2]
        xy_min = np.min(xy, axis=0, keepdims=True)
        xy_max = np.max(xy, axis=0, keepdims=True)
        xy_norm = (xy - xy_min) / np.maximum(xy_max - xy_min, 1e-5)
        coords = np.clip(np.floor(xy_norm * grid_bins).astype(np.int32), 0, grid_bins - 1)
        gx, gy = coords[:, 0], coords[:, 1]
        flat_cell = gx * grid_bins + gy
        counts_flat = np.bincount(flat_cell, minlength=grid_bins * grid_bins).astype(np.float64)
        counts_grid = counts_flat.reshape((grid_bins, grid_bins))
        padded = np.pad(counts_grid, ((1, 1), (1, 1)), mode='edge')
        smoothed_grid = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
            + 1.0 * (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
        ) / 16.0
        log_d_spatial = np.log(np.maximum(smoothed_grid[gx, gy], 1e-5))
        if ob_dim > 2 and kin_w > 0.0:
            rest = all_obs[:, 2:]
            rest_mean = np.mean(rest, axis=0, keepdims=True)
            rest_std = np.maximum(np.std(rest, axis=0, keepdims=True), 1e-4)
            z_sq = np.mean(np.square((rest - rest_mean) / rest_std), axis=-1)
            return log_d_spatial - kin_w * z_sq
        return log_d_spatial

    if mode == 'high_freq':
        log_d = compute_np_log_density()
        top_k = min(total_n, max(1, int(mb_batch_size * float(agent.config.get('anchor_fraction', 0.25)))))
        top_local = np.argpartition(-log_d, kth=min(top_k - 1, total_n - 1))[:top_k]
        top_local = top_local[np.argsort(-log_d[top_local])]
        chosen_local = np.resize(top_local, (mb_batch_size,))
        return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen_local])

    chunk_size = 100000
    scores = []
    for start in range(0, total_n, chunk_size):
        end = min(start + chunk_size, total_n)
        v_chunk = np.asarray(agent.eval_v_pi_chunk(all_obs[start:end]))
        scores.append(v_chunk)
    all_v = np.concatenate(scores, axis=0).astype(np.float64)

    if int(agent.network.step) <= 0:
        chosen_local = np.random.randint(0, total_n, size=mb_batch_size)
        return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen_local])

    if mode == 'argmax':
        top_frac = float(agent.config.get('argmax_top_fraction', 0.01))
        top_k = max(1, min(total_n, int(mb_batch_size * top_frac)))
        top_local = np.argpartition(-all_v, kth=min(top_k - 1, total_n - 1))[:top_k]
        top_local = top_local[np.argsort(-all_v[top_local])]
        chosen_local = np.resize(top_local, (mb_batch_size,))
        return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen_local])

    v_mean = np.mean(all_v)
    v_std = max(np.std(all_v), 1e-5)
    v_scaled = np.clip((all_v - v_mean) / v_std, -5.0, 5.0)

    if mode == 'exp':
        logits = float(agent.config.get('tau_exp', 3.0)) * v_scaled
    elif mode == 'qbeta':
        log_d = compute_np_log_density()
        logits = log_d + float(agent.config.get('q_beta', 3.0)) * v_scaled
    else:
        raise ValueError(f'Unknown sampling_mode: {mode}')

    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    chosen_local = np.random.choice(total_n, size=mb_batch_size, replace=True, p=probs)
    return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen_local])
