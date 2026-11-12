import jax
import jax.numpy as jnp


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
