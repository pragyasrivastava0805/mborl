import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class ReverseRolloutHighFreqAgent(flax.struct.PyTreeNode):
    """reverse-rollout-high-freq: High-Frequency Hub-Anchored Reverse Rollouts with MoREL-Style Adaptive Rollout-Until-USAD.

    Implements two key mechanisms on top of Reverse-MoREL + IQL:
      1. High-Frequency State Anchor Selection (`select_high_freq_anchors`):
         - Determines the highest-frequency states in D_real by combining a 2-D spatial kernel visitation
           histogram over (x, y) maze coordinates (`grid_bins x grid_bins` with 3x3 neighborhood smoothing)
           and cell-conditional kinematic typicality over proprioceptive dimensions (for AntMaze/HumanoidMaze).
         - Starts reverse model rollouts EXCLUSIVELY from the top `high_freq_quantile` (default 25%)
           highest-frequency states (`s_anchor in D_high_freq subset D_real`), which correspond to busy
           corridor junctions and multi-segment hubs where 4-10 dataset trajectories overlap.
      2. MoREL-Style Adaptive Rollout-Until-USAD (No Fixed Short Horizon H=5):
         - Instead of truncating at a fixed short horizon H=5, each reverse rollout continues step-by-step
           (`k = 1, 2, ..., max_rollout_steps`, default 20) as long as ensemble discrepancy satisfies
           `disc(s_cur, a_tilde) <= usad_thresh` and non-backtracking holds.
         - As soon as `disc(s_cur, a_tilde) > usad_thresh` (USAD triggers), that trajectory halts
           (`rollout_valid = 0.0` for step k and all subsequent steps), and `main_mbpo.py` retains only
           the pre-USAD steps (`rollout_valid > 0`).
      3. Exact IQL policy optimization (`update` on `D_real U D_rev_high_freq`):
         - Supports both 1-step IQL critic target `r + gamma * mask * V_psi(s_next)` and max-horizon
           hybrid target `max(r + gamma * V_psi(s_next), R^(k) + gamma^k * V_psi(s_anchor))`.
    """

    rng: Any
    network: Any
    model: Any
    elites: Any
    usad_thresh: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the IQL expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    # ------------------------------------------------------------------ #
    # Step 1: Reverse model pretraining & USAD calibration
    # ------------------------------------------------------------------ #

    def model_loss(self, batch, grad_params, rng):
        goals = batch['value_goals'] if 'value_goals' in batch else None

        rev_dist = self.model.select('reverse_dynamics')(
            batch['next_observations'], batch['actions'], params=grad_params
        )
        dynamics_loss = -rev_dist.log_prob(batch['observations'][None]).mean()
        dynamics_mse = jnp.square(rev_dist.mean() - batch['observations'][None]).mean()

        rev_act_dist = self.model.select('reverse_actor')(
            batch['next_observations'], goals=goals, params=grad_params
        )
        rev_actor_loss = -rev_act_dist.log_prob(jnp.clip(batch['actions'], -1.0 + 1e-5, 1.0 - 1e-5)).mean()

        succ_logit = self.model.select('success')(
            batch['observations'], goals, batch['actions'], params=grad_params
        )
        success = jax.nn.sigmoid(succ_logit)
        success_target = 1 - batch['masks']
        success_loss = self.bce_loss(succ_logit, success_target).mean()

        rewards = self.model.select('reward')(
            batch['observations'], goals, batch['actions'], params=grad_params
        )
        reward_loss = jnp.square(rewards - batch['rewards']).mean()

        loss = dynamics_loss + rev_actor_loss + success_loss + reward_loss
        return loss, {
            'dynamics/dynamics_loss': dynamics_loss,
            'dynamics/dynamics_mse': dynamics_mse,
            'dynamics/reverse_actor_loss': rev_actor_loss,
            'success/success_loss': success_loss,
            'success/success_mean': success.mean(),
            'reward/reward_loss': reward_loss,
            'reward/reward_mean': rewards.mean(),
        }

    @jax.jit
    def update_model(self, batch):
        """Take one gradient step on the reverse world model during pretraining."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.model_loss(batch, grad_params, rng)

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(model=new_model, rng=new_rng), info

    def compute_discrepancy(self, next_observations, actions):
        """Compute max pairwise L2 reverse ensemble mean discrepancy U_USAD(s_{t+1}, a_t)."""
        dist = self.model.select('reverse_dynamics')(next_observations, actions)
        means = dist.mean()  # (num_dynamics, batch_size, ob_dim)
        diffs = means[:, None, :, :] - means[None, :, :, :]  # (M, M, B, D)
        pairwise_l2 = jnp.linalg.norm(diffs, axis=-1)  # (M, M, B)
        return pairwise_l2.max(axis=(0, 1))  # (B,)

    @jax.jit
    def select_elites(self, holdout_batch):
        """Select elite reverse dynamics members and calibrate USAD threshold on held-out real transitions."""
        dist = self.model.select('reverse_dynamics')(
            holdout_batch['next_observations'], holdout_batch['actions']
        )
        member_mse = jnp.square(dist.mean() - holdout_batch['observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)

        disc = self.compute_discrepancy(holdout_batch['next_observations'], holdout_batch['actions'])
        calibrated_thresh = jnp.mean(disc) + self.config['morel_thresh_std_mult'] * jnp.std(disc)
        usad_thresh = jnp.where(
            self.config['auto_calibrate_thresh'], calibrated_thresh, self.config['morel_thresh']
        )

        _, model_info = self.model_loss(holdout_batch, None, self.rng)
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info['holdout/elite_mse'] = member_mse[elites].mean()
        info['holdout/usad_thresh'] = usad_thresh
        return self.replace(elites=elites, usad_thresh=usad_thresh), info

    def sample_reverse_dynamics(self, next_observations, actions, seed):
        """Sample predecessor states s_t from a uniformly chosen elite reverse dynamics member."""
        dyn_rng, member_rng = jax.random.split(seed)
        prev_obs = self.model.select('reverse_dynamics')(next_observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (prev_obs.shape[1],), 0, self.elites.shape[0])]
        return prev_obs[member, jnp.arange(prev_obs.shape[1])]

    # ------------------------------------------------------------------ #
    # Step 2: Conditional VAE P(s | V(s) >= t) Anchor Generation &
    #         Adaptive Rollout-Until-USAD
    # ------------------------------------------------------------------ #

    def cvae_loss(self, real_batch, grad_params, rng):
        """Fit Conditional VAE p_omega(s | z, c) and q_phi(z | s, c) where c = I[V_psi(s) >= t]."""
        s = real_batch['next_observations']
        value_goals = real_batch.get('value_goals', None)

        # Evaluate learned value function V_psi(s) (stop-gradient so CVAE fits density without altering V_psi)
        v = sg(self.network.select('value')(s, value_goals))
        thresh_t = jnp.quantile(v, self.config['cvae_value_quantile'])
        c = (v >= thresh_t).astype(jnp.float32)[:, None]
        cond = jnp.concatenate([c, value_goals], axis=-1) if value_goals is not None else c

        post_dist = self.network.select('cvae_encoder')(s, goals=cond, params=grad_params)
        z = post_dist.sample(seed=rng)
        recon_dist = self.network.select('cvae_decoder')(z, goals=cond, params=grad_params)
        s_hat = recon_dist.mode()

        recon_mse = jnp.mean(jnp.square(s_hat - s), axis=-1)
        mu_z = post_dist.loc
        std_z = jnp.maximum(post_dist.scale_diag, 1e-5)
        kl_div = -0.5 * jnp.sum(1.0 + 2.0 * jnp.log(std_z) - jnp.square(mu_z) - jnp.square(std_z), axis=-1)

        # Upweight c=1 high-value slice so p_omega(s | z, c=1) models P(s | V(s) >= t) sharply
        sample_weight = 1.0 + self.config['cvae_pos_weight'] * c.squeeze(-1)
        loss = jnp.mean(sample_weight * (recon_mse + self.config['cvae_kl_weight'] * kl_div))
        return loss, {
            'cvae_loss': loss,
            'cvae_recon_mse': recon_mse.mean(),
            'cvae_kl': kl_div.mean(),
            'cvae_val_thresh': thresh_t,
            'cvae_pos_frac': c.mean(),
        }

    def sample_cvae_anchors(self, batch, rng):
        """Generate anchors with the Conditional VAE P(s | V(s) >= t), reject large projection errors
        (||s_gen - s_nn||_2 > eps_proj calibrated on held-out high-value dataset states), resample,
        and replace each generated anchor with its closest suitable high-value dataset state.
        """
        obs = batch['next_observations']
        batch_size, ob_dim = obs.shape
        num_anchors = max(1, int(batch_size * float(self.config['high_freq_quantile'])))
        value_goals = batch['value_goals'] if ('value_goals' in batch and batch['value_goals'] is not None) else None
        actor_goals = batch['actor_goals'] if ('actor_goals' in batch and batch['actor_goals'] is not None) else None

        v = self.network.select('value')(obs, value_goals)
        # Suitable dataset states: high-value subset {s in D_real : V_psi(s) >= t}
        top_idx = jnp.argsort(-v)[:num_anchors]
        real_high_v_obs = obs[top_idx]
        anchor_value_goals = value_goals[top_idx] if value_goals is not None else None
        anchor_actor_goals = actor_goals[top_idx] if actor_goals is not None else None

        # Condition strictly on c = 1 (i.e. V_psi(s) >= t)
        c_ones = jnp.ones((num_anchors, 1), dtype=jnp.float32)
        cond = jnp.concatenate([c_ones, anchor_value_goals], axis=-1) if anchor_value_goals is not None else c_ones
        latent_dim = int(self.config['cvae_latent_dim'])

        def generate_cvae_draw(draw_rng):
            prior_rng, post_rng, mix_rng = jax.random.split(draw_rng, 3)
            z_prior = jax.random.normal(prior_rng, shape=(num_anchors, latent_dim))
            post_dist = self.network.select('cvae_encoder')(real_high_v_obs, goals=cond)
            z_post = post_dist.sample(seed=post_rng)
            use_prior = jax.random.bernoulli(mix_rng, p=float(self.config['cvae_prior_prob']), shape=(num_anchors, 1))
            z = jnp.where(use_prior, z_prior, z_post)
            decoded = self.network.select('cvae_decoder')(z, goals=cond).mode()
            obs_min = jnp.min(obs, axis=0, keepdims=True)
            obs_max = jnp.max(obs, axis=0, keepdims=True)
            return jnp.clip(decoded, obs_min, obs_max)

        if not self.config.get('cvae_nn_proj', True):
            decoded_obs = generate_cvae_draw(rng)
            warmup_steps = int(self.config.get('cvae_warmup_steps', 10000))
            anchor_ob = jnp.where(self.network.step >= warmup_steps, decoded_obs, real_high_v_obs)
            return anchor_ob, anchor_value_goals, anchor_actor_goals, top_idx

        # Nearest-neighbor projection onto suitable high-value dataset candidates with rejection resampling
        rng, split_rng, r1_rng, r2_rng, r3_rng, fallback_rng = jax.random.split(rng, 6)
        perm = jax.random.permutation(split_rng, num_anchors)
        cand_size = min(int(self.config.get('cvae_nn_cand_size', 2048)), max(2, (num_anchors * 3) // 4))
        hold_size = min(int(self.config.get('cvae_nn_holdout_size', 512)), max(1, num_anchors - cand_size))

        cand_local_idx = perm[:cand_size]
        hold_local_idx = perm[cand_size : cand_size + hold_size]
        cand_obs = real_high_v_obs[cand_local_idx]  # (cand_size, ob_dim)
        hold_obs = real_high_v_obs[hold_local_idx]  # (hold_size, ob_dim)
        cand_sq_norm = jnp.sum(jnp.square(cand_obs), axis=-1)[None, :]  # (1, cand_size)

        # Calibrate eps_proj using nearest-neighbor distances from held-out high-value dataset states to candidate set
        hold_dists_sq = jnp.maximum(
            jnp.sum(jnp.square(hold_obs), axis=-1, keepdims=True) + cand_sq_norm - 2.0 * (hold_obs @ cand_obs.T),
            0.0,
        )
        hold_nn_dist = jnp.sqrt(jnp.min(hold_dists_sq, axis=-1))
        eps_proj = jnp.quantile(hold_nn_dist, float(self.config.get('cvae_proj_quantile', 0.95)))

        def project_draw(draw_rng):
            gen_obs = generate_cvae_draw(draw_rng)
            dists_sq = jnp.maximum(
                jnp.sum(jnp.square(gen_obs), axis=-1, keepdims=True) + cand_sq_norm - 2.0 * (gen_obs @ cand_obs.T),
                0.0,
            )
            nn_cand_idx = jnp.argmin(dists_sq, axis=-1)
            nn_dist = jnp.sqrt(jnp.min(dists_sq, axis=-1))
            return nn_cand_idx, nn_dist

        # Round 1: initial CVAE sample + NN projection
        best_cand_idx, best_dist = project_draw(r1_rng)
        accepted = best_dist <= eps_proj

        # Round 2: resample rejected slots (where ||s_gen - s_nn||_2 > eps_proj)
        r2_cand_idx, r2_dist = project_draw(r2_rng)
        use_r2 = (~accepted) & (r2_dist < best_dist)
        best_cand_idx = jnp.where(use_r2, r2_cand_idx, best_cand_idx)
        best_dist = jnp.where(use_r2, r2_dist, best_dist)
        accepted = best_dist <= eps_proj

        # Round 3: second resample for any still-rejected slots
        r3_cand_idx, r3_dist = project_draw(r3_rng)
        use_r3 = (~accepted) & (r3_dist < best_dist)
        best_cand_idx = jnp.where(use_r3, r3_cand_idx, best_cand_idx)
        best_dist = jnp.where(use_r3, r3_dist, best_dist)
        accepted = best_dist <= eps_proj

        # Final guard: if any slot still exceeds eps_proj after 3 draws, discard it and sample a valid candidate
        fallback_cand_idx = jax.random.randint(fallback_rng, shape=(num_anchors,), minval=0, maxval=cand_size)
        final_cand_idx = jnp.where(accepted, best_cand_idx, fallback_cand_idx)

        proj_global_idx = top_idx[cand_local_idx[final_cand_idx]]
        warmup_steps = int(self.config.get('cvae_warmup_steps', 10000))
        final_global_idx = jnp.where(self.network.step >= warmup_steps, proj_global_idx, top_idx)

        anchor_ob = obs[final_global_idx]
        proj_value_goals = value_goals[final_global_idx] if value_goals is not None else None
        proj_actor_goals = actor_goals[final_global_idx] if actor_goals is not None else None
        return anchor_ob, proj_value_goals, proj_actor_goals, final_global_idx

    def select_high_freq_anchors(self, batch):
        """Fallback spatial/kinematic hub selector when use_cvae=False."""
        obs = batch['next_observations']
        batch_size, ob_dim = obs.shape
        grid_bins = int(self.config['high_freq_grid_bins'])
        high_freq_quantile = float(self.config['high_freq_quantile'])
        num_anchors = max(1, int(batch_size * high_freq_quantile))

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
        spatial_freq_score = jnp.log1p(smoothed_grid[gx, gy])

        if ob_dim > 2 and self.config['kinematic_density_weight'] > 0.0:
            rest = obs[:, 2:]
            rest_mean = jnp.mean(rest, axis=0, keepdims=True)
            rest_std = jnp.maximum(jnp.std(rest, axis=0, keepdims=True), 1e-4)
            z_sq = jnp.mean(jnp.square((rest - rest_mean) / rest_std), axis=-1)
            freq_score = spatial_freq_score - self.config['kinematic_density_weight'] * z_sq
        else:
            freq_score = spatial_freq_score

        top_indices = jnp.argsort(-freq_score)[:num_anchors]
        return top_indices

    @jax.jit
    def rollout(self, batch, rng):
        """Sample anchor states from Conditional VAE P(s | V(s) >= t) and roll backward until USAD triggers."""
        max_H = int(self.config['max_rollout_steps'])
        rng, cvae_rng = jax.random.split(rng)

        if self.config.get('use_cvae', True):
            anchor_ob, value_goals, actor_goals, _ = self.sample_cvae_anchors(batch, cvae_rng)
        else:
            top_idx = self.select_high_freq_anchors(batch)
            anchor_ob = batch['next_observations'][top_idx]
            value_goals = batch['value_goals'][top_idx] if ('value_goals' in batch and batch['value_goals'] is not None) else None
            actor_goals = batch['actor_goals'][top_idx] if ('actor_goals' in batch and batch['actor_goals'] is not None) else None

        num_anchors = anchor_ob.shape[0]

        def reverse_rollout_step(carry, _):
            cur_next_ob, prev_dist_from_anchor, alive, halted, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            rev_act_dist = self.model.select('reverse_actor')(cur_next_ob, goals=value_goals)
            rev_action = jnp.clip(rev_act_dist.sample(seed=action_rng), -1.0, 1.0)
            raw_prev_ob = self.sample_reverse_dynamics(cur_next_ob, rev_action, dyn_rng)

            # MoREL USAD stopping condition: halt trajectory as soon as ensemble discrepancy exceeds usad_thresh
            disc = self.compute_discrepancy(cur_next_ob, rev_action)
            if self.config.get('use_usad', True):
                unknown = (disc > self.usad_thresh).astype(jnp.float32)
            else:
                unknown = jnp.zeros_like(disc)

            # Optional non-backtracking check so multi-step reverse rollouts expand outward from s_anchor
            new_dist_from_anchor = jnp.linalg.norm(raw_prev_ob[:, :2] - anchor_ob[:, :2], axis=-1)
            if self.config['enforce_non_backtracking']:
                backtracked = (new_dist_from_anchor < prev_dist_from_anchor - 0.05).astype(jnp.float32)
                unknown = jnp.maximum(unknown, backtracked)

            new_halted = jnp.maximum(halted, unknown)

            prev_ob = jnp.where(new_halted[:, None] > 0.5, cur_next_ob, raw_prev_ob)
            updated_dist = jnp.where(new_halted > 0.5, prev_dist_from_anchor, new_dist_from_anchor)

            succ_logit = self.model.select('success')(prev_ob, value_goals, rev_action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                rewards = self.model.select('reward')(prev_ob, None, rev_action)
                if self.config['gc_negative']:
                    rewards = jnp.clip(rewards, -1.0, 0.0)

            masks = 1.0 - success
            rollout_valid = alive * (1.0 - new_halted)
            next_alive = rollout_valid * masks

            return (prev_ob, updated_dist, next_alive, new_halted, rng), {
                'observations': prev_ob,
                'actions': rev_action,
                'next_observations': cur_next_ob,
                'rewards': rewards,
                'masks': masks,
                'rollout_valid': rollout_valid,
            }

        init_carry = (
            anchor_ob,
            jnp.zeros((num_anchors,), dtype=jnp.float32),
            jnp.ones((num_anchors,), dtype=jnp.float32),
            jnp.zeros((num_anchors,), dtype=jnp.float32),
            rng,
        )
        _, rollout_data = jax.lax.scan(reverse_rollout_step, init_carry, None, length=max_H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(value_goals[None], max_H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(actor_goals[None], max_H, axis=0)

        return {k: v.reshape((max_H * num_anchors,) + v.shape[2:]) for k, v in rollout_data.items()}

    # ------------------------------------------------------------------ #
    # Step 3: Exact IQL Policy Optimization on D_real U D_rev_high_freq
    # ------------------------------------------------------------------ #

    def preprocess(self, real_batch, fake_batch):
        batch = {
            'observations': jnp.concatenate([real_batch['observations'], fake_batch['observations']], axis=0),
            'actions': jnp.concatenate([real_batch['actions'], fake_batch['actions']], axis=0),
            'next_observations': jnp.concatenate(
                [real_batch['next_observations'], fake_batch['next_observations']], axis=0
            ),
            'rewards': jnp.concatenate([real_batch['rewards'], fake_batch['rewards']], axis=0),
            'masks': jnp.concatenate([real_batch['masks'], fake_batch['masks']], axis=0),
        }
        if 'value_goals' not in real_batch or real_batch['value_goals'] is None:
            batch['value_goals'] = None
            batch['actor_goals'] = None
        else:
            batch['value_goals'] = jnp.concatenate([real_batch['value_goals'], fake_batch['value_goals']], axis=0)
            batch['actor_goals'] = jnp.concatenate([real_batch['actor_goals'], fake_batch['actor_goals']], axis=0)
        return batch

    def value_loss(self, batch, grad_params):
        """Compute the exact IQL expectile value loss on D_real U D_rev_high_freq."""
        value_goals = batch.get('value_goals', None)
        q1, q2 = self.network.select('target_critic')(batch['observations'], value_goals, batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], value_goals, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the 1-step IQL critic loss bootstrapping from V(next_observations)."""
        value_goals = batch.get('value_goals', None)
        next_v = self.network.select('value')(batch['next_observations'], value_goals)
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        if self.config['gc_negative']:
            q_target = jnp.clip(q_target, -1.0 / (1.0 - self.config['discount']), 0.0)
        q_target = sg(q_target)

        q1, q2 = self.network.select('critic')(
            batch['observations'], value_goals, batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_target.mean(),
            'q_max': q_target.max(),
            'q_min': q_target.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the exact IQL actor loss (AWR or DDPG+BC)."""
        actor_goals = batch.get('actor_goals', None)
        if self.config['actor_loss'] == 'awr':
            v = self.network.select('value')(batch['observations'], actor_goals)
            q1, q2 = self.network.select('critic')(batch['observations'], actor_goals, batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], actor_goals, params=grad_params)
            log_prob = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_prob).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'ddpgbc':
            dist = self.network.select('actor')(batch['observations'], actor_goals, params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], actor_goals, q_actions)
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        batch = self.preprocess(real_batch, fake_batch)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng, cvae_rng = jax.random.split(rng, 3)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        cvae_loss, cvae_info = self.cvae_loss(real_batch, grad_params, cvae_rng)
        for k, v in cvae_info.items():
            info[f'cvae/{k}'] = v

        return value_loss + critic_loss + actor_loss + cvae_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, real_batch, fake_batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(real_batch, fake_batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, real_batch, fake_batch, rng=None):
        goals = real_batch['actor_goals'] if 'actor_goals' in real_batch else None
        actions, _ = self.sample_actions(real_batch['observations'], goals, seed=rng, mode='train')
        return {'mse': jnp.mean((actions - real_batch['actions']) ** 2)}

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0, mode='test'):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = jnp.clip(dist.sample(seed=seed), -1, 1)
        return actions, {}

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, model_rng = jax.random.split(rng, 3)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals'] if 'actor_goals' in example_batch else None
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        cvae_latent_dim = int(config.get('cvae_latent_dim', 16))
        ex_c = jnp.zeros((ex_observations.shape[0], 1), dtype=jnp.float32)
        ex_cond = jnp.concatenate([ex_c, ex_goals], axis=-1) if ex_goals is not None else ex_c
        ex_latents = jnp.zeros((ex_observations.shape[0], cvae_latent_dim), dtype=jnp.float32)
        cvae_encoder_def = GCActor(
            hidden_dims=config.get('cvae_hidden_dims', (512, 512, 512)),
            action_dim=cvae_latent_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=0.01,
        )
        cvae_decoder_def = GCActor(
            hidden_dims=config.get('cvae_hidden_dims', (512, 512, 512)),
            action_dim=ob_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=0.01,
        )
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            cvae_encoder=(cvae_encoder_def, (ex_observations, ex_cond)),
            cvae_decoder=(cvae_decoder_def, (ex_latents, ex_cond)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        reverse_dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_dynamics'],
            stochastic=True,
        )
        reverse_actor_def = GCActor(
            hidden_dims=config['dynamics_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=0.01,
        )
        success_def = GCValue(
            hidden_dims=config['dynamics_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        reward_def = GCValue(
            hidden_dims=config['dynamics_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        model_info = dict(
            reverse_dynamics=(reverse_dynamics_def, (ex_observations, ex_actions)),
            reverse_actor=(reverse_actor_def, (ex_observations, ex_goals)),
            success=(success_def, (ex_observations, ex_goals, ex_actions)),
            reward=(reward_def, (ex_observations, ex_goals, ex_actions)),
        )
        model_def = ModuleDict({k: v[0] for k, v in model_info.items()})
        model_params = model_def.init(model_rng, **{k: v[1] for k, v in model_info.items()})['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=config['lr']))

        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)
        usad_thresh = jnp.asarray(config['morel_thresh'], dtype=jnp.float32)

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            model=model,
            elites=elites,
            usad_thresh=usad_thresh,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='reverse_rollout_high_freq',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            dynamics_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.003,
            const_std=True,
            num_dynamics=7,
            num_elites=5,
            model_train_steps=100000,
            model_holdout_size=1000,
            # Adaptive rollout-until-USAD (MoREL style, no fixed H=5):
            # top 25% Conditional VAE P(s | V(s) >= t) anchors x max_rollout_steps=20
            rollout_steps=20,
            max_rollout_steps=20,
            use_usad=True,
            use_cvae=True,
            cvae_nn_proj=True,
            cvae_nn_cand_size=2048,
            cvae_nn_holdout_size=512,
            cvae_proj_quantile=0.95,
            cvae_warmup_steps=10000,
            cvae_latent_dim=16,
            cvae_hidden_dims=(512, 512, 512),
            cvae_value_quantile=0.75,
            cvae_kl_weight=0.01,
            cvae_pos_weight=2.0,
            cvae_prior_prob=0.5,
            high_freq_quantile=0.25,
            high_freq_grid_bins=64,
            kinematic_density_weight=0.1,
            enforce_non_backtracking=False,
            morel_thresh=0.5,
            morel_thresh_std_mult=1.5,
            auto_calibrate_thresh=True,
            ob_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
    return config
