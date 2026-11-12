import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class ReverseRolloutsQBetaAgent(flax.struct.PyTreeNode):
    """reverse-rollouts-qbeta: Dataset-Anchored Reverse Rollouts Sampled from
       q_beta(s) = d_D(s) exp(beta V(s)) / E_{d_D}[exp(beta V(s))].

    Combines the explicit empirical dataset state density d_D(s) (2-D spatial visitation kernel
    with kinematic typicality over proprioceptive dimensions) and exponential value tilting
    exp(beta * V_psi(s)) normalized by the empirical partition function E_{d_D}[exp(beta V(s))].
    Every sampled anchor state is strictly an in-dataset state s_i in D_real.
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
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

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
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.model_loss(batch, grad_params, rng)

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(model=new_model, rng=new_rng), info

    def compute_discrepancy(self, next_observations, actions):
        dist = self.model.select('reverse_dynamics')(next_observations, actions)
        means = dist.mean()
        diffs = means[:, None, :, :] - means[None, :, :, :]
        pairwise_l2 = jnp.linalg.norm(diffs, axis=-1)
        return pairwise_l2.max(axis=(0, 1))

    @jax.jit
    def select_elites(self, holdout_batch):
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
        dyn_rng, member_rng = jax.random.split(seed)
        prev_obs = self.model.select('reverse_dynamics')(next_observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (prev_obs.shape[1],), 0, self.elites.shape[0])]
        return prev_obs[member, jnp.arange(prev_obs.shape[1])]

    @jax.jit
    def eval_v_pi_chunk(self, obs, value_goals=None, actor_goals=None):
        """Evaluate V_pi(s) = min(Q_1(s, pi(s)), Q_2(s, pi(s))) on a chunk of replay buffer states."""
        if self.config.get('use_q_pi', True):
            pi_actions = jnp.clip(self.network.select('actor')(obs, actor_goals).mode(), -1.0, 1.0)
            q1, q2 = self.network.select('target_critic')(obs, value_goals, pi_actions)
            return jnp.minimum(q1, q2)
        return self.network.select('value')(obs, value_goals)

    def sample_full_buffer_batch(self, train_dataset, mb_batch_size):
        """Evaluate q_beta(s) propto d_D(s) exp(beta * V_pi(s)) across ALL states s in the replay buffer."""
        ds_dict = train_dataset.dataset._dict
        valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
        if 'next_observations' in ds_dict:
            all_next_obs = ds_dict['next_observations'][valid_idxs]
        else:
            next_idxs = np.minimum(valid_idxs + 1, train_dataset.size - 1)
            all_next_obs = ds_dict['observations'][next_idxs]

        total_n, ob_dim = all_next_obs.shape
        grid_bins = int(self.config['high_freq_grid_bins'])
        xy = all_next_obs[:, :2]
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
        if ob_dim > 2 and float(self.config['kinematic_density_weight']) > 0.0:
            rest = all_next_obs[:, 2:]
            rest_mean = np.mean(rest, axis=0, keepdims=True)
            rest_std = np.maximum(np.std(rest, axis=0, keepdims=True), 1e-4)
            z_sq = np.mean(np.square((rest - rest_mean) / rest_std), axis=-1)
            log_d_D = log_d_spatial - float(self.config['kinematic_density_weight']) * z_sq
        else:
            log_d_D = log_d_spatial

        chunk_size = 100000
        scores = []
        for start in range(0, total_n, chunk_size):
            end = min(start + chunk_size, total_n)
            v_chunk = np.asarray(self.eval_v_pi_chunk(all_next_obs[start:end]))
            scores.append(v_chunk)
        all_v = np.concatenate(scores, axis=0).astype(np.float64)

        if int(self.network.step) > 0:
            if self.config.get('normalize_value_logits', True):
                v_mean = np.mean(all_v)
                v_std = max(np.std(all_v), 1e-5)
                v_scaled = np.clip((all_v - v_mean) / v_std, -5.0, 5.0)
            else:
                v_scaled = np.clip(all_v - np.max(all_v), -20.0, 0.0)
            beta_v = float(self.config['q_beta']) * v_scaled
            logits = log_d_D + beta_v
        else:
            logits = log_d_D

        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        chosen_local = np.random.choice(total_n, size=mb_batch_size, replace=True, p=probs)
        selected_idxs = valid_idxs[chosen_local]
        return train_dataset.sample(mb_batch_size, idxs=selected_idxs)

    def sample_qbeta_anchors(self, batch, rng):
        """Sample in-dataset anchors from q_beta(s) = d_D(s) exp(beta V_pi(s)) / E_{d_D}[exp(beta V_pi(s))]."""
        obs = batch['next_observations']
        batch_size, ob_dim = obs.shape
        grid_bins = int(self.config['high_freq_grid_bins'])
        num_anchors = max(1, int(batch_size * float(self.config['anchor_fraction'])))
        value_goals = batch['value_goals'] if ('value_goals' in batch and batch['value_goals'] is not None) else None
        actor_goals = batch['actor_goals'] if ('actor_goals' in batch and batch['actor_goals'] is not None) else None

        if self.config.get('full_buffer_qbeta', True):
            return jnp.arange(num_anchors, dtype=jnp.int32)

        # 1. Empirical state-space density d_D(s) via smoothed 2-D spatial kernel + kinematic density
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

        if ob_dim > 2 and self.config['kinematic_density_weight'] > 0.0:
            rest = obs[:, 2:]
            rest_mean = jnp.mean(rest, axis=0, keepdims=True)
            rest_std = jnp.maximum(jnp.std(rest, axis=0, keepdims=True), 1e-4)
            z_sq = jnp.mean(jnp.square((rest - rest_mean) / rest_std), axis=-1)
            log_d_D = log_d_spatial - self.config['kinematic_density_weight'] * z_sq
        else:
            log_d_D = log_d_spatial

        # 2. Value tilting exp(beta * V_pi(s)) normalized by E_{d_D}[exp(beta * V_pi(s))]
        v = self.eval_v_pi_chunk(obs, value_goals, actor_goals)
        if self.config.get('normalize_value_logits', True):
            v_mean = jnp.mean(v)
            v_std = jnp.maximum(jnp.std(v), 1e-5)
            v_scaled = jnp.clip((v - v_mean) / v_std, -5.0, 5.0)
        else:
            v_scaled = jnp.clip(v - jnp.max(v), -20.0, 0.0)

        beta_v = float(self.config['q_beta']) * v_scaled
        # Partition function log E_{d_D}[exp(beta V(s))]
        log_Z_beta = jax.scipy.special.logsumexp(beta_v) - jnp.log(float(batch_size))
        log_tilt = jnp.where(self.network.step > 0, beta_v - log_Z_beta, jnp.zeros_like(beta_v))

        # Combine log d_D(s) + beta V(s) - log E_{d_D}[exp(beta V(s))]
        log_q_beta = log_d_D + log_tilt
        sampled_idx = jax.random.categorical(rng, logits=log_q_beta, shape=(num_anchors,))
        return sampled_idx

    @jax.jit
    def rollout(self, batch, rng):
        """Start reverse rollouts from in-dataset anchors sampled ~ q_beta(s) and roll until USAD triggers."""
        max_H = int(self.config['max_rollout_steps'])
        rng, sample_rng = jax.random.split(rng)
        sampled_idx = self.sample_qbeta_anchors(batch, sample_rng)

        anchor_ob = batch['next_observations'][sampled_idx]
        num_anchors = anchor_ob.shape[0]
        value_goals = batch['value_goals'][sampled_idx] if ('value_goals' in batch and batch['value_goals'] is not None) else None
        actor_goals = batch['actor_goals'][sampled_idx] if ('actor_goals' in batch and batch['actor_goals'] is not None) else None

        def reverse_rollout_step(carry, _):
            cur_next_ob, alive, halted, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            rev_act_dist = self.model.select('reverse_actor')(cur_next_ob, goals=value_goals)
            rev_action = jnp.clip(rev_act_dist.sample(seed=action_rng), -1.0, 1.0)
            raw_prev_ob = self.sample_reverse_dynamics(cur_next_ob, rev_action, dyn_rng)

            disc = self.compute_discrepancy(cur_next_ob, rev_action)
            if self.config.get('use_usad', True):
                unknown = (disc > self.usad_thresh).astype(jnp.float32)
            else:
                unknown = jnp.zeros_like(disc)
            new_halted = jnp.maximum(halted, unknown)

            prev_ob = jnp.where(new_halted[:, None] > 0.5, cur_next_ob, raw_prev_ob)
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

            return (prev_ob, next_alive, new_halted, rng), {
                'observations': prev_ob,
                'actions': rev_action,
                'next_observations': cur_next_ob,
                'rewards': rewards,
                'masks': masks,
                'rollout_valid': rollout_valid,
            }

        init_carry = (
            anchor_ob,
            jnp.ones((num_anchors,), dtype=jnp.float32),
            jnp.zeros((num_anchors,), dtype=jnp.float32),
            rng,
        )
        _, rollout_data = jax.lax.scan(reverse_rollout_step, init_carry, None, length=max_H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(value_goals[None], max_H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(actor_goals[None], max_H, axis=0)

        return {k: v.reshape((max_H * num_anchors,) + v.shape[2:]) for k, v in rollout_data.items()}

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

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return value_loss + critic_loss + actor_loss, info

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
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
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
            agent_name='reverse_rollouts_qbeta',
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
            rollout_steps=20,
            max_rollout_steps=20,
            anchor_fraction=0.25,
            q_beta=3.0,
            use_q_pi=True,
            full_buffer_qbeta=True,
            high_freq_grid_bins=64,
            kinematic_density_weight=0.1,
            normalize_value_logits=True,
            use_usad=True,
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
