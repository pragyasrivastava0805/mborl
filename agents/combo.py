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
from utils.networks import Dynamics, GCActor, GCValue, LogParam

sg = lambda x: jax.lax.stop_gradient(x)


def _concat(*parts):
    parts = [p for p in parts if p is not None]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


def log_partition(rand_qs, pi_qs, pi_log_probs, action_dim):
    """Estimate log int_{[-1, 1]^d} exp Q(s, a) da from uniform and policy proposals.

    Args:
        rand_qs: Q values of uniform actions, shape (..., num_states, n).
        pi_qs: Q values of policy actions, shape (..., num_states, n).
        pi_log_probs: Log densities of the policy actions, shape (num_states, n).
        action_dim: Action dimension.

    Returns the log of the average of the 2n importance weights exp(Q) / q(a), shape (..., num_states).
    """
    rand_log_density = -action_dim * jnp.log(2.0)
    weighted = jnp.concatenate([rand_qs - rand_log_density, pi_qs - pi_log_probs], axis=-1)
    return jax.scipy.special.logsumexp(weighted, axis=-1) - jnp.log(weighted.shape[-1])


class COMBOAgent(flax.struct.PyTreeNode):
    """COMBO (Yu et al., 2021) agent.

    The learned model (dynamics ensemble, success and reward heads) lives in `model`, which has its own optimizer and is
    only updated by `update_model` during the pretraining phase. `update` trains the actor, critics and temperature in
    `network` and never touches `model`, so the model and its optimizer state stay frozen during policy learning.
    """

    rng: Any
    network: Any
    model: Any
    elites: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        """Compute the BCE loss."""
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    # ------------------------------------------------------------------ #
    # Model pretraining
    # ------------------------------------------------------------------ #

    def model_loss(self, batch, grad_params, rng):
        """Compute the dynamics (Gaussian NLL), success (BCE) and reward (MSE) losses."""
        goals = batch['value_goals'] if 'value_goals' in batch else None

        dist = self.model.select('dynamics')(batch['observations'], batch['actions'], params=grad_params)
        dynamics_loss = -dist.log_prob(batch['next_observations'][None]).mean()
        dynamics_mse = jnp.square(dist.mean() - batch['next_observations'][None]).mean()

        succ_logit = self.model.select('success')(batch['observations'], goals, batch['actions'], params=grad_params)
        success = jax.nn.sigmoid(succ_logit)
        success_target = 1 - batch['masks']
        success_loss = self.bce_loss(succ_logit, success_target).mean()

        rewards = self.model.select('reward')(batch['observations'], goals, batch['actions'], params=grad_params)
        reward_loss = jnp.square(rewards - batch['rewards']).mean()

        loss = dynamics_loss + success_loss + reward_loss
        return loss, {
            'dynamics/dynamics_loss': dynamics_loss,
            'dynamics/dynamics_mse': dynamics_mse,
            'success/success_loss': success_loss,
            'success/success_mean': success.mean(),
            'success/accuracy_0.5': jnp.mean((success > 0.5) == (success_target > 0.5)),
            'success/data_success_mean': success_target.mean(),
            'reward/reward_loss': reward_loss,
            'reward/reward_mean': rewards.mean(),
            'reward/data_reward_mean': batch['rewards'].mean(),
        }

    @jax.jit
    def update_model(self, batch):
        """Take one gradient step on the model. Only used during the pretraining phase."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.model_loss(batch, grad_params, rng)

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(model=new_model, rng=new_rng), info

    @jax.jit
    def select_elites(self, holdout_batch):
        """Select the members with the lowest held-out next-state MSE as elites."""
        dist = self.model.select('dynamics')(holdout_batch['observations'], holdout_batch['actions'])
        member_mse = jnp.square(dist.mean() - holdout_batch['next_observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)

        _, model_info = self.model_loss(holdout_batch, None, self.rng)
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info.update({f'holdout/member_mse_{i}': member_mse[i] for i in range(self.config['num_dynamics'])})
        info['holdout/elite_mse'] = member_mse[elites].mean()
        return self.replace(elites=elites), info

    def sample_dynamics(self, observations, actions, seed):
        """Sample next states, using a uniformly chosen elite member for every transition."""
        dyn_rng, member_rng = jax.random.split(seed)
        next_obs = self.model.select('dynamics')(observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (next_obs.shape[1],), 0, self.elites.shape[0])]
        return next_obs[member, jnp.arange(next_obs.shape[1])]

    # ------------------------------------------------------------------ #
    # Model rollouts
    # ------------------------------------------------------------------ #

    @jax.jit
    def rollout(self, batch, rng):
        """Roll out the frozen model from dataset states (mirrors `MOBILEAgent.rollout`).

        Goal-conditioned rollouts condition the rollout policy on `value_goals`, hold both goals fixed across the
        horizon, and label transitions with the learned success classifier. Every field is flattened from (H, B, ...)
        to (H * B, ...). `rollout_valid` marks transitions to keep; it is all ones unless `rollout_stop_on_terminal`.
        """
        H = self.config['rollout_steps']
        init_ob = batch['observations']
        batch_size = init_ob.shape[0]
        value_goals = batch['value_goals'] if 'value_goals' in batch else None

        def rollout_step(carry, _):
            ob, alive, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            if self.config['rollout_policy'] == 'uniform':
                action = jax.random.uniform(action_rng, (batch_size, self.config['action_dim']), minval=-1.0, maxval=1.0)
            else:
                action, _ = self.sample_actions(ob, value_goals, action_rng, mode='train')
            next_ob = self.sample_dynamics(ob, action, dyn_rng)

            succ_logit = self.model.select('success')(ob, value_goals, action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                rewards = self.model.select('reward')(ob, None, action)

            next_alive = alive * (1.0 - success) if self.config['rollout_stop_on_terminal'] else alive
            return (next_ob, next_alive, rng), {
                'observations': ob,
                'actions': action,
                'next_observations': next_ob,
                'rewards': rewards,
                'masks': 1.0 - success,
                'rollout_valid': alive,
            }

        init_carry = (init_ob, jnp.ones((batch_size,)), rng)
        _, rollout_data = jax.lax.scan(rollout_step, init_carry, None, length=H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(batch['value_goals'][None], H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(batch['actor_goals'][None], H, axis=0)

        return {k: v.reshape((H * batch_size,) + v.shape[2:]) for k, v in rollout_data.items()}

    # ------------------------------------------------------------------ #
    # Policy optimization
    # ------------------------------------------------------------------ #

    def aggregate(self, qs, agg):
        return qs.min(axis=0) if agg == 'min' else qs.mean(axis=0)

    @staticmethod
    def batch_sizes(config):
        """Return (real, model) sample counts of one d_f batch, plus the real count for E_D when f = 0.

        A d_f batch holds round(f * B) real and B - round(f * B) model transitions, as in the paper. With f = 0 the batch
        has no real transitions, so a separate batch of B real transitions is used for E_D[Q] only.
        """
        n_real = int(round(config['real_ratio'] * config['batch_size']))
        n_model = config['batch_size'] - n_real
        return n_real, n_model, n_real if n_real > 0 else config['batch_size']

    def split_batches(self, real_batch, fake_batch):
        """Return the real part of the d_f batch (None when f = 0) and the model part (None when f = 1)."""
        df_real = real_batch if self.config['real_ratio'] > 0 else None
        df_fake = fake_batch if self.config['real_ratio'] < 1 else None
        return df_real, df_fake

    def critic_loss(self, real_batch, fake_batch, grad_params, rng):
        """Compute the COMBO critic loss (Eq. 2 / Eq. 29) for every critic in the ensemble.

        L = sum_i 0.5 * E_{d_f}[(Q_i - y)^2] + beta * (E_{s~rho}[log int exp Q_i(s, a) da] - E_D[Q_i(s, a)])

        d_f is the pooled batch of `real_batch` and `fake_batch` (fractions f and 1 - f). rho(s) is d_f
        (`conservative_states='mixture'`) or the model states ('model'). E_D always uses `real_batch`. With f = 1 there are
        no model terms and the loss equals CQL(H) with uniform and pi(.|s) proposals.

        The log-partition is estimated with `num_samples` uniform and `num_samples` current-policy proposals per state:
            log int exp Q da ~= logsumexp_j(Q(s, a_j) - log q_j(a_j)) - log(2 * num_samples),
        which is the log of an unbiased estimate of the integral. Proposals and their densities are stop-gradiented.
        """
        gc = 'value_goals' in real_batch
        df_real, df_fake = self.split_batches(real_batch, fake_batch)
        n_df_real = 0 if df_real is None else df_real['observations'].shape[0]
        get = lambda k: _concat(*(None if b is None else b[k] for b in (df_real, df_fake)))
        obs, actions, next_obs = get('observations'), get('actions'), get('next_observations')
        rewards, masks = get('rewards'), get('masks')
        value_goals = get('value_goals') if gc else None

        next_rng, rand_rng, pi_rng = jax.random.split(rng, 3)

        # TD target.
        next_dist = self.network.select('actor')(next_obs, goals=value_goals)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_rng)
        next_qs = self.network.select('target_critic')(next_obs, goals=value_goals, actions=next_actions)
        next_q = self.aggregate(next_qs, self.config['q_agg'])
        if self.config['backup_entropy']:
            next_q = next_q - self.network.select('lam')() * next_log_probs
        target_q = rewards + self.config['discount'] * masks * next_q
        if gc:
            if self.config['gc_negative']:
                target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
            else:
                target_q = jnp.clip(target_q, 0, 1)
        target_q = sg(target_q)

        qs = self.network.select('critic')(obs, goals=value_goals, actions=actions, params=grad_params)
        td_loss = 0.5 * jnp.square(qs - target_q).mean(axis=1)

        # Conservative term: push down on rho(s) x soft-max actions, push up on dataset pairs.
        if self.config['conservative_states'] == 'model':
            cons_obs = df_fake['observations']
            cons_goals = df_fake['value_goals'] if gc else None
        else:
            cons_obs = obs
            cons_goals = value_goals
        num_states = cons_obs.shape[0]
        n = self.config['num_samples']
        action_dim = self.config['action_dim']
        rep_obs = jnp.repeat(cons_obs, n, axis=0)
        rep_goals = jnp.repeat(cons_goals, n, axis=0) if gc else None

        rand_actions = jax.random.uniform(rand_rng, (num_states * n, action_dim), minval=-1.0, maxval=1.0)
        pi_dist = self.network.select('actor')(rep_obs, goals=rep_goals)
        pi_actions, pi_log_probs = pi_dist.sample_and_log_prob(seed=pi_rng)
        pi_actions, pi_log_probs = sg(pi_actions), sg(pi_log_probs)

        rand_qs = self.network.select('critic')(rep_obs, goals=rep_goals, actions=rand_actions, params=grad_params)
        pi_qs = self.network.select('critic')(rep_obs, goals=rep_goals, actions=pi_actions, params=grad_params)
        num_qs = rand_qs.shape[0]
        log_z = log_partition(
            rand_qs.reshape(num_qs, num_states, n),
            pi_qs.reshape(num_qs, num_states, n),
            pi_log_probs.reshape(num_states, n),
            action_dim,
        )
        cons_pos = log_z.mean(axis=1)
        if n_df_real > 0:
            q_data = qs[:, :n_df_real]
        else:
            q_data = self.network.select('critic')(
                real_batch['observations'], goals=real_batch['value_goals'] if gc else None,
                actions=real_batch['actions'], params=grad_params,
            )
        cons_neg = q_data.mean(axis=1)
        regularizer = cons_pos - cons_neg

        critic_loss = (td_loss + self.config['beta'] * regularizer).sum()

        info = {
            'critic_loss': critic_loss,
            'td_loss': td_loss.mean(),
            'conservative_loss': self.config['beta'] * regularizer.mean(),
            'regularizer': regularizer.mean(),
            'cons_pos': cons_pos.mean(),
            'cons_neg': cons_neg.mean(),
            'q_data_mean': q_data.mean(),
            'q_rand_mean': rand_qs.mean(),
            'q_pi_mean': pi_qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_mean': target_q.mean(),
            'target_max': target_q.max(),
            'target_min': target_q.min(),
        }
        if df_fake is not None:
            info['q_model_mean'] = qs[:, n_df_real:].mean()
        for i in range(num_qs):
            info[f'q{i}_data_mean'] = q_data[i].mean()
            info[f'q{i}_regularizer'] = regularizer[i]
        return critic_loss, info

    def actor_loss(self, real_batch, fake_batch, grad_params, rng):
        """Compute the SAC actor and temperature losses on the d_f batch."""
        gc = 'actor_goals' in real_batch
        df = [b for b in self.split_batches(real_batch, fake_batch) if b is not None]
        obs = _concat(*(b['observations'] for b in df))
        actor_goals = _concat(*(b['actor_goals'] for b in df)) if gc else None

        dist = self.network.select('actor')(obs, goals=actor_goals, params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        qs = self.network.select('critic')(obs, goals=actor_goals, actions=actions)
        q = self.aggregate(qs, self.config['actor_q_agg'])
        actor_loss = (self.network.select('lam')() * log_probs - q).mean()

        lam = self.network.select('lam')(params=grad_params)
        entropy = -sg(log_probs).mean()
        lam_loss = lam * (entropy - self.config['target_entropy'])

        return actor_loss + lam_loss, {
            'actor_loss': actor_loss,
            'lam_loss': lam_loss,
            'lam': lam,
            'entropy': entropy,
            'std': dist._distribution.stddev().mean() if self.config['tanh_squash'] else dist.stddev().mean(),
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        """Compute the policy-learning loss. The model is not part of this loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        actor_rng, critic_rng = jax.random.split(rng)

        critic_loss, critic_info = self.critic_loss(real_batch, fake_batch, grad_params, critic_rng)
        info.update({f'critic/{k}': v for k, v in critic_info.items()})

        actor_loss, actor_info = self.actor_loss(real_batch, fake_batch, grad_params, actor_rng)
        info.update({f'actor/{k}': v for k, v in actor_info.items()})

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, real_batch, fake_batch):
        """Update the actor, critics and temperature. The model is left untouched."""
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
        metrics = {'mse': jnp.mean((actions - real_batch['actions']) ** 2)}
        if fake_batch is not None:
            metrics['fake_reward_mean'] = fake_batch['rewards'].mean()
            metrics['fake_mask_mean'] = fake_batch['masks'].mean()
        return metrics

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        mode='test',
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions, {}

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        assert config['value_loss'] == 'squared', 'COMBO only supports the squared TD loss.'
        assert config['conservative_states'] in ('mixture', 'model')
        assert config['rollout_policy'] in ('policy', 'uniform')
        assert 0.0 <= config['real_ratio'] <= 1.0, 'real_ratio (f) must be in [0, 1].'
        assert not (config['conservative_states'] == 'model' and config['real_ratio'] == 1.0), (
            "conservative_states='model' needs model data (real_ratio < 1)."
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng, model_rng = jax.random.split(rng, 3)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals'] if 'actor_goals' in example_batch else None
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Policy networks.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            lam=(LogParam(), ()),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        network_params['modules_target_critic'] = network_params['modules_critic']

        lr_labels = {
            'modules_actor': 'actor',
            'modules_critic': 'critic',
            'modules_lam': 'lam',
            'modules_target_critic': 'frozen',
        }
        network_tx = optax.multi_transform(
            {
                'actor': optax.adam(learning_rate=config['actor_lr']),
                'critic': optax.adam(learning_rate=config['critic_lr']),
                'lam': optax.adam(learning_rate=config['alpha_lr']),
                'frozen': optax.set_to_zero(),
            },
            lambda params: {k: jax.tree_util.tree_map(lambda _: lr_labels[k], v) for k, v in params.items()},
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Model networks.
        dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['model_layer_norm'],
            num_ensembles=config['num_dynamics'],
            stochastic=True,
        )
        success_def = GCValue(
            hidden_dims=config['dynamics_hidden_dims'],
            layer_norm=config['model_layer_norm'],
            num_ensembles=1,
        )
        reward_def = GCValue(
            hidden_dims=config['dynamics_hidden_dims'],
            layer_norm=config['model_layer_norm'],
            num_ensembles=1,
        )
        model_info = dict(
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
            success=(success_def, (ex_observations, ex_goals, ex_actions)),
            reward=(reward_def, (ex_observations, ex_goals, ex_actions)),
        )
        model_def = ModuleDict({k: v[0] for k, v in model_info.items()})
        model_params = model_def.init(model_rng, **{k: v[1] for k, v in model_info.items()})['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=config['model_lr']))

        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        return cls(rng, network=network, model=model, elites=elites, config=flax.core.FrozenDict(**config))


def get_config(profile='ogbench'):
    """Return the COMBO config.

    Profiles (select with `--agent=agents/combo.py:<profile>`):
        ogbench: shared MOPO/MOBILE settings for goal-conditioned OGBench comparisons (default).
        paper: settings from Yu et al. (2021), Appendix B, for D4RL-style single-task reproduction.
    """
    assert profile in ('ogbench', 'paper'), f'Unknown COMBO profile: {profile}'
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='combo',  # Agent name.
            profile=profile,  # Hyperparameter profile ('ogbench' or 'paper').
            batch_size=1024,  # Total batch size (real + model).
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            dynamics_hidden_dims=(1024, 1024, 1024, 1024),  # Dynamics, success and reward network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization in the actor and critics.
            model_layer_norm=True,  # Whether to use layer normalization in the model.
            actor_lr=1e-4,  # Actor learning rate.
            critic_lr=3e-4,  # Critic learning rate.
            alpha_lr=3e-4,  # Temperature learning rate.
            model_lr=3e-4,  # Model learning rate.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            backup_entropy=False,  # Whether to subtract alpha * log pi(a'|s') in the TD target.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            num_qs=2,  # Number of Q ensembles.
            q_agg='min',  # Aggregation of target Q values.
            actor_q_agg='mean',  # Aggregation of Q values in the actor loss.
            value_loss='squared',  # Value loss type (COMBO only supports 'squared').
            # Model hyperparameters.
            num_dynamics=7,  # Dynamics ensemble size.
            num_elites=5,  # Number of elite members used for rollouts.
            model_train_steps=100000,  # Number of model pretraining steps.
            model_holdout_size=1000,  # Number of held-out transitions for elite selection.
            # COMBO hyperparameters.
            real_ratio=0.5,  # Real-data fraction f of each batch (paper: 0.5, 0.8); f = 1 is CQL(H). Replaces --mb_batch_ratio.
            beta=1.0,  # Conservative coefficient (paper candidates: 0.5, 1.0, 5.0).
            num_samples=10,  # Action samples per proposal (uniform and policy) for the log-sum-exp estimate.
            conservative_states='mixture',  # States for the push-down term ('mixture' of real/model or 'model').
            rollout_policy='policy',  # Rollout policy ('policy' or 'uniform').
            rollout_steps=5,  # Model rollout horizon.
            rollout_stop_on_terminal=False,  # Whether to drop transitions after a predicted termination.
            ob_dim=ml_collections.config_dict.placeholder(int),  # Observation dimension (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name ('CDataset' for single-task).
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
        )
    )
    if profile == 'paper':
        config.update(
            dict(
                batch_size=256,
                actor_hidden_dims=(256, 256, 256),
                value_hidden_dims=(256, 256, 256),
                dynamics_hidden_dims=(200, 200, 200, 200),
                layer_norm=False,
                model_layer_norm=False,
                model_lr=1e-3,
                discount=0.99,
                target_entropy_multiplier=1.0,
                backup_entropy=True,
                actor_q_agg='min',
                dataset_class='CDataset',
            )
        )
    return config
