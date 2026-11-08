import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.mopo import get_config as get_mopo_config
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue, LogParam

sg = lambda x: jax.lax.stop_gradient(x)


class MOReLAgent(flax.struct.PyTreeNode):
    """MOReL (Kidambi et al., NeurIPS 2020) Pessimistic MDP agent with USAD uncertainty truncation.

    Follows Algorithm 1 of Kidambi et al. (2020):
      1. Model pretraining (`update_model` & `select_elites`): Train the dynamics ensemble, reward, and
         termination heads in `model` on D_real, select elite members on held-out transitions, and calibrate
         the Unknown State-Action Detector (USAD) threshold `usad_thresh`.
      2. P-MDP construction (`rollout` with USAD): At each rollout step k, compute pairwise ensemble mean
         discrepancy across elite dynamics members:
             disc(s_k, a_k) = max_{i, j} || mu_i(s_k, a_k) - mu_j(s_k, a_k) ||_2
         Whenever disc(s_k, a_k) > usad_thresh while the trajectory is alive:
           - Transition k enters the absorbing HALT state:
             rewards_k = halt_penalty, masks_k = 0.0 (zeroing out terminal Q bootstrap), rollout_valid_k = 1.0.
           - All subsequent steps k' > k along that branch are truncated (rollout_valid_{k'} = 0.0) and
             remain in the absorbing HALT state.
      3. Policy optimization (`update`): Train actor, critic, and entropy temperature in `network` on pooled
         real and P-MDP transitions while `model` remains frozen.
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

    # ------------------------------------------------------------------ #
    # Step 1: Model pretraining & USAD calibration
    # ------------------------------------------------------------------ #

    def model_loss(self, batch, grad_params, rng):
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
            'reward/reward_loss': reward_loss,
            'reward/reward_mean': rewards.mean(),
        }

    @jax.jit
    def update_model(self, batch):
        """Take one gradient step on the world model during pretraining."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.model_loss(batch, grad_params, rng)

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(model=new_model, rng=new_rng), info

    def compute_discrepancy(self, observations, actions):
        """Compute max pairwise L2 ensemble mean discrepancy U_MOReL(s, a)."""
        dist = self.model.select('dynamics')(observations, actions)
        means = dist.mean()  # (num_dynamics, batch_size, ob_dim)
        diffs = means[:, None, :, :] - means[None, :, :, :]  # (M, M, B, D)
        pairwise_l2 = jnp.linalg.norm(diffs, axis=-1)  # (M, M, B)
        return pairwise_l2.max(axis=(0, 1))  # (B,)

    @jax.jit
    def select_elites(self, holdout_batch):
        """Select elite dynamics members and calibrate USAD threshold on held-out real transitions."""
        dist = self.model.select('dynamics')(holdout_batch['observations'], holdout_batch['actions'])
        member_mse = jnp.square(dist.mean() - holdout_batch['next_observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)

        disc = self.compute_discrepancy(holdout_batch['observations'], holdout_batch['actions'])
        calibrated_thresh = jnp.mean(disc) + self.config['morel_thresh_std_mult'] * jnp.std(disc)
        usad_thresh = jnp.where(self.config['auto_calibrate_thresh'], calibrated_thresh, self.config['morel_thresh'])

        _, model_info = self.model_loss(holdout_batch, None, self.rng)
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info['holdout/elite_mse'] = member_mse[elites].mean()
        info['holdout/usad_thresh'] = usad_thresh
        return self.replace(elites=elites, usad_thresh=usad_thresh), info

    def sample_dynamics(self, observations, actions, seed):
        """Sample next states from a uniformly chosen elite member."""
        dyn_rng, member_rng = jax.random.split(seed)
        next_obs = self.model.select('dynamics')(observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (next_obs.shape[1],), 0, self.elites.shape[0])]
        return next_obs[member, jnp.arange(next_obs.shape[1])]

    # ------------------------------------------------------------------ #
    # Step 2: USAD-Truncated P-MDP Rollouts
    # ------------------------------------------------------------------ #

    @jax.jit
    def rollout(self, batch, rng):
        """Perform USAD-truncated P-MDP rollouts (Kidambi et al., 2020)."""
        H = self.config['rollout_steps']
        init_ob = batch['observations']
        batch_size = init_ob.shape[0]
        value_goals = batch['value_goals'] if 'value_goals' in batch else None

        def rollout_step(carry, _):
            ob, alive, halted, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            action, _ = self.sample_actions(ob, value_goals, action_rng, mode='train')
            raw_next_ob = self.sample_dynamics(ob, action, dyn_rng)
            disc = self.compute_discrepancy(ob, action)
            unknown = (disc > self.usad_thresh).astype(jnp.float32)
            new_halted = jnp.maximum(halted, unknown)

            succ_logit = self.model.select('success')(ob, value_goals, action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                step_rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                step_rewards = self.model.select('reward')(ob, None, action)

            # USAD HALT absorbing state: overwrite reward with halt_penalty and set terminal mask = 0.0
            next_ob = jnp.where(halted[:, None] > 0.5, ob, raw_next_ob)
            rewards = jnp.where(new_halted > 0.5, self.config['halt_penalty'], step_rewards)
            masks = jnp.where(new_halted > 0.5, 0.0, 1.0 - success)
            rollout_valid = alive
            next_alive = alive * (1.0 - new_halted) * (1.0 - success)

            return (next_ob, next_alive, new_halted, rng), {
                'observations': ob,
                'actions': action,
                'next_observations': next_ob,
                'rewards': rewards,
                'masks': masks,
                'rollout_valid': rollout_valid,
            }

        init_carry = (
            init_ob,
            jnp.ones((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            rng,
        )
        _, rollout_data = jax.lax.scan(rollout_step, init_carry, None, length=H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(batch['value_goals'][None], H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(batch['actor_goals'][None], H, axis=0)

        return {k: v.reshape((H * batch_size,) + v.shape[2:]) for k, v in rollout_data.items()}

    # ------------------------------------------------------------------ #
    # Step 3: Policy optimization in the P-MDP
    # ------------------------------------------------------------------ #

    def preprocess(self, real_batch, fake_batch):
        batch = {
            'observations': jnp.concatenate([real_batch['observations'], fake_batch['observations']], axis=0),
            'actions': jnp.concatenate([real_batch['actions'], fake_batch['actions']], axis=0),
            'next_observations': jnp.concatenate([real_batch['next_observations'], fake_batch['next_observations']], axis=0),
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

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'], goals=batch['value_goals'])
        next_actions, _ = next_dist.sample_and_log_prob(seed=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'], goals=batch['value_goals'], actions=next_actions
        )
        next_q = next_qs.min(axis=0) if self.config['q_agg'] == 'min' else next_qs.mean(axis=0)
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        if batch['value_goals'] is not None:
            if self.config['gc_negative']:
                target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
            else:
                target_q = jnp.clip(target_q, 0, 1)

        q = self.network.select('critic')(
            batch['observations'], goals=batch['value_goals'], actions=batch['actions'], params=grad_params
        )
        critic_loss = jnp.square(q - sg(target_q)).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        dist = self.network.select('actor')(batch['observations'], goals=batch['actor_goals'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions)
        q = jnp.mean(qs, axis=0)
        actor_loss = (log_probs * self.network.select('lam')() - q).mean()

        lam = self.network.select('lam')(params=grad_params)
        entropy = -sg(log_probs).mean()
        lam_loss = (lam * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + lam_loss
        action_std = dist._distribution.stddev().mean() if self.config['tanh_squash'] else dist.stddev().mean()
        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'lam_loss': lam_loss,
            'lam': lam,
            'entropy': entropy,
            'std': action_std,
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        actor_rng, critic_rng = jax.random.split(rng)

        batch = self.preprocess(real_batch, fake_batch)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        info.update({f'critic/{k}': v for k, v in critic_info.items()})

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update({f'actor/{k}': v for k, v in actor_info.items()})

        return critic_loss + actor_loss, info

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

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

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
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_dynamics'],
            stochastic=True,
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
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
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
    config = get_mopo_config()
    config.agent_name = 'morel'
    config.beta = 0.0  # MOReL replaces per-step soft penalty with hard USAD HALT truncation + halt_penalty
    config.num_elites = 5
    config.model_train_steps = 100000
    config.model_holdout_size = 1000
    config.morel_thresh = 0.5  # Fallback USAD discrepancy threshold in normalized state space
    config.morel_thresh_std_mult = 1.5  # USAD threshold = mean(disc_real) + 1.5 * std(disc_real)
    config.auto_calibrate_thresh = True
    config.halt_penalty = -10.0  # Absorbing HALT state penalty r_halt
    config.dataset_class = 'CDataset'
    return config
