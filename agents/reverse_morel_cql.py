import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.combo import COMBOAgent, get_config as get_combo_config
from utils.anchor_sampling import select_anchor_indices
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue, LogParam

sg = lambda x: jax.lax.stop_gradient(x)


class ReverseMOReLCQLAgent(COMBOAgent):
    """CQL + Reverse Model Rollouts with MoREL USAD Uncertainty Truncation (`reverse_morel_cql`).

    Combines CQL (`COMBO` conservative Q-learning on D_real U D_model) with dataset-anchored reverse model
    rollouts truncated by MoREL's USAD uncertainty criterion (`disc > usad_thresh` -> `rollout_valid = 0.0`).
    """

    usad_thresh: Any = 0.5

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
    def rollout(self, batch, rng):
        H = self.config['rollout_steps']
        raw_ob = batch['next_observations']
        raw_vg = batch['value_goals'] if ('value_goals' in batch and batch['value_goals'] is not None) else None
        raw_ag = batch['actor_goals'] if ('actor_goals' in batch and batch['actor_goals'] is not None) else None

        rng, sample_rng = jax.random.split(rng)
        pi_act = jnp.clip(self.network.select('actor')(raw_ob, raw_ag).mode(), -1.0, 1.0)
        qs = self.network.select('target_critic')(raw_ob, raw_vg, pi_act)
        v_scores = jnp.min(qs, axis=0)
        idx = select_anchor_indices(raw_ob, v_scores, self.network.step, self.config, sample_rng)

        anchor_ob = raw_ob[idx]
        batch_size = anchor_ob.shape[0]
        value_goals = raw_vg[idx] if raw_vg is not None else None
        actor_goals = raw_ag[idx] if raw_ag is not None else None

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
            jnp.ones((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            rng,
        )
        _, rollout_data = jax.lax.scan(reverse_rollout_step, init_carry, None, length=H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(value_goals[None], H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(actor_goals[None], H, axis=0)

        return {k: v.reshape((H * batch_size,) + v.shape[2:]) for k, v in rollout_data.items()}

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

        reverse_dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['model_layer_norm'],
            num_ensembles=config['num_dynamics'],
            stochastic=True,
        )
        reverse_actor_def = GCActor(
            hidden_dims=config['dynamics_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['model_layer_norm'],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=0.01,
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
            reverse_dynamics=(reverse_dynamics_def, (ex_observations, ex_actions)),
            reverse_actor=(reverse_actor_def, (ex_observations, ex_goals)),
            success=(success_def, (ex_observations, ex_goals, ex_actions)),
            reward=(reward_def, (ex_observations, ex_goals, ex_actions)),
        )
        model_def = ModuleDict({k: v[0] for k, v in model_info.items()})
        model_params = model_def.init(model_rng, **{k: v[1] for k, v in model_info.items()})['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=config['model_lr']))

        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)
        usad_thresh = jnp.asarray(config['morel_thresh'], dtype=jnp.float32)

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        return cls(
            rng=rng,
            network=network,
            model=model,
            elites=elites,
            usad_thresh=usad_thresh,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = get_combo_config()
    config.agent_name = 'reverse_morel_cql'
    config.actor_hidden_dims = (512, 512, 512)
    config.value_hidden_dims = (512, 512, 512)
    config.dynamics_hidden_dims = (512, 512, 512)
    config.discount = 0.99
    config.rollout_steps = 20
    config.use_usad = True
    config.sampling_mode = 'uniform'
    config.anchor_fraction = 0.25
    config.tau_exp = 3.0
    config.q_beta = 3.0
    config.argmax_top_fraction = 0.01
    config.high_freq_grid_bins = 64
    config.kinematic_density_weight = 0.1
    config.morel_thresh = 0.5
    config.morel_thresh_std_mult = 1.5
    config.auto_calibrate_thresh = True
    config.dataset_class = 'CDataset'
    return config
