import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.combo import log_partition
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, LogParam

sg = lambda x: jax.lax.stop_gradient(x)


class CQLAgent(flax.struct.PyTreeNode):
    """Conservative Q-Learning (CQL, Kumar et al., 2020) model-free offline RL agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate(self, qs, agg):
        return qs.min(axis=0) if agg == 'min' else qs.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the CQL(H) critic loss on real dataset transitions."""
        value_goals = batch.get('value_goals', None)
        obs, actions, next_obs = batch['observations'], batch['actions'], batch['next_observations']
        rewards, masks = batch['rewards'], batch['masks']

        next_rng, rand_rng, pi_rng = jax.random.split(rng, 3)

        next_dist = self.network.select('actor')(next_obs, goals=value_goals)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_rng)
        next_qs = self.network.select('target_critic')(next_obs, goals=value_goals, actions=next_actions)
        next_q = self.aggregate(next_qs, self.config['q_agg'])
        if self.config['backup_entropy']:
            next_q = next_q - self.network.select('lam')() * next_log_probs
        target_q = rewards + self.config['discount'] * masks * next_q
        if value_goals is not None:
            if self.config['gc_negative']:
                target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
            else:
                target_q = jnp.clip(target_q, 0, 1)
        target_q = sg(target_q)

        qs = self.network.select('critic')(obs, goals=value_goals, actions=actions, params=grad_params)
        td_loss = 0.5 * jnp.square(qs - target_q).mean(axis=1)

        num_states = obs.shape[0]
        n = self.config['num_samples']
        action_dim = self.config['action_dim']
        rep_obs = jnp.repeat(obs, n, axis=0)
        rep_goals = jnp.repeat(value_goals, n, axis=0) if value_goals is not None else None

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
        cons_neg = qs.mean(axis=1)
        regularizer = cons_pos - cons_neg

        critic_loss = (td_loss + self.config['beta'] * regularizer).sum()

        return critic_loss, {
            'critic_loss': critic_loss,
            'td_loss': td_loss.mean(),
            'conservative_loss': self.config['beta'] * regularizer.mean(),
            'regularizer': regularizer.mean(),
            'q_data_mean': qs.mean(),
            'q_rand_mean': rand_qs.mean(),
            'q_pi_mean': pi_qs.mean(),
            'target_mean': target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor and temperature losses."""
        actor_goals = batch.get('actor_goals', None)
        obs = batch['observations']

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
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        actor_rng, critic_rng = jax.random.split(rng)

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
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        mode='test',
    ):
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch.get('actor_goals', None)
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

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            layer_norm=True,
            actor_lr=1e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            discount=0.99,
            tau=0.005,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            backup_entropy=False,
            tanh_squash=True,
            state_dependent_std=True,
            actor_fc_scale=0.01,
            num_qs=2,
            q_agg='min',
            actor_q_agg='mean',
            beta=1.0,
            num_samples=10,
            action_dim=ml_collections.config_dict.placeholder(int),
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
    return config
