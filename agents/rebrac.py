import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDeterActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class ReBRACAgent(flax.struct.PyTreeNode):
    """Revisited Behavior Regularized Actor-Critic (ReBRAC, Tarasov et al., 2023) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC critic loss with target policy smoothing and target BC penalty."""
        value_goals = batch.get('value_goals', None)
        obs, actions, next_obs = batch['observations'], batch['actions'], batch['next_observations']
        rewards, masks = batch['rewards'], batch['masks']
        next_actions_data = batch.get('next_actions', actions)

        noise = jax.random.normal(rng, actions.shape) * self.config['policy_noise']
        noise = jnp.clip(noise, -self.config['noise_clip'], self.config['noise_clip'])
        next_pi_actions = self.network.select('target_actor')(next_obs, goals=value_goals)
        next_actions = jnp.clip(next_pi_actions + noise, -1.0, 1.0)

        next_q1, next_q2 = self.network.select('target_critic')(next_obs, value_goals, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        bc_penalty = jnp.sum((next_actions - next_actions_data) ** 2, axis=-1)
        target_q = rewards + self.config['discount'] * masks * (next_q - self.config['beta_critic'] * bc_penalty)
        if self.config.get('gc_negative', True):
            target_q = jnp.clip(target_q, -1.0 / (1.0 - self.config['discount']), 0.0)
        target_q = sg(target_q)

        q1, q2 = self.network.select('critic')(obs, value_goals, actions, params=grad_params)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q1.mean(),
            'q_max': q1.max(),
            'q_min': q1.min(),
            'bc_penalty': bc_penalty.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the ReBRAC actor loss (normalized Q maximization + BC MSE penalty)."""
        actor_goals = batch.get('actor_goals', None)
        pi_actions = self.network.select('actor')(batch['observations'], goals=actor_goals, params=grad_params)
        q1, q2 = self.network.select('critic')(batch['observations'], actor_goals, pi_actions)
        q = jnp.minimum(q1, q2)

        lmbda = 1.0 / (sg(jnp.abs(q).mean()) + 1e-6)
        q_loss = -lmbda * q.mean()
        bc_mse = jnp.mean((pi_actions - batch['actions']) ** 2)
        actor_loss = q_loss + self.config['beta_actor'] * bc_mse

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_mse': bc_mse,
            'q_mean': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        critic_rng, actor_rng = jax.random.split(rng)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

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
        self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=0.0,
    ):
        actions = self.network.select('actor')(observations, goals=goals, temperature=temperature)
        actions = jnp.clip(actions, -1.0, 1.0)
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

        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        actor_def = GCDeterActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=True,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_params['modules_target_actor'] = network_params['modules_actor']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rebrac',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            beta_actor=0.03,
            beta_critic=0.001,
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
    return config
