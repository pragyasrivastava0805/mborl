import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class HIQLAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning (HIQL) agent (Park et al., 2023)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the IQL expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the action-free IQL value loss V(s, g)."""
        next_v1, next_v2 = self.network.select('target_value')(
            batch['next_observations'], batch['high_value_goals']
        )
        next_v = (next_v1 + next_v2) / 2
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        v1_t, v2_t = self.network.select('target_value')(
            batch['observations'], batch['high_value_goals']
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['high_value_goals'], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level subgoal actor loss pi_h(w | s, g)."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(
            batch['high_actor_next_observations'], batch['high_actor_goals']
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=grad_params
        )
        log_prob = dist.log_prob(batch['high_actor_actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['high_actor_actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level primitive actor loss pi_l(a | s, w)."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('low_actor')(
            batch['observations'], batch['low_actor_goals'], params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = value_loss + high_actor_loss + low_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions hierarchically: subgoal w ~ pi_h(.|s, g), action a ~ pi_l(.|s, w)."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        subgoals = high_dist.sample(seed=high_seed)

        low_dist = self.network.select('low_actor')(observations, subgoals, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)
        actions = jnp.clip(actions, -1, 1)
        return actions, {}

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new HIQL agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=goal_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='hiql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,
            high_alpha=3.0,
            const_std=True,
            action_dim=ml_collections.config_dict.placeholder(int),
            goal_dim=ml_collections.config_dict.placeholder(int),
            dataset_class='HGCDataset',
            subgoal_steps=25,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
        )
    )
    return config
