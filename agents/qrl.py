import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, LogParam, MLP

sg = lambda x: jax.lax.stop_gradient(x)


def iqe_distance(x, y, iqe_dim=16):
    """Compute the Interval Quasimetric Embedding (IQE, Wang & Isola, 2022) distance d_IQE(x, y)."""
    batch_shape = x.shape[:-1]
    latent_dim = x.shape[-1]
    num_groups = latent_dim // iqe_dim
    x = x.reshape(*batch_shape, num_groups, iqe_dim)
    y = jnp.maximum(x, y.reshape(*batch_shape, num_groups, iqe_dim))
    pts = jnp.concatenate([x, y], axis=-1)
    pts_sorted = jnp.sort(pts, axis=-1)
    mids = 0.5 * (pts_sorted[..., :-1] + pts_sorted[..., 1:])
    segs = pts_sorted[..., 1:] - pts_sorted[..., :-1]
    covered = jnp.any((x[..., None] <= mids[..., None, :]) & (mids[..., None, :] <= y[..., None]), axis=-2)
    return jnp.sum(segs * covered, axis=-1).mean(axis=-1)


class QRLAgent(flax.struct.PyTreeNode):
    """Quasimetric Reinforcement Learning (QRL, Wang et al., 2023; OGBench baseline) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def quasimetric_dist(self, obs, goals, params=None):
        phi_s = self.network.select('phi')(obs, params=params)
        phi_g = self.network.select('phi')(goals, params=params)
        return iqe_distance(phi_s, phi_g, iqe_dim=self.config['iqe_dim'])

    def value_loss(self, batch, grad_params):
        """Compute the QRL quasimetric value loss and dual multiplier loss."""
        d_neg = self.quasimetric_dist(batch['observations'], batch['value_goals'], params=grad_params)
        d_pos = self.quasimetric_dist(batch['observations'], batch['next_observations'], params=grad_params)

        neg_loss = (100.0 * jax.nn.softplus(5.0 - d_neg / 100.0)).mean()
        pos_loss = (jax.nn.relu(d_pos - 1.0) ** 2).mean()

        lam = self.network.select('lam')(params=grad_params)
        eps_sq = self.config['eps'] ** 2
        qrl_loss = neg_loss + sg(lam) * (pos_loss - eps_sq)
        lam_loss = -lam * sg(pos_loss - eps_sq)

        return qrl_loss + lam_loss, {
            'value_loss': qrl_loss,
            'neg_loss': neg_loss,
            'pos_loss': pos_loss,
            'd_neg_mean': d_neg.mean(),
            'd_pos_mean': d_pos.mean(),
            'lam': lam,
        }

    def critic_loss(self, batch, grad_params):
        """Train Q(s, a, g) to regress 1-step quasimetric target r + gamma * mask * (-d_IQE(s', g))."""
        next_d = sg(self.quasimetric_dist(batch['next_observations'], batch['value_goals']))
        next_v = -next_d
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q1.mean(),
            'q_max': q1.max(),
            'q_min': q1.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the DDPG+BC actor loss using the quasimetric critic."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
        q = jnp.minimum(q1, q2)

        q_loss = -q.mean() / sg(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

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

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
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
        ex_goals = example_batch['actor_goals']
        action_dim = ex_actions.shape[-1]

        phi_def = MLP(
            hidden_dims=(*config['value_hidden_dims'], config['latent_dim']),
            activate_final=False,
            layer_norm=config['layer_norm'],
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
        lam_def = LogParam(init_value=1.0)

        network_info = dict(
            phi=(phi_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            lam=(lam_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qrl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            latent_dim=512,
            iqe_dim=16,
            eps=0.05,
            layer_norm=True,
            discount=0.999,
            alpha=1.0,
            const_std=True,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=0.0,
            value_p_randomgoal=1.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=True,
        )
    )
    return config
