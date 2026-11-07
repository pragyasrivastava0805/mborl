from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class CRLAgent(flax.struct.PyTreeNode):
    """Contrastive Reinforcement Learning (CRL, Eysenbach et al., 2022; OGBench baseline) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def bilinear_logits(self, observations, actions, goals, params=None):
        """Compute bilinear inner-product similarity <phi(s, a), psi(g)> for all pairs or elementwise."""
        phi = self.network.select('critic_phi')(observations, actions=actions, params=params)  # (2, B, D)
        psi = self.network.select('critic_psi')(goals, params=params)  # (2, B, D)
        return phi, psi

    def critic_loss(self, batch, grad_params):
        """Compute symmetric InfoNCE contrastive loss over in-trajectory future goals."""
        phi, psi = self.bilinear_logits(
            batch['observations'], batch['actions'], batch['value_goals'], params=grad_params
        )
        # Similarity matrix: (2, B, B)
        logits = jnp.einsum('ebd,ecd->ebc', phi, psi) / jnp.sqrt(self.config['latent_dim'])
        batch_size = batch['observations'].shape[0]
        I = jnp.eye(batch_size)

        loss_fwd = -jnp.mean(jnp.diag(jax.nn.log_softmax(logits[0], axis=-1)) + jnp.diag(jax.nn.log_softmax(logits[1], axis=-1)))
        loss_bwd = -jnp.mean(jnp.diag(jax.nn.log_softmax(logits[0], axis=-2)) + jnp.diag(jax.nn.log_softmax(logits[1], axis=-2)))
        critic_loss = 0.5 * (loss_fwd + loss_bwd)

        pos_sim = jnp.sum(logits * I[None], axis=-1).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'pos_sim': pos_sim,
            'logits_mean': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute DDPG+BC actor loss maximizing contrastive similarity <phi(s, pi(s, g)), psi(g)>."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        phi, psi = self.bilinear_logits(batch['observations'], q_actions, batch['actor_goals'])
        qs = jnp.sum(phi * psi, axis=-1) / jnp.sqrt(self.config['latent_dim'])  # (2, B)
        q = jnp.minimum(qs[0], qs[1])

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

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return critic_loss + actor_loss, info

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

        critic_phi_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            output_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )
        critic_psi_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            output_dim=config['latent_dim'],
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
            critic_phi=(critic_phi_def, (ex_observations, None, ex_actions)),
            critic_psi=(critic_psi_def, (ex_goals,)),
            actor=(actor_def, (ex_observations, ex_goals)),
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
            agent_name='crl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            latent_dim=512,
            layer_norm=True,
            discount=0.999,
            alpha=1.0,
            const_std=True,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            gc_negative=True,
        )
    )
    return config
