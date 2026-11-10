import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.combo import COMBOAgent, get_config as get_combo_config, log_partition
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue, LogParam

sg = lambda x: jax.lax.stop_gradient(x)


class ForwardMOReLCQLAgent(COMBOAgent):
    """CQL + Forward Model Rollouts with MoREL USAD Uncertainty Truncation (`forward_morel_cql`).

    Combines CQL (`COMBO` conservative Q-learning on D_real U D_model) with MoREL's USAD uncertainty-based
    rollout truncation (`disc > usad_thresh` -> `rollout_valid = 0.0`).
    """

    usad_thresh: Any = 0.5

    def compute_discrepancy(self, observations, actions):
        dist = self.model.select('dynamics')(observations, actions)
        means = dist.mean()
        diffs = means[:, None, :, :] - means[None, :, :, :]
        pairwise_l2 = jnp.linalg.norm(diffs, axis=-1)
        return pairwise_l2.max(axis=(0, 1))

    @jax.jit
    def select_elites(self, holdout_batch):
        dist = self.model.select('dynamics')(holdout_batch['observations'], holdout_batch['actions'])
        member_mse = jnp.square(dist.mean() - holdout_batch['next_observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)

        disc = self.compute_discrepancy(holdout_batch['observations'], holdout_batch['actions'])
        calibrated_thresh = jnp.mean(disc) + self.config['morel_thresh_std_mult'] * jnp.std(disc)
        usad_thresh = jnp.where(
            self.config['auto_calibrate_thresh'], calibrated_thresh, self.config['morel_thresh']
        )

        _, model_info = self.model_loss(holdout_batch, None, self.rng)
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info['holdout/elite_mse'] = member_mse[elites].mean()
        info['holdout/usad_thresh'] = usad_thresh
        return self.replace(elites=elites, usad_thresh=usad_thresh), info

    @jax.jit
    def rollout(self, batch, rng):
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

            next_ob = jnp.where(new_halted[:, None] > 0.5, ob, raw_next_ob)
            succ_logit = self.model.select('success')(ob, value_goals, action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                rewards = self.model.select('reward')(ob, None, action)
                if self.config['gc_negative']:
                    rewards = jnp.clip(rewards, -1.0, 0.0)

            masks = 1.0 - success
            rollout_valid = alive * (1.0 - new_halted)
            next_alive = rollout_valid * masks

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

    @classmethod
    def create(cls, seed, example_batch, config):
        base = super().create(seed, example_batch, config)
        return cls(
            rng=base.rng,
            network=base.network,
            model=base.model,
            elites=base.elites,
            usad_thresh=jnp.asarray(config['morel_thresh'], dtype=jnp.float32),
            config=base.config,
        )


def get_config():
    config = get_combo_config()
    config.agent_name = 'forward_morel_cql'
    config.actor_hidden_dims = (512, 512, 512)
    config.value_hidden_dims = (512, 512, 512)
    config.dynamics_hidden_dims = (512, 512, 512)
    config.discount = 0.99
    config.morel_thresh = 0.5
    config.morel_thresh_std_mult = 1.5
    config.auto_calibrate_thresh = True
    config.dataset_class = 'CDataset'
    return config
