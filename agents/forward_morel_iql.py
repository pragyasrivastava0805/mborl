import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class ForwardMOReLIQLAgent(flax.struct.PyTreeNode):
    """Forward-MoREL + IQL: Forward Model Rollouts with MoREL USAD Uncertainty Truncation and IQL Critic/Actor.

    Exact forward-rollout counterpart to `ReverseMOReLAgent` (`agents/reverse_morel.py`):
      1. Forward world model pretraining (`update_model` & `select_elites`):
         - Train an M-member forward dynamics ensemble `dynamics(s_t, a_t) -> s_{t+1}` alongside `reward`
           and `success` heads on D_real, and calibrate `usad_thresh = mean(disc) + 1.5 * std(disc)` on holdout data.
      2. Dataset-started USAD-truncated forward rollouts (`rollout`):
         - Start at random dataset states `s_t in D_real` and roll forward k = 1..H steps under the learned policy,
           truncating any rollout branch as soon as ensemble discrepancy exceeds `usad_thresh` (`rollout_valid = 0.0`,
           with no terminal HALT cliff).
      3. Exact IQL policy optimization (`update` on `D_real U D_fwd-USAD`):
         - Identical IQL `value_loss`, `critic_loss`, and `actor_loss` (`ddpgbc` / `awr`) as `ReverseMOReLAgent` and `IQLAgent`.
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

    # ------------------------------------------------------------------ #
    # Step 1: Forward model pretraining & USAD calibration
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
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.model_loss(batch, grad_params, rng)

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(model=new_model, rng=new_rng), info

    def compute_discrepancy(self, observations, actions):
        dist = self.model.select('dynamics')(observations, actions)
        means = dist.mean()  # (M, B, D)
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

    def sample_dynamics(self, observations, actions, seed):
        dyn_rng, member_rng = jax.random.split(seed)
        next_obs = self.model.select('dynamics')(observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (next_obs.shape[1],), 0, self.elites.shape[0])]
        return next_obs[member, jnp.arange(next_obs.shape[1])]

    # ------------------------------------------------------------------ #
    # Step 2: USAD-Truncated Forward Rollouts (No Terminal HALT Cliff)
    # ------------------------------------------------------------------ #

    @jax.jit
    def rollout(self, batch, rng):
        H = self.config['rollout_steps']
        init_ob = batch['observations']
        batch_size = init_ob.shape[0]
        value_goals = batch['value_goals'] if 'value_goals' in batch else None

        def forward_rollout_step(carry, _):
            cur_ob, alive, halted, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            action, _ = self.sample_actions(cur_ob, value_goals, action_rng, mode='train')
            raw_next_ob = self.sample_dynamics(cur_ob, action, dyn_rng)

            disc = self.compute_discrepancy(cur_ob, action)
            unknown = (disc > self.usad_thresh).astype(jnp.float32)
            new_halted = jnp.maximum(halted, unknown)

            next_ob = jnp.where(new_halted[:, None] > 0.5, cur_ob, raw_next_ob)
            succ_logit = self.model.select('success')(cur_ob, value_goals, action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                rewards = self.model.select('reward')(cur_ob, None, action)
                if self.config['gc_negative']:
                    rewards = jnp.clip(rewards, -1.0, 0.0)

            masks = 1.0 - success
            rollout_valid = alive * (1.0 - new_halted)
            next_alive = rollout_valid * masks

            return (next_ob, next_alive, new_halted, rng), {
                'observations': cur_ob,
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
        _, rollout_data = jax.lax.scan(forward_rollout_step, init_carry, None, length=H)

        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(batch['value_goals'][None], H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(batch['actor_goals'][None], H, axis=0)

        return {k: v.reshape((H * batch_size,) + v.shape[2:]) for k, v in rollout_data.items()}

    # ------------------------------------------------------------------ #
    # Step 3: Exact IQL Policy Optimization on D_real U D_fwd-USAD
    # ------------------------------------------------------------------ #

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
    config = ml_collections.ConfigDict(
        dict(
            agent_name='forward_morel_iql',
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
