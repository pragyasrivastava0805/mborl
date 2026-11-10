import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCDeterActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


class ForwardMOReLReBRACAgent(flax.struct.PyTreeNode):
    """ReBRAC + Forward Model Rollouts with MoREL USAD Uncertainty Truncation (`forward_morel_rebrac`)."""

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

    def model_loss(self, batch, grad_params, rng):
        goals = batch['value_goals'] if 'value_goals' in batch else None
        dist = self.model.select('dynamics')(batch['observations'], batch['actions'], params=grad_params)
        dynamics_loss = -dist.log_prob(batch['next_observations'][None]).mean()
        dynamics_mse = jnp.square(dist.mean() - batch['next_observations'][None]).mean()

        succ_logit = self.model.select('success')(batch['observations'], goals, batch['actions'], params=grad_params)
        success = jax.nn.sigmoid(succ_logit)
        success_loss = self.bce_loss(succ_logit, 1 - batch['masks']).mean()

        rewards = self.model.select('reward')(batch['observations'], goals, batch['actions'], params=grad_params)
        reward_loss = jnp.square(rewards - batch['rewards']).mean()

        loss = dynamics_loss + success_loss + reward_loss
        return loss, {
            'dynamics/dynamics_loss': dynamics_loss,
            'dynamics/dynamics_mse': dynamics_mse,
            'success/success_loss': success_loss,
            'reward/reward_loss': reward_loss,
        }

    @jax.jit
    def update_model(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_model, info = self.model.apply_loss_fn(loss_fn=lambda p: self.model_loss(batch, p, rng))
        return self.replace(model=new_model, rng=new_rng), info

    def compute_discrepancy(self, observations, actions):
        dist = self.model.select('dynamics')(observations, actions)
        means = dist.mean()
        diffs = means[:, None, :, :] - means[None, :, :, :]
        return jnp.linalg.norm(diffs, axis=-1).max(axis=(0, 1))

    @jax.jit
    def select_elites(self, holdout_batch):
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
        dyn_rng, member_rng = jax.random.split(seed)
        next_obs = self.model.select('dynamics')(observations, actions).sample(seed=dyn_rng)
        member = self.elites[jax.random.randint(member_rng, (next_obs.shape[1],), 0, self.elites.shape[0])]
        return next_obs[member, jnp.arange(next_obs.shape[1])]

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
            new_halted = jnp.maximum(halted, (disc > self.usad_thresh).astype(jnp.float32))
            next_ob = jnp.where(new_halted[:, None] > 0.5, ob, raw_next_ob)

            succ_logit = self.model.select('success')(ob, value_goals, action)
            success = (jax.nn.sigmoid(succ_logit) > 0.5).astype(jnp.float32)
            if value_goals is not None:
                rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            else:
                rewards = jnp.clip(self.model.select('reward')(ob, None, action), -1.0, 0.0)
            masks = 1.0 - success
            rollout_valid = alive * (1.0 - new_halted)
            return (next_ob, rollout_valid * masks, new_halted, rng), {
                'observations': ob,
                'actions': action,
                'next_observations': next_ob,
                'rewards': rewards,
                'masks': masks,
                'rollout_valid': rollout_valid,
            }

        init_carry = (init_ob, jnp.ones((batch_size,)), jnp.zeros((batch_size,)), rng)
        _, rollout_data = jax.lax.scan(rollout_step, init_carry, None, length=H)
        if value_goals is not None:
            rollout_data['value_goals'] = jnp.repeat(batch['value_goals'][None], H, axis=0)
            rollout_data['actor_goals'] = jnp.repeat(batch['actor_goals'][None], H, axis=0)
        return {k: v.reshape((H * batch_size,) + v.shape[2:]) for k, v in rollout_data.items()}

    def preprocess(self, real_batch, fake_batch):
        batch = {
            'observations': jnp.concatenate([real_batch['observations'], fake_batch['observations']], axis=0),
            'actions': jnp.concatenate([real_batch['actions'], fake_batch['actions']], axis=0),
            'next_observations': jnp.concatenate([real_batch['next_observations'], fake_batch['next_observations']], axis=0),
            'rewards': jnp.concatenate([real_batch['rewards'], fake_batch['rewards']], axis=0),
            'masks': jnp.concatenate([real_batch['masks'], fake_batch['masks']], axis=0),
        }
        batch['value_goals'] = None
        batch['actor_goals'] = None
        return batch

    def critic_loss(self, batch, grad_params, rng):
        obs, actions, next_obs = batch['observations'], batch['actions'], batch['next_observations']
        rewards, masks = batch['rewards'], batch['masks']

        noise = jnp.clip(jax.random.normal(rng, actions.shape) * self.config['policy_noise'], -self.config['noise_clip'], self.config['noise_clip'])
        next_pi_actions = self.network.select('target_actor')(next_obs)
        next_actions = jnp.clip(next_pi_actions + noise, -1.0, 1.0)

        next_q1, next_q2 = self.network.select('target_critic')(next_obs, None, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        bc_penalty = jnp.sum((next_actions - actions) ** 2, axis=-1)
        target_q = rewards + self.config['discount'] * masks * (next_q - self.config['beta_critic'] * bc_penalty)
        if self.config['gc_negative']:
            target_q = jnp.clip(target_q, -1.0 / (1.0 - self.config['discount']), 0.0)
        target_q = sg(target_q)

        q1, q2 = self.network.select('critic')(obs, None, actions, params=grad_params)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q1.mean(), 'q_max': q1.max(), 'q_min': q1.min()}

    def actor_loss(self, batch, grad_params, rng=None):
        pi_actions = self.network.select('actor')(batch['observations'], params=grad_params)
        q1, q2 = self.network.select('critic')(batch['observations'], None, pi_actions)
        q = jnp.minimum(q1, q2)
        lmbda = 1.0 / (sg(jnp.abs(q).mean()) + 1e-6)
        q_loss = -lmbda * q.mean()
        bc_mse = jnp.mean((pi_actions - batch['actions']) ** 2)
        actor_loss = q_loss + self.config['beta_actor'] * bc_mse
        return actor_loss, {'actor_loss': actor_loss, 'q_loss': q_loss, 'bc_mse': bc_mse, 'mse': bc_mse}

    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        critic_rng, actor_rng = jax.random.split(rng)
        batch = self.preprocess(real_batch, fake_batch)
        c_loss, c_info = self.critic_loss(batch, grad_params, critic_rng)
        a_loss, a_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update({f'critic/{k}': v for k, v in c_info.items()})
        info.update({f'actor/{k}': v for k, v in a_info.items()})
        return c_loss + a_loss, info

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
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(real_batch, fake_batch, p, rng=rng))
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, real_batch, fake_batch, rng=None):
        actions, _ = self.sample_actions(real_batch['observations'], seed=rng, mode='train')
        return {'mse': jnp.mean((actions - real_batch['actions']) ** 2)}

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0, mode='test'):
        actions = jnp.clip(self.network.select('actor')(observations, goals=goals, temperature=temperature), -1.0, 1.0)
        return actions, {}

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, model_rng = jax.random.split(rng, 3)
        ex_observations, ex_actions = example_batch['observations'], example_batch['actions']
        ob_dim, action_dim = ex_observations.shape[-1], ex_actions.shape[-1]

        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=2)
        actor_def = GCDeterActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim, layer_norm=config['layer_norm'], tanh_squash=True)
        network_info = dict(
            critic=(critic_def, (ex_observations, None, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, None)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, None)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_params['modules_target_actor'] = network_params['modules_actor']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        dynamics_def = Dynamics(hidden_dims=config['dynamics_hidden_dims'], output_dim=ob_dim, layer_norm=config['layer_norm'], num_ensembles=config['num_dynamics'], stochastic=True)
        success_def = GCValue(hidden_dims=config['dynamics_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=1)
        reward_def = GCValue(hidden_dims=config['dynamics_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=1)
        model_info = dict(
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
            success=(success_def, (ex_observations, None, ex_actions)),
            reward=(reward_def, (ex_observations, None, ex_actions)),
        )
        model_def = ModuleDict({k: v[0] for k, v in model_info.items()})
        model_params = model_def.init(model_rng, **{k: v[1] for k, v in model_info.items()})['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=config['lr']))

        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)
        usad_thresh = jnp.asarray(config['morel_thresh'], dtype=jnp.float32)
        return cls(rng, network=network, model=model, elites=elites, usad_thresh=usad_thresh, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='forward_morel_rebrac',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            dynamics_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            beta_actor=0.03,
            beta_critic=0.001,
            num_dynamics=7,
            num_elites=5,
            model_train_steps=100000,
            model_holdout_size=1000,
            rollout_steps=5,
            morel_thresh=0.5,
            morel_thresh_std_mult=1.5,
            auto_calibrate_thresh=True,
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
