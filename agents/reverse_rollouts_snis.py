import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCActor, GCValue

sg = lambda x: jax.lax.stop_gradient(x)


def snis_reverse_rollout(model, elites, config, anchor_ob, value_goals, actor_goals, v_bar_fn, rng):
    """Self-normalized importance-sampling (SNIS) reverse rollouts from `anchor_ob`.

    At every reverse state x (starting from the anchors) and for H = config['max_rollout_steps'] steps:
      1. Propose K actions from the pretrained reverse-BC policy: a_1..a_K ~ mu^-(. | x).
      2. For every a_j, draw M predecessors s_jm ~ P^-(. | x, a_j) from the fixed reverse model (one sample from each of
         M elite members) and score D_j = mean_m [ r(s_jm, a_j) + gamma * m(s_jm, a_j) * V_bar(x) - V_bar(s_jm) ].
      3. Resample j* ~ Categorical(softmax(D / beta)). This approximates rho*(a | x) ∝ mu^-(a | x) exp(D(x, a) / beta).
      4. Draw a fresh predecessor s ~ P^-(. | x, a_j*) from a random elite, store (s, a_j*, r, x) and set x <- s.

    Backup modes (config['backup_mode']):
      'single': chained transitions (s_k, a_k, r_k, s_{k-1}); only the first reverse transition uses the real anchor.
      'multi' : every synthetic state bootstraps directly on the real anchor x_0 with its k-step return,
                (s_k, a_k, R_k, x_0) with R_k = r_k + gamma * m_k * R_{k-1} and mask M_k = m_k * gamma^{k-1} * prod m,
                so the critic target r + gamma * mask * Q(x_0) equals the k-step backup to the anchor.
    Anchors are processed in chunks of config['rollout_chunk'] to bound memory. Returns arrays of shape (H * N, ...).
    """
    H = int(config['max_rollout_steps'])
    K = int(config['num_rev_candidates'])
    M = int(min(config['num_mc_predecessors'], config['num_elites']))
    beta = float(config['snis_beta'])
    discount = config['discount']
    clip_reward = bool(config.get('clip_model_reward', False))
    gc_negative = bool(config['gc_negative'])
    multi = config.get('backup_mode', 'single') == 'multi'

    n = anchor_ob.shape[0]
    chunk = int(min(config.get('rollout_chunk', 4096), n))
    n_chunks = (n + chunk - 1) // chunk
    pad_idx = jnp.arange(n_chunks * chunk) % n

    def to_chunks(x):
        return None if x is None else x[pad_idx].reshape((n_chunks, chunk) + x.shape[1:])

    def labels(prev_ob, goals, actions):
        """Reward and mask of the forward transition (prev_ob, actions) -> x, from the learned success/reward heads."""
        success = (jax.nn.sigmoid(model.select('success')(prev_ob, goals, actions)) > 0.5).astype(jnp.float32)
        if goals is not None:
            rewards = success - (1.0 if gc_negative else 0.0)
        else:
            rewards = model.select('reward')(prev_ob, None, actions)
            if clip_reward:
                rewards = jnp.clip(rewards, -1.0, 0.0)
        return rewards, 1.0 - success

    def rollout_chunk(xs):
        x0, vg, ag, chunk_rng = xs

        def step(carry, _):
            x, alive, cum_r, cum_d, rng = carry
            rng, prop_rng, mc_rng, cat_rng, fresh_rng, member_rng = jax.random.split(rng, 6)

            # 1. K proposals from the reverse-BC policy.
            bc_dist = model.select('reverse_actor')(x, goals=vg)
            cand = jnp.clip(bc_dist.sample(seed=prop_rng, sample_shape=(K,)), -1.0, 1.0)  # (K, C, A)

            # 2. M predecessors per proposal, one from each of M elite members (a single ensemble forward pass).
            x_k = jnp.broadcast_to(x[None], (K,) + x.shape)
            preds = model.select('reverse_dynamics')(x_k, cand).sample(seed=mc_rng)  # (E, K, C, D)
            s_jm = preds[elites[:M]]  # (M, K, C, D)
            cand_m = jnp.broadcast_to(cand[None], (M,) + cand.shape)
            vg_m = None if vg is None else jnp.broadcast_to(vg[None, None], (M, K) + vg.shape)
            ag_m = None if ag is None else jnp.broadcast_to(ag[None, None], (M, K) + ag.shape)
            r_jm, m_jm = labels(s_jm, vg_m, cand_m)
            v_x = v_bar_fn(x, vg, ag)[None, None]  # (1, 1, C)
            v_s = v_bar_fn(s_jm, vg_m, ag_m)  # (M, K, C)
            d_hat = jnp.mean(r_jm + discount * m_jm * v_x - v_s, axis=0)  # (K, C)

            # 3. Resample one proposal per state with weights softmax(D / beta).
            logits = d_hat / beta
            j_star = jax.random.categorical(cat_rng, logits, axis=0)  # (C,)
            ar = jnp.arange(x.shape[0])
            action = cand[j_star, ar]

            # 4. Fresh predecessor from a uniformly chosen elite; store (s, a*, r, x).
            fresh = model.select('reverse_dynamics')(x, action).sample(seed=fresh_rng)  # (E, C, D)
            member = elites[jax.random.randint(member_rng, (x.shape[0],), 0, elites.shape[0])]
            prev = fresh[member, ar]
            rewards, masks = labels(prev, vg, action)

            if multi:
                # k-step return and survival discount from prev to the real anchor x0.
                new_cum_r = rewards + discount * masks * cum_r
                anchor_mask = masks * cum_d
                out_next, out_r, out_m = x0, new_cum_r, anchor_mask
                new_cum_d = discount * anchor_mask
            else:
                out_next, out_r, out_m = x, rewards, masks
                new_cum_r, new_cum_d = cum_r, cum_d
            out = {
                'observations': prev,
                'actions': action,
                'next_observations': out_next,
                'rewards': out_r,
                'masks': out_m,
                'rollout_valid': alive,
            }
            # Reaching a predicted goal state ends the backward chain (its predecessors would bootstrap from a terminal).
            return (prev, alive * masks, new_cum_r, new_cum_d, rng), out

        c = x0.shape[0]
        init = (x0, jnp.ones((c,)), jnp.zeros((c,)), jnp.ones((c,)), chunk_rng)
        _, data = jax.lax.scan(step, init, None, length=H)
        return data  # (H, C, ...)

    rngs = jax.random.split(rng, n_chunks)
    data = jax.lax.map(rollout_chunk, (to_chunks(anchor_ob), to_chunks(value_goals), to_chunks(actor_goals), rngs))
    # (n_chunks, H, C, ...) -> (H, N, ...)
    data = {k: jnp.swapaxes(v, 0, 1).reshape((H, n_chunks * chunk) + v.shape[3:])[:, :n] for k, v in data.items()}
    if value_goals is not None:
        data['value_goals'] = jnp.repeat(value_goals[None], H, axis=0)
        data['actor_goals'] = jnp.repeat(actor_goals[None], H, axis=0)
    return {k: v.reshape((H * n,) + v.shape[2:]) for k, v in data.items()}


def reverse_model_loss(model, batch, grad_params):
    """Reverse dynamics (Gaussian NLL), reverse-BC actor (NLL), success (BCE) and reward (MSE) losses."""
    goals = batch['value_goals'] if 'value_goals' in batch else None

    rev_dist = model.select('reverse_dynamics')(batch['next_observations'], batch['actions'], params=grad_params)
    dynamics_loss = -rev_dist.log_prob(batch['observations'][None]).mean()
    dynamics_mse = jnp.square(rev_dist.mean() - batch['observations'][None]).mean()

    rev_act_dist = model.select('reverse_actor')(batch['next_observations'], goals=goals, params=grad_params)
    rev_actor_loss = -rev_act_dist.log_prob(jnp.clip(batch['actions'], -1.0 + 1e-5, 1.0 - 1e-5)).mean()

    succ_logit = model.select('success')(batch['observations'], goals, batch['actions'], params=grad_params)
    success_target = 1 - batch['masks']
    log_p, log_not_p = jax.nn.log_sigmoid(succ_logit), jax.nn.log_sigmoid(-succ_logit)
    success_loss = (-(log_p * success_target + log_not_p * (1 - success_target))).mean()

    rewards = model.select('reward')(batch['observations'], goals, batch['actions'], params=grad_params)
    reward_loss = jnp.square(rewards - batch['rewards']).mean()

    loss = dynamics_loss + rev_actor_loss + success_loss + reward_loss
    return loss, {
        'dynamics/dynamics_loss': dynamics_loss,
        'dynamics/dynamics_mse': dynamics_mse,
        'dynamics/reverse_actor_loss': rev_actor_loss,
        'success/success_loss': success_loss,
        'success/success_mean': jax.nn.sigmoid(succ_logit).mean(),
        'reward/reward_loss': reward_loss,
        'reward/reward_mean': rewards.mean(),
    }


def reverse_model_defs(config, ob_dim, action_dim, layer_norm):
    return dict(
        reverse_dynamics=Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=layer_norm,
            num_ensembles=config['num_dynamics'],
            stochastic=True,
        ),
        reverse_actor=GCActor(
            hidden_dims=config['dynamics_hidden_dims'],
            action_dim=action_dim,
            layer_norm=layer_norm,
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=0.01,
        ),
        success=GCValue(hidden_dims=config['dynamics_hidden_dims'], layer_norm=layer_norm, num_ensembles=1),
        reward=GCValue(hidden_dims=config['dynamics_hidden_dims'], layer_norm=layer_norm, num_ensembles=1),
    )


class ReverseRolloutsSNISAgent(flax.struct.PyTreeNode):
    """Reverse rollouts with SNIS action resampling + a critic that evaluates the current forward policy pi_k.

    Critic (policy evaluation of pi_k, optionally with a CQL(H) penalty):
        Q(s, a) <- r + gamma * m * min_j Q_target_j(s', pi_k(s'))
    V_bar_k(s) = min_j Q_target_j(s, pi_k(s)) (slow Polyak target) is the value used to score reverse actions.
    Actor: DDPG+BC (same as the other reverse-rollout agents). Anchors: top-K V_pi(s') states of the full buffer.
    """

    rng: Any
    network: Any
    model: Any
    elites: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ model
    @jax.jit
    def update_model(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_model, info = self.model.apply_loss_fn(loss_fn=lambda p: reverse_model_loss(self.model, batch, p))
        return self.replace(model=new_model, rng=new_rng), info

    @jax.jit
    def select_elites(self, holdout_batch):
        dist = self.model.select('reverse_dynamics')(holdout_batch['next_observations'], holdout_batch['actions'])
        member_mse = jnp.square(dist.mean() - holdout_batch['observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)
        _, model_info = reverse_model_loss(self.model, holdout_batch, None)
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info['holdout/elite_mse'] = member_mse[elites].mean()
        return self.replace(elites=elites), info

    # ------------------------------------------------------------------ values
    def pi_mode(self, obs, actor_goals=None):
        return jnp.clip(self.network.select('actor')(obs, actor_goals).mode(), -1.0, 1.0)

    def v_bar(self, obs, value_goals=None, actor_goals=None):
        """V_bar_k(s) = min_j Q_target_j(s, pi_k(s))."""
        qs = self.network.select('target_critic')(obs, value_goals, self.pi_mode(obs, actor_goals))
        return qs.min(axis=0)

    @jax.jit
    def eval_v_pi_chunk(self, obs, value_goals=None, actor_goals=None):
        return self.v_bar(obs, value_goals, actor_goals)

    def sample_full_buffer_batch(self, train_dataset, mb_batch_size):
        """Return a rollout batch whose states are the top `argmax_top_fraction` V_bar(s') states of the whole buffer."""
        ds_dict = train_dataset.dataset._dict
        valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
        if 'next_observations' in ds_dict:
            all_next_obs = ds_dict['next_observations'][valid_idxs]
        else:
            all_next_obs = ds_dict['observations'][np.minimum(valid_idxs + 1, train_dataset.size - 1)]
        all_v = np.concatenate(
            [np.asarray(self.eval_v_pi_chunk(all_next_obs[s:s + 100000])) for s in range(0, len(valid_idxs), 100000)]
        )
        top_k = max(1, int(mb_batch_size * float(self.config['argmax_top_fraction'])))
        top_local = np.argpartition(-all_v, top_k)[:top_k] if top_k < len(all_v) else np.arange(len(all_v))
        top_pool = valid_idxs[top_local]
        selected = np.tile(top_pool, (mb_batch_size + len(top_pool) - 1) // len(top_pool))[:mb_batch_size]
        return train_dataset.sample(mb_batch_size, idxs=selected)

    @jax.jit
    def rollout(self, batch, rng):
        obs = batch['next_observations']
        vg = batch['value_goals'] if batch.get('value_goals', None) is not None else None
        ag = batch['actor_goals'] if batch.get('actor_goals', None) is not None else None
        num_anchors = max(1, int(obs.shape[0] * float(self.config['anchor_fraction'])))
        # Batch states are already the top-K V_bar states (sample_full_buffer_batch); keep the highest-V ones.
        idx = jnp.argsort(-self.v_bar(obs, vg, ag))[:num_anchors]
        return snis_reverse_rollout(
            self.model, self.elites, self.config, obs[idx],
            None if vg is None else vg[idx], None if ag is None else ag[idx], self.v_bar, rng,
        )

    # ------------------------------------------------------------------ policy learning
    def critic_loss(self, batch, grad_params, rng):
        vg = batch.get('value_goals', None)
        ag = batch.get('actor_goals', None)
        next_q = self.network.select('target_critic')(
            batch['next_observations'], vg, self.pi_mode(batch['next_observations'], ag)
        ).min(axis=0)
        target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        if self.config['gc_negative']:
            target = jnp.clip(target, -1.0 / (1.0 - self.config['discount']), 0.0)
        target = sg(target)

        qs = self.network.select('critic')(batch['observations'], vg, batch['actions'], params=grad_params)
        td_loss = jnp.square(qs - target[None]).mean(axis=1).sum()
        info = {'td_loss': td_loss, 'q_mean': qs.mean(), 'q_max': qs.max(), 'q_min': qs.min(), 'target_mean': target.mean()}

        loss = td_loss
        if self.config['cql_alpha'] > 0:
            # CQL(H): push down log-sum-exp over uniform + pi_k proposals, push up dataset actions (real half only).
            n_real = self.config['n_real']
            obs, g = batch['observations'][:n_real], None if vg is None else vg[:n_real]
            ns, ad = self.config['cql_num_samples'], self.config['action_dim']
            rep_obs = jnp.repeat(obs, ns, axis=0)
            rep_g = None if g is None else jnp.repeat(g, ns, axis=0)
            rand_rng, pi_rng = jax.random.split(rng)
            rand_a = jax.random.uniform(rand_rng, (rep_obs.shape[0], ad), minval=-1.0, maxval=1.0)
            pi_dist = self.network.select('actor')(rep_obs, None if ag is None else jnp.repeat(ag[:n_real], ns, axis=0))
            pi_a = sg(jnp.clip(pi_dist.sample(seed=pi_rng), -1.0, 1.0))
            q_rand = self.network.select('critic')(rep_obs, rep_g, rand_a, params=grad_params).reshape(-1, n_real, ns)
            q_pi = self.network.select('critic')(rep_obs, rep_g, pi_a, params=grad_params).reshape(-1, n_real, ns)
            log_z = jax.scipy.special.logsumexp(jnp.concatenate([q_rand, q_pi], axis=-1), axis=-1) - jnp.log(2 * ns)
            cql_term = (log_z - qs[:, :n_real]).mean(axis=1).sum()
            loss = loss + self.config['cql_alpha'] * cql_term
            info['cql_term'] = cql_term
        info['critic_loss'] = loss
        return loss, info

    def actor_loss(self, batch, grad_params):
        ag = batch.get('actor_goals', None)
        vg = batch.get('value_goals', None)
        dist = self.network.select('actor')(batch['observations'], ag, params=grad_params)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        q = self.network.select('critic')(batch['observations'], vg, q_actions).min(axis=0)
        q_loss = -q.mean() / sg(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        return q_loss + bc_loss, {
            'actor_loss': q_loss + bc_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    @staticmethod
    def preprocess(real_batch, fake_batch):
        keys = ['observations', 'actions', 'next_observations', 'rewards', 'masks']
        if 'value_goals' in real_batch and real_batch['value_goals'] is not None:
            keys += ['value_goals', 'actor_goals']
        batch = {k: jnp.concatenate([real_batch[k], fake_batch[k]], axis=0) for k in keys}
        if 'value_goals' not in batch:
            batch['value_goals'] = None
            batch['actor_goals'] = None
        return batch

    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch = self.preprocess(real_batch, fake_batch)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info = {f'critic/{k}': v for k, v in critic_info.items()}
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    @jax.jit
    def update(self, real_batch, fake_batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(real_batch, fake_batch, p, rng=rng)
        )
        new_network.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params['modules_critic'],
            self.network.params['modules_target_critic'],
        )
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, real_batch, fake_batch, rng=None):
        goals = real_batch['actor_goals'] if 'actor_goals' in real_batch else None
        actions, _ = self.sample_actions(real_batch['observations'], goals, seed=rng, mode='train')
        return {
            'mse': jnp.mean((actions - real_batch['actions']) ** 2),
            'fake_reward_mean': fake_batch['rewards'].mean(),
            'fake_mask_mean': fake_batch['masks'].mean(),
        }

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0, mode='test'):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), {}

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, model_rng = jax.random.split(rng, 3)
        ex_obs, ex_actions = example_batch['observations'], example_batch['actions']
        ex_goals = example_batch['actor_goals'] if 'actor_goals' in example_batch else None
        ob_dim, action_dim = ex_obs.shape[-1], ex_actions.shape[-1]

        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=2)
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        network_info = dict(
            critic=(critic_def, (ex_obs, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs, ex_goals, ex_actions)),
            actor=(actor_def, (ex_obs, ex_goals)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        defs = reverse_model_defs(config, ob_dim, action_dim, config['layer_norm'])
        model_args = dict(
            reverse_dynamics=(ex_obs, ex_actions),
            reverse_actor=(ex_obs, ex_goals),
            success=(ex_obs, ex_goals, ex_actions),
            reward=(ex_obs, ex_goals, ex_actions),
        )
        model_def = ModuleDict(defs)
        model_params = model_def.init(model_rng, **model_args)['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=config['lr']))

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        # main_mbpo.py draws int(batch_size * mb_batch_ratio) real transitions (default ratio 0.5).
        config['n_real'] = int(config['batch_size'] * config['real_fraction'])
        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)
        return cls(rng, network=network, model=model, elites=elites, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='reverse_rollouts_snis',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            dynamics_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            alpha=0.003,  # DDPG+BC coefficient (same as the other reverse-rollout agents).
            const_std=True,
            cql_alpha=0.0,  # > 0 adds a CQL(H) penalty to the pi_k policy-evaluation critic.
            cql_num_samples=10,
            real_fraction=0.5,  # Must match --mb_batch_ratio (real part of each update batch).
            # Reverse model.
            num_dynamics=7,
            num_elites=5,
            model_train_steps=100000,
            model_holdout_size=1000,
            # SNIS reverse rollouts.
            num_rev_candidates=16,  # K
            num_mc_predecessors=5,  # M (<= num_elites; one predecessor per elite member)
            snis_beta=0.5,  # beta in softmax(D / beta)
            rollout_steps=5,  # Read by main_mbpo.py for the buffer capacity; keep equal to max_rollout_steps.
            max_rollout_steps=5,  # H
            rollout_chunk=4096,
            anchor_fraction=0.25,
            argmax_top_fraction=0.05,
            full_buffer_argmax=True,
            clip_model_reward=True,
            ob_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
            n_real=ml_collections.config_dict.placeholder(int),
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
