import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.combo import build_model_tx
from agents.reverse_rollouts_snis import (
    anchor_resampling_logits,
    reverse_actor_init_args,
    reverse_model_defs,
    reverse_model_loss,
    snis_reverse_rollout,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, rcfm_distill_loss, rcfm_member_mse

sg = lambda x: jax.lax.stop_gradient(x)


class IQLReverseSNISAgent(flax.struct.PyTreeNode):
    """1-step IQL paired with RCFM-1s (or Gaussian) SNIS reverse rollouts and U_rho(s) anchor sampling.

    Unlike COMBOReverseSNISAgent (which uses CQL(H)'s conservative critic and can carve artificial far-maze value
    pits that fool U_rho(s)), this agent uses a pure 1-step IQL critic/value/actor (zero CQL penalty) so V(s) stays
    pit-free for both SNIS reverse action scoring and multi-step advantage anchor sampling U_rho(s).

    Supports staged real-only warmup (`--staged_warmup_steps > 0` in main_mbpo.py) by handling `fake_batch=None`
    during Stage 0 and concatenating real + synthetic transitions in Stages 1..K.
    """

    rng: Any
    network: Any
    model: Any
    elites: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1.0 - expectile))
        return weight * (diff**2)

    # ------------------------------------------------------------------ #
    # Reverse world model pretraining (Gaussian or RCFM-1s)
    # ------------------------------------------------------------------ #

    def _delta_stats(self):
        dm = jnp.asarray(self.config['delta_mean'], dtype=jnp.float32)
        ds = jnp.asarray(self.config['delta_std'], dtype=jnp.float32)
        return -dm, ds

    def _rcfm_distill_tx(self):
        return optax.chain(
            optax.clip_by_global_norm(self.config['rcfm_grad_clip']),
            optax.adam(learning_rate=self.config['rcfm_lr']),
        )

    def init_distill_opt(self):
        """Initialize a fresh Adam state solely for the 1-step distilled reverse dynamics student."""
        return self._rcfm_distill_tx().init(self.model.params['modules_reverse_dynamics_1step'])

    @jax.jit
    def distill_step(self, d_opt_state, batch):
        """Distill reverse RCFM teacher into the 1-step reverse flow map while freezing all other params."""
        student_key = 'modules_reverse_dynamics_1step'
        student_params = self.model.params[student_key]
        d_tx = self._rcfm_distill_tx()
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(sp):
            full_params = {**self.model.params, student_key: sp}
            return rcfm_distill_loss(
                self.model.select('reverse_dynamics'),
                self.model.select('reverse_dynamics_1step'),
                batch['next_observations'],
                batch['actions'],
                self.config['num_dynamics'],
                self.config['rcfm_teacher_steps'],
                rng,
                params=full_params,
            )

        loss, grads = jax.value_and_grad(loss_fn)(student_params)
        updates, new_d_opt_state = d_tx.update(grads, d_opt_state, student_params)
        new_student_params = optax.apply_updates(student_params, updates)
        new_params = {**self.model.params, student_key: new_student_params}
        new_model = self.model.replace(params=new_params)
        return self.replace(model=new_model, rng=new_rng), new_d_opt_state, {'dynamics/distill_loss': loss}

    @jax.jit
    def update_model(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_model, info = self.model.apply_loss_fn(
            loss_fn=lambda p: reverse_model_loss(self.model, batch, p, self.config, rng)
        )
        return self.replace(model=new_model, rng=new_rng), info

    @jax.jit
    def select_elites(self, holdout_batch):
        if self.config.get('dynamics_type', 'gaussian') == 'rcfm':
            rev_dm, rev_ds = self._delta_stats()
            member_mse = rcfm_member_mse(
                self.model,
                'reverse_dynamics',
                'reverse_dynamics_1step',
                holdout_batch['next_observations'],
                holdout_batch['actions'],
                holdout_batch['observations'],
                rev_dm,
                rev_ds,
                self.config['num_dynamics'],
                self.config['rcfm_sample_mode'],
                self.config['rcfm_ode_steps'],
                self.rng,
            )
        else:
            dist = self.model.select('reverse_dynamics')(holdout_batch['next_observations'], holdout_batch['actions'])
            member_mse = jnp.square(dist.mean() - holdout_batch['observations'][None]).mean(axis=(1, 2))
        elites = jnp.argsort(member_mse)[: self.config['num_elites']].astype(jnp.int32)
        _, model_info = reverse_model_loss(self.model, holdout_batch, None, self.config, self.rng)
        if self.config.get('dynamics_type', 'gaussian') == 'rcfm':
            model_info['dynamics/dynamics_mse'] = member_mse.mean()
        info = {f'holdout/{k}': v for k, v in model_info.items()}
        info.update({f'holdout/member_mse_{i}': member_mse[i] for i in range(self.config['num_dynamics'])})
        info['holdout/elite_mse'] = member_mse[elites].mean()
        return self.replace(elites=elites), info

    # ------------------------------------------------------------------ #
    # Value surface, U_rho(s) anchor sampling, and SNIS reverse rollouts
    # ------------------------------------------------------------------ #

    def v_bar(self, obs, value_goals=None, actor_goals=None):
        """Pit-free IQL state-value function V(s)."""
        return self.network.select('value')(obs, value_goals)

    @jax.jit
    def eval_v_pi_chunk(self, obs):
        return self.v_bar(obs)

    @jax.jit
    def score_pdl_anchors_chunk(self, anchor_ob, rng):
        """Score anchors on a fixed-size chunk.

        'lme' (default): log anchor weight beta' G(s) = log E_{(s', a) ~ P^-_mu(.|s)} exp(beta' A_hat(s', a, s)), with
            A_hat = r(s', a) + gamma m(s', a) V(s) - V(s') and beta' = 1 / snis_beta, so rho*(s) ∝ nu(s) exp(beta' G(s)).
        'sample': U_rho(s) = mean_{h=1..H} [R_h + gamma^h M_h V(s) - V(s_h)] along one SNIS reverse rollout.
        """
        if self.config.get('u_rho_estimator', 'lme') == 'lme':
            # Only the anchor's own predecessor samples are scored, so a single-step rollout suffices.
            config = dict(self.config, max_rollout_steps=1)
            return snis_reverse_rollout(self.model, self.elites, config, anchor_ob, None, None, self.v_bar, rng)['anchor_log_w']

        H = int(self.config['max_rollout_steps'])
        discount = float(self.config['discount'])
        chunk_n = anchor_ob.shape[0]

        roll = snis_reverse_rollout(self.model, self.elites, self.config, anchor_ob, None, None, self.v_bar, rng)
        prev_ob = roll['observations'].reshape((H, chunk_n, -1))
        next_ob = roll['next_observations'].reshape((H, chunk_n, -1))
        rewards = roll['rewards'].reshape((H, chunk_n))
        masks = roll['masks'].reshape((H, chunk_n))
        valid = roll['rollout_valid'].reshape((H, chunk_n))

        v_prev = self.v_bar(prev_ob)
        if self.config.get('backup_mode', 'multi') == 'multi':
            v_target = self.v_bar(anchor_ob)[None, :]
        else:
            v_target = self.v_bar(next_ob)
        advantages = rewards + discount * masks * v_target - v_prev
        return jnp.sum(advantages * valid, axis=0) / jnp.maximum(jnp.sum(valid, axis=0), 1.0)

    def _sample_pdl_u_rho_indices(self, ds_dict, valid_idxs, dataset_size, mb_batch_size):
        """Score a uniform pool of real candidate anchors with U_rho(s) and resample via exp(anchor_beta * z)."""
        n = len(valid_idxs)
        pool_mult = max(1, int(self.config.get('anchor_pool_mult', 2)))
        pool_size = mb_batch_size * pool_mult
        pool_local = np.random.randint(0, n, size=pool_size)
        pool_idxs = valid_idxs[pool_local]
        if 'next_observations' in ds_dict:
            cand_obs = ds_dict['next_observations'][pool_idxs]
        else:
            cand_obs = ds_dict['observations'][np.minimum(pool_idxs + 1, dataset_size - 1)]

        estimator = self.config.get('u_rho_estimator', 'lme')
        if estimator == 'real':
            # Each candidate's logged transition (s, a, r, s') is a real draw from P^-_mu(.|s'), so exp(A_hat / temp) is an
            # unbiased, model-free estimate of the rho* anchor weight exp(beta' G(s')); V is only queried on dataset states.
            temp = float(self.config.get('anchor_lme_temp') or self.config['snis_beta'])
            v_next = np.asarray(self.eval_v_pi_chunk(cand_obs))
            v_prev = np.asarray(self.eval_v_pi_chunk(ds_dict['observations'][pool_idxs]))
            masks = ds_dict['masks'][pool_idxs]
            adv = ds_dict['rewards'][pool_idxs] + float(self.config['discount']) * masks * v_next - v_prev
            utilities = (adv / temp).astype(np.float64)
        else:
            chunk_size = min(pool_size, int(self.config.get('rollout_chunk', 2048)))
            n_chunks = (pool_size + chunk_size - 1) // chunk_size
            chunk_keys = jax.random.split(jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1))), n_chunks)
            scores = []
            for idx, start in enumerate(range(0, pool_size, chunk_size)):
                end = min(start + chunk_size, pool_size)
                chunk_obs = cand_obs[start:end]
                actual_len = end - start
                if actual_len < chunk_size:
                    pad_obs = np.repeat(chunk_obs[-1:], chunk_size - actual_len, axis=0)
                    chunk_obs = np.concatenate([chunk_obs, pad_obs], axis=0)
                chunk_u = np.asarray(self.score_pdl_anchors_chunk(chunk_obs, chunk_keys[idx]))[:actual_len]
                scores.append(chunk_u)

            utilities = np.concatenate(scores, axis=0).astype(np.float64)
        if estimator in ('lme', 'real'):
            # Scores are already log weights of rho*(s) / nu(s); the uniform pool is nu, so resample with them as-is.
            # Anchors whose predecessors were all filtered as too uncertain carry -inf and are never chosen.
            logits = anchor_resampling_logits(
                utilities, self.config.get('anchor_weight_transform', 'none'), float(self.config.get('anchor_weight_clip', 20.0))
            )
        else:
            z = np.clip((utilities - utilities.mean()) / max(utilities.std(), 1e-5), -5.0, 5.0)
            logits = float(self.config['anchor_beta']) * z
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        ess_pct = 100.0 / (pool_size * float(np.sum(probs**2)))
        u = utilities[np.isfinite(utilities)] if np.isfinite(utilities).any() else np.zeros(1)
        print(
            f"[iql_pdl_u_rho step={int(self.network.step)}] "
            f"u_mean={u.mean():.3f} u_std={u.std():.3f} "
            f"u_p90={np.percentile(u, 90):.3f} u_p99={np.percentile(u, 99):.3f} u_max={u.max():.3f} "
            f"excluded={100.0 * (1.0 - u.size / pool_size):.1f}% ess={ess_pct:.1f}%",
            flush=True,
        )
        chosen_in_pool = np.random.choice(pool_size, size=mb_batch_size, replace=True, p=probs)
        return pool_local[chosen_in_pool]

    def sample_full_buffer_batch(self, train_dataset, mb_batch_size):
        """Sample real-dataset anchors using 'pdl_u_rho', 'exp_value', 'frontier', or 'uniform' weighting."""
        ds_dict = train_dataset.dataset._dict
        valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
        if self.config.get('exclude_terminal_anchors', True) and 'masks' in ds_dict:
            non_term = valid_idxs[ds_dict['masks'][valid_idxs] > 0.5]
            if len(non_term) > 0:
                valid_idxs = non_term
        n = len(valid_idxs)
        anchor_mode = self.config.get('anchor_mode', 'pdl_u_rho')
        if int(self.network.step) == 0 or anchor_mode == 'uniform':
            chosen = np.random.randint(0, n, size=mb_batch_size)
        elif anchor_mode == 'pdl_u_rho':
            chosen = self._sample_pdl_u_rho_indices(ds_dict, valid_idxs, train_dataset.size, mb_batch_size)
        else:
            if 'next_observations' in ds_dict:
                all_next_obs = ds_dict['next_observations'][valid_idxs]
            else:
                all_next_obs = ds_dict['observations'][np.minimum(valid_idxs + 1, train_dataset.size - 1)]
            v = np.concatenate([np.asarray(self.eval_v_pi_chunk(all_next_obs[s:s + 100000])) for s in range(0, n, 100000)])
            v = v.astype(np.float64)
            z = np.clip((v - v.mean()) / max(v.std(), 1e-5), -5.0, 5.0)
            if anchor_mode == 'frontier':
                target_z = float(self.config.get('anchor_target_z', 0.5))
                band_std = max(float(self.config.get('anchor_band_std', 0.75)), 1e-3)
                logits = -0.5 * ((z - target_z) / band_std) ** 2
            else:
                logits = float(self.config['anchor_beta']) * z
                adv_beta = float(self.config.get('anchor_adv_beta', 0.0))
                if adv_beta > 0:
                    # Value stays the main signal; the anchor's logged one-step advantage r + gamma m V(s') - V(s) (model-free,
                    # V only on dataset states) breaks ties, z-scored and clipped to +-3 so TD-error outliers cannot dominate.
                    prev_obs = ds_dict['observations'][valid_idxs]
                    v_prev = np.concatenate([np.asarray(self.eval_v_pi_chunk(prev_obs[s:s + 100000])) for s in range(0, n, 100000)])
                    masks = ds_dict['masks'][valid_idxs]
                    adv = ds_dict['rewards'][valid_idxs] + float(self.config['discount']) * masks * v - v_prev
                    z_adv = np.clip((adv - adv.mean()) / max(adv.std(), 1e-5), -3.0, 3.0)
                    logits = logits + adv_beta * z_adv
                    p = np.exp(logits - logits.max())
                    p /= p.sum()
                    print(
                        f"[value_adv step={int(self.network.step)}] corr(zV, zA)={np.corrcoef(z, z_adv)[0, 1]:.3f} "
                        f"adv_std={adv.std():.3f} adv_max={adv.max():.3f} ess={100.0 / (n * float(np.sum(p**2))):.1f}%",
                        flush=True,
                    )
            p = np.exp(logits - logits.max())
            chosen = np.random.choice(n, size=mb_batch_size, replace=True, p=p / p.sum())
        return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen])

    @jax.jit
    def rollout(self, batch, rng):
        vg = batch['value_goals'] if 'value_goals' in batch else None
        ag = batch['actor_goals'] if 'actor_goals' in batch else None
        return snis_reverse_rollout(
            self.model, self.elites, self.config, batch['next_observations'], vg, ag, self.v_bar, rng
        )

    # ------------------------------------------------------------------ #
    # 1-step IQL policy learning (supports pure-real Stage 0 and mixed Stages 1..K)
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess(real_batch, fake_batch):
        keys = ['observations', 'actions', 'next_observations', 'rewards', 'masks']
        has_goals = 'value_goals' in real_batch and real_batch['value_goals'] is not None
        if has_goals:
            keys += ['value_goals', 'actor_goals']
        if fake_batch is None:
            batch = {k: real_batch[k] for k in keys}
        else:
            batch = {k: jnp.concatenate([real_batch[k], fake_batch[k]], axis=0) for k in keys}
        if not has_goals:
            batch['value_goals'] = None
            batch['actor_goals'] = None
        return batch

    def value_loss(self, batch, grad_params):
        vg = batch.get('value_goals', None)
        q1, q2 = self.network.select('target_critic')(batch['observations'], vg, batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], vg, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        vg = batch.get('value_goals', None)
        next_v = self.network.select('value')(batch['next_observations'], vg)
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        if self.config['gc_negative']:
            q_target = jnp.clip(q_target, -1.0 / (1.0 - self.config['discount']), 0.0)
        q_target = sg(q_target)

        q1, q2 = self.network.select('critic')(batch['observations'], vg, batch['actions'], params=grad_params)
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_target.mean(),
            'q_max': q_target.max(),
            'q_min': q_target.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        ag = batch.get('actor_goals', None)
        if self.config['actor_loss'] == 'awr':
            v = self.network.select('value')(batch['observations'], ag)
            q1, q2 = self.network.select('critic')(batch['observations'], ag, batch['actions'])
            adv = jnp.minimum(q1, q2) - v
            exp_a = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)
            dist = self.network.select('actor')(batch['observations'], ag, params=grad_params)
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
            dist = self.network.select('actor')(batch['observations'], ag, params=grad_params)
            q_actions = jnp.clip(dist.mode() if self.config['const_std'] else dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], ag, q_actions)
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
            lambda p, tp: p * self.config['tau'] + tp * (1.0 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, real_batch, fake_batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(real_batch, fake_batch, p, rng=rng)
        )
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, real_batch, fake_batch, rng=None):
        goals = real_batch['actor_goals'] if 'actor_goals' in real_batch else None
        actions, _ = self.sample_actions(real_batch['observations'], goals, seed=rng, mode='train')
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            'mse': jnp.mean((actions - real_batch['actions']) ** 2),
            'fake_reward_mean': zero if fake_batch is None else fake_batch['rewards'].mean(),
            'fake_mask_mean': zero if fake_batch is None else fake_batch['masks'].mean(),
        }

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0, mode='test'):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1.0, 1.0), {}

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, model_rng = jax.random.split(rng, 3)

        ex_obs = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals'] if 'actor_goals' in example_batch else None
        ob_dim = ex_obs.shape[-1]
        action_dim = ex_actions.shape[-1]

        value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=1)
        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=2)
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        network_info = dict(
            value=(value_def, (ex_obs, ex_goals)),
            critic=(critic_def, (ex_obs, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs, ex_goals, ex_actions)),
            actor=(actor_def, (ex_obs, ex_goals)),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_params = network_def.init(init_rng, **{k: v[1] for k, v in network_info.items()})['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        defs = reverse_model_defs(config, ob_dim, action_dim, config['model_layer_norm'])
        init_kwargs = dict(
            reverse_actor=reverse_actor_init_args(config, ex_obs, ex_goals, ex_actions),
            success=(ex_obs, ex_goals, ex_actions),
            reward=(ex_obs, ex_goals, ex_actions),
        )
        if config.get('dynamics_type', 'gaussian') == 'rcfm':
            assert len(config['delta_mean']) == ob_dim and len(config['delta_std']) == ob_dim, (
                'RCFM dynamics requires delta_mean and delta_std populated from the training dataset.'
            )
            num_ens = config['num_dynamics']
            ex_x = jnp.zeros((num_ens, ex_obs.shape[0], ob_dim))
            ex_t = jnp.zeros((num_ens, ex_obs.shape[0], 1))
            init_kwargs['reverse_dynamics'] = (ex_obs, ex_actions, ex_x, ex_t)
            init_kwargs['reverse_dynamics_1step'] = (ex_obs, ex_actions, ex_x)
            model_tx = build_model_tx(
                config,
                rcfm_keys=('modules_reverse_dynamics',),
                frozen_keys=('modules_reverse_dynamics_1step',),
            )
        else:
            init_kwargs['reverse_dynamics'] = (ex_obs, ex_actions)
            model_tx = build_model_tx(config)

        model_def = ModuleDict(defs)
        model_params = model_def.init(model_rng, **init_kwargs)['params']
        model = TrainState.create(model_def, model_params, tx=model_tx)

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        elites = jnp.arange(config['num_elites'], dtype=jnp.int32)
        return cls(rng, network=network, model=model, elites=elites, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='iql_reverse_snis',
            # IQL policy & critic hyperparameters (matching agents/iql.py).
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            dynamics_hidden_dims=(512, 512, 512),
            layer_norm=True,
            model_layer_norm=True,
            model_lr=3e-4,
            discount=0.995,
            tau=0.005,
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.003,
            const_std=True,
            # Reverse world model (RCFM-1s by default, or Gaussian).
            num_dynamics=7,
            num_elites=5,
            model_train_steps=100000,
            model_holdout_size=1000,
            reverse_actor_type='gaussian',  # 'gaussian' (unimodal diag Gaussian) or 'flow' (conditional flow-matching ODE).
            reverse_actor_flow_steps=10,
            reverse_actor_time_half_dim=16,
            dynamics_type='rcfm',
            rcfm_hidden_dims=(512, 512, 512, 512),
            rcfm_layer_norm=True,
            rcfm_lr=3e-4,
            rcfm_grad_clip=10.0,
            rcfm_time_half_dim=16,
            rcfm_teacher_steps=10,
            rcfm_distill_steps=25000,
            rcfm_sample_mode='1step',
            rcfm_ode_steps=5,
            delta_mean=(),
            delta_std=(),
            # SNIS reverse rollouts & anchor sampling.
            num_rev_candidates=16,
            num_mc_predecessors=5,
            snis_beta=0.5,
            snis_mode='mean',  # 'mean' (E_s[A]/beta + unif s), 'tilted' (LME + fresh tilted s), or 'joint' (LME + reused s).
            rollout_steps=5,
            max_rollout_steps=5,
            rollout_chunk=2048,
            clip_model_reward=True,
            backup_mode='multi',
            full_buffer_exp=True,
            anchor_mode='pdl_u_rho',
            anchor_beta=0.5,
            anchor_pool_mult=2,
            u_rho_estimator='lme',  # 'lme' (log-mean-exp model-predecessor advantage, used as log weight), 'real' (logged predecessor), or 'sample'.
            anchor_lme_temp=2.0,  # Temperature of the 'lme' anchor weight mean exp(A_hat / temp); snis_beta gives exact rho*.
            anchor_unc_coef=0.0,  # 'lme' anchors: A_hat - coef * std over elite predecessors (model-uncertainty LCB).
            anchor_unc_quantile=1.0,  # 'lme' anchors: average only over (s, a) with ensemble std of A_hat below this quantile.
            anchor_score='lme',  # 'lme' anchor log-weight: 'lme' (log-mean-exp A_hat) or 'ens_lcb' (mean_m S_m - coef * std_m S_m).
            anchor_ens_coef=1.0,  # 'ens_lcb': weight on the std across elites of the per-member expected advantage S_m.
            anchor_weight_transform='none',  # 'lme'/'real' anchor weights: 'none', 'clip', 'norm_clip' (AWR), 'robust_clip' (median/IQR), or 'binary' (CRR).
            anchor_weight_clip=20.0,  # Weight cap for 'clip' / 'norm_clip' / 'robust_clip'.
            anchor_adv_beta=0.0,  # 'exp_value': add anchor_adv_beta * clip(z(logged A_hat), +-3) to the value logits (0 = exp_value).
            anchor_target_z=0.5,
            anchor_band_std=0.75,
            exclude_terminal_anchors=True,
            ob_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
            dataset_class='CDataset',
            gc_negative=True,
        )
    )
