import flax
import jax
import numpy as np
import jax.numpy as jnp
import ml_collections
import optax

from agents.combo import COMBOAgent, build_model_tx, get_config as combo_get_config
from agents.reverse_rollouts_snis import (
    anchor_resampling_logits,
    reverse_model_defs,
    reverse_model_loss,
    snis_reverse_rollout,
)
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import rcfm_member_mse


class COMBOReverseSNISAgent(COMBOAgent):
    """COMBO with the forward policy rollouts replaced by SNIS reverse rollouts.

    Everything used for policy learning (CQL(H)-style conservative critic, SAC actor, networks, batch split f,
    rollout budget, buffer size) is inherited unchanged from COMBOAgent. Only the model (reverse dynamics +
    reverse-BC actor + success/reward heads) and `rollout` differ, so CQL (f = 1), COMBO and this agent differ only
    in where the synthetic transitions come from. Rollouts start from the same uniformly sampled dataset states as COMBO.
    V_bar_k(s) = min_j Q_target_j(s, pi_k(s)) scores the reverse actions.

    Anchors are always real dataset states. With anchor_mode='exp_value' they are drawn from the whole dataset with
    P(s) ∝ exp(anchor_beta * z(V_pi(s))), where V_pi(s) = min_j Q_target_j(s, pi(s)) and z is the z-score clipped to
    [-5, 5] (same normalization as reverse_rollouts_exp). The first rollout batch (before any update) is uniform.
    """

    def init_distill_opt(self):
        """Initialize a fresh Adam state solely for the 1-step distilled reverse dynamics student."""
        return self._rcfm_distill_tx().init(self.model.params['modules_reverse_dynamics_1step'])

    @jax.jit
    def distill_step(self, d_opt_state, batch):
        """Distill reverse RCFM teacher into the 1-step reverse flow map."""
        return self._rcfm_distill_step(
            d_opt_state, 'reverse_dynamics', 'reverse_dynamics_1step', batch['next_observations'], batch['actions']
        )

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
            rev_dm, rev_ds = self._delta_stats(reverse=True)
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

    def v_bar(self, obs, value_goals=None, actor_goals=None):
        actions = jnp.clip(self.network.select('actor')(obs, goals=actor_goals).mode(), -1.0, 1.0)
        return self.network.select('target_critic')(obs, goals=value_goals, actions=actions).min(axis=0)

    @jax.jit
    def eval_v_pi_chunk(self, obs):
        return self.v_bar(obs)

    @jax.jit
    def score_pdl_anchors_chunk(self, anchor_ob, rng):
        """Score anchors on a fixed-size chunk.

        'lme' (default): log anchor weight beta' G(s) = log E_{(s', a) ~ P^-_mu(.|s)} exp(beta' A_hat(s', a, s)), with
            A_hat = r(s', a) + gamma m(s', a) V_bar(s) - V_bar(s') and beta' = 1 / snis_beta, so rho*(s) ∝ nu(s) exp(beta' G(s)).
        'sample': U_rho(s) = mean_{h=1..H} [R_h + gamma^h M_h V_bar(s) - V_bar(s_h)] along one SNIS reverse rollout.
        """
        if self.config.get('u_rho_estimator', 'lme') == 'lme':
            # Only the anchor's own predecessor samples are scored, so a single-step rollout suffices.
            config = dict(self.config, max_rollout_steps=1)
            return snis_reverse_rollout(self.model, self.elites, config, anchor_ob, None, None, self.v_bar, rng)['anchor_log_w']

        H = int(self.config['max_rollout_steps'])
        discount = float(self.config['discount'])
        chunk_n = anchor_ob.shape[0]

        # 1. Generate H-step reverse rollouts tau^- | s under the current reverse policy rho.
        roll = snis_reverse_rollout(self.model, self.elites, self.config, anchor_ob, None, None, self.v_bar, rng)
        prev_ob = roll['observations'].reshape((H, chunk_n, -1))
        next_ob = roll['next_observations'].reshape((H, chunk_n, -1))
        rewards = roll['rewards'].reshape((H, chunk_n))
        masks = roll['masks'].reshape((H, chunk_n))
        valid = roll['rollout_valid'].reshape((H, chunk_n))

        # 2. Compute telescoping anchor-backed advantages along tau^- and average over valid rollout steps.
        v_prev = self.v_bar(prev_ob)
        if self.config.get('backup_mode', 'single') == 'multi':
            v_target = self.v_bar(anchor_ob)[None, :]
        else:
            v_target = self.v_bar(next_ob)
        advantages = rewards + discount * masks * v_target - v_prev
        return jnp.sum(advantages * valid, axis=0) / jnp.maximum(jnp.sum(valid, axis=0), 1.0)

    def _sample_pdl_u_rho_indices(self, ds_dict, valid_idxs, dataset_size, mb_batch_size):
        """Score a uniform pool of candidate anchors with U_rho(s) and resample via exp(anchor_beta * z(U_rho(s)))."""
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
            # Pad the final chunk to a fixed static shape so score_pdl_anchors_chunk compiles once.
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
        chosen_in_pool = np.random.choice(pool_size, size=mb_batch_size, replace=True, p=probs)
        return pool_local[chosen_in_pool]

    def sample_full_buffer_batch(self, train_dataset, mb_batch_size):
        """Sample real-dataset anchors using 'exp_value', 'frontier', 'pdl_u_rho', or 'uniform' weighting."""
        ds_dict = train_dataset.dataset._dict
        valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
        if self.config.get('exclude_terminal_anchors', False) and 'masks' in ds_dict:
            non_term = valid_idxs[ds_dict['masks'][valid_idxs] > 0.5]
            if len(non_term) > 0:
                valid_idxs = non_term
        n = len(valid_idxs)
        anchor_mode = self.config.get('anchor_mode', 'exp_value')
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
            chosen = np.random.choice(n, size=mb_batch_size, replace=True, p=p / p.sum())
        return train_dataset.sample(mb_batch_size, idxs=valid_idxs[chosen])

    @jax.jit
    def rollout(self, batch, rng):
        vg = batch['value_goals'] if 'value_goals' in batch else None
        ag = batch['actor_goals'] if 'actor_goals' in batch else None
        return snis_reverse_rollout(self.model, self.elites, self.config, batch['next_observations'], vg, ag, self.v_bar, rng)

    @classmethod
    def create(cls, seed, example_batch, config):
        # Build the COMBO policy networks, then swap in the reverse model.
        agent = COMBOAgent.create(seed, example_batch, config)
        ob_dim, action_dim = agent.config['ob_dim'], agent.config['action_dim']
        model_rng = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        ex_obs, ex_actions = example_batch['observations'], example_batch['actions']
        ex_goals = example_batch['actor_goals'] if 'actor_goals' in example_batch else None
        model_def = ModuleDict(reverse_model_defs(agent.config, ob_dim, action_dim, agent.config['model_layer_norm']))
        init_kwargs = dict(
            reverse_actor=(ex_obs, ex_goals),
            success=(ex_obs, ex_goals, ex_actions),
            reward=(ex_obs, ex_goals, ex_actions),
        )
        if agent.config.get('dynamics_type', 'gaussian') == 'rcfm':
            num_ens = agent.config['num_dynamics']
            ex_x = jnp.zeros((num_ens, ex_obs.shape[0], ob_dim))
            ex_t = jnp.zeros((num_ens, ex_obs.shape[0], 1))
            init_kwargs['reverse_dynamics'] = (ex_obs, ex_actions, ex_x, ex_t)
            init_kwargs['reverse_dynamics_1step'] = (ex_obs, ex_actions, ex_x)
            model_tx = build_model_tx(
                agent.config,
                rcfm_keys=('modules_reverse_dynamics',),
                frozen_keys=('modules_reverse_dynamics_1step',),
            )
        else:
            init_kwargs['reverse_dynamics'] = (ex_obs, ex_actions)
            model_tx = build_model_tx(agent.config)
        model_params = model_def.init(model_rng, **init_kwargs)['params']
        model = TrainState.create(model_def, model_params, tx=model_tx)
        return cls(agent.rng, network=agent.network, model=model, elites=agent.elites, config=agent.config)


def get_config(profile='ogbench'):
    config = combo_get_config(profile)
    config.update(
        dict(
            agent_name='combo_reverse_snis',
            num_rev_candidates=16,  # K
            num_mc_predecessors=5,  # M (<= num_elites)
            snis_beta=0.5,  # beta in softmax(D / beta)
            max_rollout_steps=5,  # H; keep equal to rollout_steps (buffer capacity).
            rollout_chunk=2048,
            clip_model_reward=False,  # COMBO labels model rewards without clipping.
            backup_mode='single',  # 'single' (chained) or 'multi' (all synthetic states bootstrap on the real anchor).
            full_buffer_exp=True,  # main_mbpo.py then calls sample_full_buffer_batch for rollout anchors.
            anchor_mode='exp_value',  # 'exp_value', 'frontier', 'pdl_u_rho', or 'uniform'.
            anchor_beta=3.0,  # P(anchor) ∝ exp(anchor_beta * z(score)) for 'exp_value', or 'pdl_u_rho' with u_rho_estimator='sample'.
            anchor_pool_mult=2,  # Candidate anchor pool multiplier for 'pdl_u_rho' (pool = mult * mb_batch_size).
            u_rho_estimator='lme',  # 'pdl_u_rho' score: 'lme' (log-mean-exp model-predecessor advantage), 'real' (logged predecessor), or 'sample'.
            anchor_lme_temp=2.0,  # Temperature of the 'lme' anchor weight mean exp(A_hat / temp); snis_beta gives exact rho*.
            anchor_unc_coef=0.0,  # 'lme' anchors: A_hat - coef * std over elite predecessors (model-uncertainty LCB).
            anchor_unc_quantile=1.0,  # 'lme' anchors: average only over (s, a) with ensemble std of A_hat below this quantile.
            anchor_score='lme',  # 'lme' anchor log-weight: 'lme' (log-mean-exp A_hat) or 'ens_lcb' (mean_m S_m - coef * std_m S_m).
            anchor_ens_coef=1.0,  # 'ens_lcb': weight on the std across elites of the per-member expected advantage S_m.
            anchor_weight_transform='none',  # 'lme'/'real' anchor weights: 'none', 'clip', 'norm_clip' (AWR), 'robust_clip' (median/IQR), or 'binary' (CRR).
            anchor_weight_clip=20.0,  # Weight cap for 'clip' / 'norm_clip' / 'robust_clip'.
            anchor_adv_beta=0.0,  # 'exp_value': add anchor_adv_beta * clip(z(logged A_hat), +-3) to the value logits (0 = exp_value).
            anchor_target_z=0.5,  # Center z-score when anchor_mode='frontier'.
            anchor_band_std=0.75,  # Width in z-score units when anchor_mode='frontier'.
            exclude_terminal_anchors=False,  # Exclude transitions with masks <= 0.5 from reverse anchors.
        )
    )
    return config
