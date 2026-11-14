import flax
import jax
import numpy as np
import jax.numpy as jnp
import ml_collections
import optax

from agents.combo import COMBOAgent, get_config as combo_get_config
from agents.reverse_rollouts_snis import reverse_model_defs, reverse_model_loss, snis_reverse_rollout
from utils.flax_utils import ModuleDict, TrainState


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

    def v_bar(self, obs, value_goals=None, actor_goals=None):
        actions = jnp.clip(self.network.select('actor')(obs, goals=actor_goals).mode(), -1.0, 1.0)
        return self.network.select('target_critic')(obs, goals=value_goals, actions=actions).min(axis=0)

    @jax.jit
    def eval_v_pi_chunk(self, obs):
        return self.v_bar(obs)

    def sample_full_buffer_batch(self, train_dataset, mb_batch_size):
        """Anchors from the real dataset: P(s) ∝ exp(anchor_beta * z(V_pi(s))) over all dataset next-states."""
        ds_dict = train_dataset.dataset._dict
        valid_idxs = getattr(train_dataset, 'valid_idxs', np.arange(train_dataset.size))
        if 'next_observations' in ds_dict:
            all_next_obs = ds_dict['next_observations'][valid_idxs]
        else:
            all_next_obs = ds_dict['observations'][np.minimum(valid_idxs + 1, train_dataset.size - 1)]
        n = len(valid_idxs)
        if int(self.network.step) == 0:
            chosen = np.random.randint(0, n, size=mb_batch_size)
        else:
            v = np.concatenate([np.asarray(self.eval_v_pi_chunk(all_next_obs[s:s + 100000])) for s in range(0, n, 100000)])
            v = v.astype(np.float64)
            z = np.clip((v - v.mean()) / max(v.std(), 1e-5), -5.0, 5.0)
            logits = float(self.config['anchor_beta']) * z
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
        model_params = model_def.init(
            model_rng,
            reverse_dynamics=(ex_obs, ex_actions),
            reverse_actor=(ex_obs, ex_goals),
            success=(ex_obs, ex_goals, ex_actions),
            reward=(ex_obs, ex_goals, ex_actions),
        )['params']
        model = TrainState.create(model_def, model_params, tx=optax.adam(learning_rate=agent.config['model_lr']))
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
            anchor_beta=3.0,  # P(anchor) ∝ exp(anchor_beta * z(V_pi(s))).
        )
    )
    return config
