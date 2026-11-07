import ml_collections
from agents.iql import IQLAgent


class NStepIQLAgent(IQLAgent):
    """Same-Episode n-Step IQL agent.

    Uses NStepCDataset so the critic backs up n-step discounted returns strictly within the same dataset trajectory,
    isolating same-episode multi-step credit propagation from cross-episode trajectory stitching.
    """

    def critic_loss(self, batch, grad_params):
        """Compute the same-episode n-step IQL critic loss."""
        value_goals = batch.get('value_goals', None)
        next_obs = batch.get('nstep_next_observations', batch['next_observations'])
        nstep_rewards = batch.get('nstep_rewards', batch['rewards'])
        nstep_discounts = batch.get('nstep_discounts', self.config['discount'] * batch['masks'])

        next_v = self.network.select('value')(next_obs, value_goals)
        q_target = nstep_rewards + nstep_discounts * next_v

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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nstep_iql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            nstep=5,
            actor_loss='ddpgbc',
            alpha=1.0,
            const_std=True,
            dataset_class='NStepCDataset',
        )
    )
    return config
