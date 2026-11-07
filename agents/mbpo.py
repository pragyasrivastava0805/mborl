import ml_collections
from agents.mopo import MOPOAgent, get_config as get_mopo_config


class MBPOAgent(MOPOAgent):
    """MBPO (Janner et al., 2019) agent (unpenalized model-based rollouts, beta = 0.0)."""
    pass


def get_config():
    config = get_mopo_config()
    config.agent_name = 'mbpo'
    config.beta = 0.0
    config.dataset_class = 'CDataset'
    return config
