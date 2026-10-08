import os
import h5py
import numpy as np
import urllib.request
from tqdm import tqdm


SOURCES = {"halfcheetah-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
           "walker2d-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5",
           "hopper-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
           "ant-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_replay-v2.hdf5",

           "halfcheetah-random-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_random-v2.hdf5",
           "walker2d-random-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_random-v2.hdf5",
           "hopper-random-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_random-v2.hdf5",
           "ant-random-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_random-v2.hdf5",

           "halfcheetah-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
           "walker2d-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
           "hopper-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
           "ant-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-v2.hdf5",

           "halfcheetah-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5",
           "walker2d-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5",
           "hopper-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5",
           "ant-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_expert-v2.hdf5",

           "halfcheetah-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5",
           "walker2d-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_expert-v2.hdf5",
           "hopper-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
           "ant-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_expert-v2.hdf5",
           }

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


# Same cache as D4RL (d4rl/offline_env.py), so the notebook reads the exact files the training runs use.
# The SOURCES urls are D4RL's own urls, and D4RL names each file after the url.
DATA_DIR = os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets"))


def load(name, link):
    os.makedirs(DATA_DIR, exist_ok=True)
    h5path = os.path.join(DATA_DIR, os.path.basename(link))
    if not os.path.exists(h5path):
        # download to a temp file so an interrupted download (e.g. disk quota) never leaves a truncated cache
        tmppath = h5path + '.part'
        try:
            urllib.request.urlretrieve(link, tmppath)
            os.replace(tmppath, h5path)
        finally:
            if os.path.exists(tmppath):
                os.remove(tmppath)

    dataset = dict()
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]
    return dataset["observations"], dataset["actions"], dataset["rewards"], \
           np.invert(np.logical_or(dataset["terminals"], dataset["timeouts"]))


if __name__ == "__main__":

    for name, link in tqdm(SOURCES.items()):
        states, actions, rewards, not_dones = load(name, link)

        print(name)
        print(states.shape, actions.shape, rewards.shape, not_dones.shape)