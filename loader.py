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

           "halfcheetah-full-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_full_replay-v2.hdf5",
           "walker2d-full-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_full_replay-v2.hdf5",
           "hopper-full-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_full_replay-v2.hdf5",
           "ant-full-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_full_replay-v2.hdf5",
           }

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load(name, link):
    os.makedirs("data", exist_ok=True)
    h5path = os.path.join('.', 'data', name + '.hdf5')
    if not os.path.exists(h5path):
        urllib.request.urlretrieve(link, h5path)

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