
import numpy as np
import pandas as pd
from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class QNetwork(eqx.Module):
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    dense3: eqx.nn.Linear

    def __init__(self, in_features, hidden_features, out_features, key):
        keys = jax.random.split(key, 3)
        self.dense1 = eqx.nn.Linear(in_features, hidden_features, key=keys[0])
        self.dense2 = eqx.nn.Linear(hidden_features, hidden_features, key=keys[1])
        self.dense3 = eqx.nn.Linear(hidden_features, out_features, key=keys[2])

    def __call__(self, x):
        x = jax.nn.relu(jax.vmap(self.dense1)(x))
        x = jax.nn.relu(jax.vmap(self.dense2)(x))
        x = jax.vmap(self.dense3)(x)
        return x


def make_network(action_dim: int, key: jax.random.PRNGKey, input_shape: Tuple[int, ...]) -> QNetwork:
    in_features = np.prod(input_shape) 
    #print('in_features', in_features)
    hidden_features = 120  
    out_features = action_dim
    return QNetwork(in_features, hidden_features, out_features, key)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
