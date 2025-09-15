# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

import mediapy as media
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp


#@title Import The Playground
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


# 训练函数
def build_train_fun(env_name, ppo_params):

    # 训练进度条函数
    def progress(num_steps, metrics):
        global pbar, times
        
        times.append(datetime.now())
        
        # 初始化进度条
        if pbar is None:
            pbar = tqdm(total=ppo_params["num_timesteps"], 
                        desc="Training", 
                        unit="steps",
                        position=0)
        
        # 更新进度条
        current_steps = num_steps
        pbar.n = current_steps
        
        # 更新描述信息
        reward = metrics.get("eval/episode_reward", 0)
        reward_std = metrics.get("eval/episode_reward_std", 0)
        
        pbar.set_postfix({
            'reward': f'{reward:.3f}',
            'reward_std': f'{reward_std:.3f}',
            'elapsed': str(times[-1] - times[0]).split('.')[0]
        })
        
        pbar.refresh()

    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
        seed=0,
    )

    return train_fn


# 保存参数
# 参考教程 https://github.com/google/brax/blob/main/notebooks/training.ipynb
# 参考帖子 https://github.com/google/brax/issues/438
def save_model(params, 
               relative_path: str) -> None:

    path_current = os.getcwd()
    path_save_dir = os.path.join(path_current, relative_path) 
    os.makedirs(path_save_dir, exist_ok=True)

    file_name = f"{datetime.now().strftime('%y-%m-%d_%H-%M')}_trained_params.pkl"
    path_save_file = os.path.join(path_save_dir, file_name)

    model.save_params(path_save_file, params)
    print(f"Model params saved to {path_save_file}")


# 保存视频
def save_video(frames, 
               fps: int,
               relative_path: str) -> None:
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    video_filename = os.path.join(relative_path, f"{timestamp}_quad.mp4")
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)
    
    media.write_video(video_filename, frames, fps=fps)
    print(f"Video saved to: {video_filename}")