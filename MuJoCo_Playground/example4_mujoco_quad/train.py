
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import os
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
from mujoco import mjx
import numpy as np

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#@title Import The Playground
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

import tools


def main():

    # 创建环境
    env_name = 'Go1JoystickFlatTerrain'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    # 训练参数
    ppo_params = locomotion_params.brax_ppo_config(env_name)  # 默认四足机器人训练参数

    # 训练参数设定
    ppo_params.num_timesteps = int(5*1e7)

    # 生成训练函数
    train_fn = tools.build_train_fun(env_name, ppo_params)

    # 开始训练
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    # 保存训练模型
    tools.save_model(params, relative_path="./MuJoCo_Playground/example4_mujoco_quad/saved_models")


if __name__ == "__main__":
    main()