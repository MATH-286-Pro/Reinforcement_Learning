"""
MuJoCo Playground CartPole 环境配置
定义不同的训练配置和参数
"""
from ml_collections import config_dict
import jax.numpy as jp
from typing import Dict, Any


def get_default_cartpole_config() -> config_dict.ConfigDict:
    """获取默认 CartPole 环境配置"""
    config = config_dict.ConfigDict()
    
    # 环境基础参数
    config.xml_path = r"c:\Users\Jiajun Hu\Desktop\Code\Reinforcement_Learning\MY_TEST\project1\env\inverted_pendulum.xml"
    config.force_limit = 20.0
    config.ctrl_dt = 0.02
    config.sim_dt = 0.002
    config.episode_length = 1000
    config.backend = 'mjx'
    
    # 终止条件
    config.x_threshold = 2.4
    config.theta_threshold = 0.2094  # 12 degrees in radians
    
    # 奖励设计
    config.healthy_reward = 1.0
    config.ctrl_cost_weight = 0.001
    config.forward_reward_weight = 0.0
    config.position_penalty_weight = 0.1
    config.angle_reward_weight = 1.0
    
    # 随机化参数
    config.randomization = config_dict.ConfigDict()
    config.randomization.enable = False
    config.randomization.force_noise_std = 0.1
    config.randomization.mass_range = [0.8, 1.2]  # 质量随机化范围
    config.randomization.length_range = [0.95, 1.05]  # 长度随机化范围
    
    return config


def get_training_cartpole_config() -> config_dict.ConfigDict:
    """获取训练优化的 CartPole 环境配置"""
    config = get_default_cartpole_config()
    
    # 训练优化设置
    config.episode_length = 500  # 缩短训练时的 episode 长度
    config.ctrl_dt = 0.01  # 更高频率控制
    
    # 增强奖励设计
    config.healthy_reward = 1.0
    config.ctrl_cost_weight = 0.01  # 增加控制成本权重
    config.position_penalty_weight = 0.2
    config.angle_reward_weight = 2.0
    
    # 启用域随机化
    config.randomization.enable = True
    config.randomization.force_noise_std = 0.05
    
    return config


def get_evaluation_cartpole_config() -> config_dict.ConfigDict:
    """获取评估用的 CartPole 环境配置"""
    config = get_default_cartpole_config()
    
    # 评估设置
    config.episode_length = 1000  # 更长的评估长度
    
    # 关闭随机化
    config.randomization.enable = False
    
    # 更严格的终止条件
    config.x_threshold = 2.0
    config.theta_threshold = 0.174  # 10 degrees
    
    return config


def get_brax_ppo_config(env_name: str = 'CartPolePlayground') -> config_dict.ConfigDict:
    """获取 Brax PPO 训练配置"""
    config = config_dict.ConfigDict()
    
    # PPO 算法参数
    config.algorithm = 'ppo'
    config.num_timesteps = 2_000_000
    config.num_evals = 20
    config.reward_scaling = 1.0
    config.episode_length = 1000
    config.normalize_observations = True
    config.action_repeat = 1
    config.unroll_length = 5
    config.num_minibatches = 32
    config.num_updates_per_batch = 4
    config.discounting = 0.97
    config.learning_rate = 3e-4
    config.entropy_cost = 1e-2
    config.num_envs = 2048  # 大规模并行环境数量
    config.batch_size = 1024
    config.seed = 1
    
    # 网络架构
    config.network_factory = 'make_ppo_networks'
    config.policy_hidden_layer_sizes = (256, 256)
    config.value_hidden_layer_sizes = (256, 256)
    
    # 训练优化
    config.max_devices_per_host = None
    config.num_eval_envs = 128
    config.log_frequency = 20
    config.normalize_advantage = True
    config.gae_lambda = 0.95
    config.clipping_epsilon = 0.3
    config.max_gradient_norm = 1.0
    
    return config


def get_brax_sac_config(env_name: str = 'CartPolePlayground') -> config_dict.ConfigDict:
    """获取 Brax SAC 训练配置"""
    config = config_dict.ConfigDict()
    
    # SAC 算法参数
    config.algorithm = 'sac'
    config.num_timesteps = 1_000_000
    config.num_evals = 20
    config.reward_scaling = 1.0
    config.episode_length = 1000
    config.normalize_observations = True
    config.action_repeat = 1
    config.num_envs = 1024  # SAC 通常需要较少环境
    config.batch_size = 256
    config.min_replay_size = 10000
    config.max_replay_size = 1000000
    config.grad_updates_per_step = 1
    config.seed = 1
    
    # 网络架构
    config.network_factory = 'make_sac_networks'
    config.policy_hidden_layer_sizes = (256, 256)
    config.critic_hidden_layer_sizes = (256, 256)
    
    # SAC 特定参数
    config.learning_rate = 3e-4
    config.alpha_learning_rate = 3e-4
    config.alpha = 0.2
    config.discounting = 0.99
    config.tau = 0.005
    
    return config


# 预定义配置字典
CARTPOLE_CONFIGS = {
    'default': get_default_cartpole_config,
    'training': get_training_cartpole_config,
    'evaluation': get_evaluation_cartpole_config,
}

TRAINING_CONFIGS = {
    'ppo': get_brax_ppo_config,
    'sac': get_brax_sac_config,
}


def get_config(config_name: str = 'default') -> config_dict.ConfigDict:
    """获取指定名称的配置"""
    if config_name in CARTPOLE_CONFIGS:
        return CARTPOLE_CONFIGS[config_name]()
    else:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(CARTPOLE_CONFIGS.keys())}")


def get_training_config(algorithm: str = 'ppo') -> config_dict.ConfigDict:
    """获取指定算法的训练配置"""
    if algorithm in TRAINING_CONFIGS:
        return TRAINING_CONFIGS[algorithm]()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(TRAINING_CONFIGS.keys())}")


def print_config(config: config_dict.ConfigDict, title: str = "Configuration"):
    """打印配置信息"""
    print(f"\n=== {title} ===")
    for key, value in config.items():
        if isinstance(value, config_dict.ConfigDict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("=" * (len(title) + 8))


if __name__ == "__main__":
    # 测试配置
    print("测试 CartPole 环境配置...")
    
    # 默认配置
    default_config = get_config('default')
    print_config(default_config, "Default CartPole Config")
    
    # 训练配置
    training_config = get_config('training')
    print_config(training_config, "Training CartPole Config")
    
    # PPO 训练配置
    ppo_config = get_training_config('ppo')
    print_config(ppo_config, "PPO Training Config")
    
    # SAC 训练配置
    sac_config = get_training_config('sac')
    print_config(sac_config, "SAC Training Config")
    
    print("\n配置测试完成！")