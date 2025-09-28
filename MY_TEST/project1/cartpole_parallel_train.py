"""
MuJoCo Playground CartPole 大规模并行训练脚本
使用 JAX/Brax 框架进行高效并行训练
"""
import os
import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable

import jax
import jax.numpy as jp
import numpy as np
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.io import model
from flax.training import checkpoints, orbax_utils
import mediapy as media
from tqdm import tqdm

# 导入我们的环境和配置
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mujoco_playground_cartpole_env import CartPoleEnv, create_cartpole_env
from cartpole_config import get_config, get_training_config, print_config


class CartPoleTrainer:
    """CartPole 环境大规模并行训练器"""
    
    def __init__(
        self,
        algorithm: str = 'ppo',
        env_config_name: str = 'training',
        num_envs: int = 2048,
        num_timesteps: int = 2_000_000,
        save_dir: str = './cartpole_models',
        device_count: Optional[int] = None
    ):
        self.algorithm = algorithm
        self.env_config_name = env_config_name
        self.num_envs = num_envs
        self.num_timesteps = num_timesteps
        self.save_dir = save_dir
        
        # 设置 JAX 设备
        if device_count is None:
            self.device_count = jax.device_count()
        else:
            self.device_count = min(device_count, jax.device_count())
        
        print(f"使用设备数量: {self.device_count}")
        print(f"JAX 设备: {jax.devices()}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化环境和配置
        self._setup_environment()
        self._setup_training_config()
    
    def _setup_environment(self):
        """设置环境"""
        print("设置 CartPole 环境...")
        
        # 获取环境配置
        self.env_config = get_config(self.env_config_name)
        print_config(self.env_config, f"Environment Config ({self.env_config_name})")
        
        # 创建环境
        self.env = create_cartpole_env(**self.env_config)
        
        print(f"环境创建完成:")
        print(f"  观测空间维度: {self.env.observation_size}")
        print(f"  动作空间维度: {self.env.action_size}")
    
    def _setup_training_config(self):
        """设置训练配置"""
        print(f"设置 {self.algorithm.upper()} 训练配置...")
        
        # 获取训练配置
        self.train_config = get_training_config(self.algorithm)
        
        # 更新配置参数
        self.train_config.num_envs = self.num_envs
        self.train_config.num_timesteps = self.num_timesteps
        self.train_config.episode_length = self.env_config.episode_length
        
        print_config(self.train_config, f"{self.algorithm.upper()} Training Config")
    
    def train(self) -> Tuple[Any, Dict[str, Any]]:
        """执行训练"""
        print(f"\n开始 {self.algorithm.upper()} 训练...")
        print(f"训练环境数量: {self.num_envs}")
        print(f"总训练步数: {self.num_timesteps:,}")
        
        start_time = time.time()
        
        # 选择训练函数
        if self.algorithm == 'ppo':
            train_fn = ppo.train
        elif self.algorithm == 'sac':
            train_fn = sac.train
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # 执行训练
        make_inference_fn, params, training_metrics = train_fn(
            environment=self.env,
            num_timesteps=self.train_config.num_timesteps,
            num_evals=self.train_config.num_evals,
            reward_scaling=self.train_config.reward_scaling,
            episode_length=self.train_config.episode_length,
            normalize_observations=self.train_config.normalize_observations,
            action_repeat=self.train_config.action_repeat,
            unroll_length=getattr(self.train_config, 'unroll_length', 5),
            num_minibatches=getattr(self.train_config, 'num_minibatches', 32),
            num_updates_per_batch=getattr(self.train_config, 'num_updates_per_batch', 4),
            discounting=self.train_config.discounting,
            learning_rate=self.train_config.learning_rate,
            entropy_cost=getattr(self.train_config, 'entropy_cost', 1e-2),
            num_envs=self.train_config.num_envs,
            batch_size=self.train_config.batch_size,
            seed=self.train_config.seed,
            max_devices_per_host=self.train_config.max_devices_per_host,
            num_eval_envs=getattr(self.train_config, 'num_eval_envs', 128),
            log_frequency=getattr(self.train_config, 'log_frequency', 20),
            normalize_advantage=getattr(self.train_config, 'normalize_advantage', True),
            progress_fn=self._progress_callback,
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 用时: {training_time:.2f} 秒")
        
        # 保存模型
        model_path = self._save_model(params, training_metrics)
        print(f"模型已保存到: {model_path}")
        
        return make_inference_fn, params, training_metrics
    
    def _progress_callback(self, step: int, metrics: Dict[str, Any]):
        """训练进度回调"""
        if step % 100 == 0:  # 每100步打印一次
            eval_reward = metrics.get('eval/episode_reward', 0)
            print(f"Step {step:,}: Eval Reward = {eval_reward:.2f}")
    
    def _save_model(self, params: Any, metrics: Dict[str, Any]) -> str:
        """保存训练好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cartpole_{self.algorithm}_{timestamp}"
        model_path = os.path.join(self.save_dir, f"{model_name}.pkl")
        
        # 保存参数
        model.save_params(model_path, params)
        
        # 保存训练信息
        info_path = os.path.join(self.save_dir, f"{model_name}_info.npz")
        info_data = {
            'algorithm': self.algorithm,
            'env_config_name': self.env_config_name,
            'num_envs': self.num_envs,
            'num_timesteps': self.num_timesteps,
            'final_eval_reward': metrics.get('eval/episode_reward', [])[-1] if metrics.get('eval/episode_reward') else 0,
        }
        np.savez(info_path, **info_data)
        
        return model_path
    
    def evaluate(
        self,
        params: Any,
        make_inference_fn: Callable,
        num_episodes: int = 10,
        render: bool = False,
        save_video: bool = False
    ) -> Dict[str, float]:
        """评估训练好的模型"""
        print(f"\n评估模型 ({num_episodes} episodes)...")
        
        # 创建评估环境
        eval_env_config = get_config('evaluation')
        eval_env = create_cartpole_env(**eval_env_config)
        
        # 创建推理函数
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        
        # 评估指标
        episode_rewards = []
        episode_lengths = []
        success_episodes = 0  # 完成整个 episode 的次数
        
        for episode in range(num_episodes):
            rng = jax.random.PRNGKey(episode)
            state = jit_reset(rng)
            
            episode_reward = 0.0
            episode_length = 0
            
            rollout_states = [] if save_video else None
            
            for step in range(eval_env_config.episode_length):
                if save_video:
                    rollout_states.append(state)
                
                # 选择动作
                action, _ = jit_inference_fn(state.obs, jax.random.PRNGKey(step))
                
                # 执行动作
                state = jit_step(state, action)
                episode_reward += float(state.reward)
                episode_length += 1
                
                if state.done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode_length >= eval_env_config.episode_length * 0.9:  # 90% 完成度算成功
                success_episodes += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            # 保存视频（如果需要）
            if save_video and episode == 0:  # 只保存第一个 episode
                self._save_rollout_video(rollout_states, f"cartpole_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # 计算统计信息
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_episodes / num_episodes,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
        }
        
        print(f"\n评估结果:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value:.3f}")
        
        return eval_stats
    
    def _save_rollout_video(self, states, filename: str):
        """保存 rollout 视频"""
        try:
            # 这里需要实现视频保存逻辑
            print(f"视频保存功能需要进一步实现: {filename}")
        except Exception as e:
            print(f"保存视频失败: {e}")


def load_and_evaluate(model_path: str, num_episodes: int = 10):
    """加载并评估保存的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载模型参数
    params = model.load_params(model_path)
    
    # 加载模型信息
    info_path = model_path.replace('.pkl', '_info.npz')
    if os.path.exists(info_path):
        info = np.load(info_path, allow_pickle=True)
        algorithm = str(info['algorithm'])
        env_config_name = str(info['env_config_name'])
        print(f"算法: {algorithm}, 环境配置: {env_config_name}")
    else:
        algorithm = 'ppo'
        env_config_name = 'evaluation'
        print("未找到模型信息，使用默认设置")
    
    # 创建训练器并评估
    trainer = CartPoleTrainer(algorithm=algorithm, env_config_name=env_config_name)
    
    # 重新创建 make_inference_fn（这里需要根据实际情况调整）
    if algorithm == 'ppo':
        train_fn = ppo.train
    else:
        train_fn = sac.train
    
    # 创建临时训练来获取 make_inference_fn
    make_inference_fn, _, _ = train_fn(
        environment=trainer.env,
        num_timesteps=1,  # 最小训练步数
        num_envs=1,
        seed=42
    )
    
    # 评估模型
    stats = trainer.evaluate(params, make_inference_fn, num_episodes=num_episodes)
    return stats


def main():
    """主训练函数"""
    print("MuJoCo Playground CartPole 大规模并行训练")
    print("=" * 50)
    
    # 检查 JAX 设备
    print(f"可用设备: {jax.devices()}")
    print(f"设备数量: {jax.device_count()}")
    
    # 创建训练器
    trainer = CartPoleTrainer(
        algorithm='ppo',  # 或 'sac'
        env_config_name='training',
        num_envs=2048,  # 大规模并行
        num_timesteps=2_000_000,
        save_dir='./cartpole_models'
    )
    
    # 执行训练
    make_inference_fn, params, metrics = trainer.train()
    
    # 评估模型
    eval_stats = trainer.evaluate(
        params, 
        make_inference_fn, 
        num_episodes=10,
        save_video=True
    )
    
    print(f"\n最终评估结果: {eval_stats}")
    print("训练完成!")


if __name__ == "__main__":
    main()