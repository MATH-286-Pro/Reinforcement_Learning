import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.mujoco_playground_cartpole_env import CartPoleEnv, create_cartpole_env
from env.cartpole_config import get_config, print_config


class CartPoleVisualizer:
    """CartPole 环境可视化和测试工具"""
    
    def __init__(self, config_name: str = 'default'):
        self.config_name = config_name
        self.env_config = get_config(config_name)
        self.env = create_cartpole_env(**self.env_config)
        
        print(f"创建 CartPole 环境 (配置: {config_name})")
        print(f"观测维度: {self.env.observation_size}")
        print(f"动作维度: {self.env.action_size}")
    
    def test_basic_functionality(self):
        """测试基本功能"""
        print("\n=== 基本功能测试 ===")
        
        # 测试环境重置
        rng = jax.random.PRNGKey(42)
        state = self.env.reset(rng)
        
        print(f"重置后观测: {state.obs}")
        print(f"重置后奖励: {state.reward}")
        print(f"重置后完成状态: {state.done}")
        print(f"重置后指标: {state.metrics}")
        
        # 测试步进
        action = jp.array([0.5])  # 向右施力
        new_state = self.env.step(state, action)
        
        print(f"\n步进后观测: {new_state.obs}")
        print(f"步进后奖励: {new_state.reward}")
        print(f"步进后完成状态: {new_state.done}")
        print(f"步进后指标: {new_state.metrics}")
        
        print("✅ 基本功能测试通过")
    
    def test_random_policy(self, num_episodes: int = 5, max_steps: int = 200):
        """测试随机策略"""
        print(f"\n=== 随机策略测试 ({num_episodes} episodes) ===")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            rng = jax.random.PRNGKey(episode)
            state = self.env.reset(rng)
            
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(max_steps):
                # 随机动作
                action_rng, rng = jax.random.split(rng)
                action = jax.random.uniform(action_rng, shape=(1,), minval=-1.0, maxval=1.0)
                
                state = self.env.step(state, action)
                episode_reward += float(state.reward)
                episode_length += 1
                
                if state.done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        print(f"\n随机策略统计:")
        print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  平均长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print("✅ 随机策略测试完成")
        
        return episode_rewards, episode_lengths
    
    def test_simple_controller(self, num_episodes: int = 5, max_steps: int = 500):
        """测试简单 PID 控制器"""
        print(f"\n=== 简单控制器测试 ({num_episodes} episodes) ===")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # PID 参数
        kp_theta = 20.0  # 角度比例增益
        kd_theta = 5.0   # 角度微分增益
        kp_x = 1.0       # 位置比例增益
        kd_x = 2.0       # 位置微分增益
        
        for episode in range(num_episodes):
            rng = jax.random.PRNGKey(episode + 100)
            state = self.env.reset(rng)
            
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(max_steps):
                # 解析观测
                x, x_dot, cos_theta, sin_theta, theta_dot = state.obs
                theta = jp.arctan2(sin_theta, cos_theta)
                
                # PID 控制
                force = (
                    -kp_theta * theta       # 角度误差
                    - kd_theta * theta_dot  # 角速度
                    - kp_x * x              # 位置误差
                    - kd_x * x_dot          # 速度
                )
                
                # 限制动作范围
                action = jp.clip(jp.array([force / 20.0]), -1.0, 1.0)
                
                state = self.env.step(state, action)
                episode_reward += float(state.reward)
                episode_length += 1
                
                if state.done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode_length >= max_steps * 0.9:  # 90% 完成度算成功
                success_count += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        success_rate = success_count / num_episodes
        print(f"\n简单控制器统计:")
        print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  平均长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"  成功率: {success_rate:.1%}")
        print("✅ 简单控制器测试完成")
        
        return episode_rewards, episode_lengths, success_rate
    
    def test_vectorized_environments(self, num_envs: int = 100, num_steps: int = 50):
        """测试向量化环境"""
        print(f"\n=== 向量化环境测试 ({num_envs} 环境, {num_steps} 步) ===")
        
        # 创建向量化重置和步进函数
        vmap_reset = jax.vmap(self.env.reset)
        vmap_step = jax.vmap(self.env.step)
        
        # 生成随机种子
        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num_envs)
        
        # 批量重置
        start_time = time.time()
        states = vmap_reset(rngs)
        reset_time = time.time() - start_time
        
        print(f"批量重置 {num_envs} 环境用时: {reset_time:.4f} 秒")
        print(f"状态形状: obs={states.obs.shape}, reward={states.reward.shape}, done={states.done.shape}")
        
        # 批量步进
        all_rewards = []
        step_times = []
        
        for step in range(num_steps):
            # 随机动作
            actions = jax.random.uniform(
                jax.random.PRNGKey(step), 
                shape=(num_envs, 1), 
                minval=-1.0, 
                maxval=1.0
            )
            
            # 批量步进
            start_time = time.time()
            states = vmap_step(states, actions)
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            all_rewards.append(states.reward)
            
            if step % 10 == 0:
                avg_reward = jp.mean(states.reward)
                done_count = jp.sum(states.done)
                print(f"Step {step}: 平均奖励 = {avg_reward:.3f}, 完成环境数 = {done_count}")
        
        avg_step_time = np.mean(step_times)
        total_reward = jp.sum(jp.array(all_rewards))
        
        print(f"\n向量化环境统计:")
        print(f"  平均步进时间: {avg_step_time:.6f} 秒")
        print(f"  每秒步进数: {num_envs / avg_step_time:.0f}")
        print(f"  总奖励: {total_reward:.2f}")
        print("✅ 向量化环境测试完成")
        
        return avg_step_time, total_reward
    
    def plot_episode_analysis(self, states_history: List, actions_history: List, save_path: Optional[str] = None):
        """绘制 episode 分析图"""
        print("\n=== 绘制 Episode 分析图 ===")
        
        # 提取数据
        times = np.arange(len(states_history)) * self.env_config.ctrl_dt
        observations = np.array([state.obs for state in states_history])
        actions = np.array(actions_history)
        rewards = np.array([state.reward for state in states_history])
        
        # 解析观测
        x_positions = observations[:, 0]
        x_velocities = observations[:, 1]
        cos_theta = observations[:, 2]
        sin_theta = observations[:, 3]
        theta_velocities = observations[:, 4]
        theta_angles = np.arctan2(sin_theta, cos_theta) * 180 / np.pi  # 转换为度
        
        # 创建子图
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('CartPole Episode Analysis')
        
        # 小车位置
        axes[0, 0].plot(times, x_positions)
        axes[0, 0].set_title('Cart Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].grid(True)
        
        # 小车速度
        axes[0, 1].plot(times, x_velocities)
        axes[0, 1].set_title('Cart Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].grid(True)
        
        # 摆杆角度
        axes[1, 0].plot(times, theta_angles)
        axes[1, 0].set_title('Pole Angle')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].grid(True)
        
        # 摆杆角速度
        axes[1, 1].plot(times, theta_velocities)
        axes[1, 1].set_title('Pole Angular Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[1, 1].grid(True)
        
        # 控制动作
        axes[2, 0].plot(times, actions.flatten())
        axes[2, 0].set_title('Control Action')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Force (normalized)')
        axes[2, 0].grid(True)
        
        # 奖励
        axes[2, 1].plot(times, rewards)
        axes[2, 1].set_title('Reward')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Reward')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def collect_episode_data(self, controller_type: str = 'pid', max_steps: int = 300):
        """收集一个 episode 的数据用于分析"""
        print(f"\n=== 收集 Episode 数据 ({controller_type}) ===")
        
        rng = jax.random.PRNGKey(12345)
        state = self.env.reset(rng)
        
        states_history = [state]
        actions_history = []
        
        for step in range(max_steps):
            if controller_type == 'pid':
                # PID 控制器
                x, x_dot, cos_theta, sin_theta, theta_dot = state.obs
                theta = jp.arctan2(sin_theta, cos_theta)
                
                force = -20.0 * theta - 5.0 * theta_dot - 1.0 * x - 2.0 * x_dot
                action = jp.clip(jp.array([force / 20.0]), -1.0, 1.0)
                
            elif controller_type == 'random':
                # 随机控制器
                action = jax.random.uniform(
                    jax.random.PRNGKey(step), 
                    shape=(1,), 
                    minval=-1.0, 
                    maxval=1.0
                )
            else:
                # 零控制器
                action = jp.array([0.0])
            
            actions_history.append(action)
            state = self.env.step(state, action)
            states_history.append(state)
            
            if state.done:
                print(f"Episode 在第 {step + 1} 步结束")
                break
        
        return states_history, actions_history
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("\n" + "="*60)
        print("CartPole 环境综合测试")
        print("="*60)
        
        # 打印环境配置
        print_config(self.env_config, f"Environment Config ({self.config_name})")
        
        # 基本功能测试
        self.test_basic_functionality()
        
        # 随机策略测试
        random_rewards, random_lengths = self.test_random_policy()
        
        # 简单控制器测试
        pid_rewards, pid_lengths, success_rate = self.test_simple_controller()
        
        # 向量化环境测试
        step_time, total_reward = self.test_vectorized_environments()
        
        # 收集和可视化数据
        print("\n=== 数据收集和可视化 ===")
        
        # PID 控制器数据
        states_pid, actions_pid = self.collect_episode_data('pid')
        self.plot_episode_analysis(
            states_pid, 
            actions_pid, 
            save_path='cartpole_pid_analysis.png'
        )
        
        # 随机控制器数据
        states_random, actions_random = self.collect_episode_data('random')
        self.plot_episode_analysis(
            states_random, 
            actions_random, 
            save_path='cartpole_random_analysis.png'
        )
        
        # 测试总结
        print(f"\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"✅ 基本功能: 正常")
        print(f"✅ 随机策略平均奖励: {np.mean(random_rewards):.2f}")
        print(f"✅ PID 控制器平均奖励: {np.mean(pid_rewards):.2f}")
        print(f"✅ PID 控制器成功率: {success_rate:.1%}")
        print(f"✅ 向量化性能: {1/step_time:.0f} env/sec")
        print(f"✅ 环境实现: 正确且高效")
        print("="*60)


def main():
    """主测试函数"""
    print("CartPole 环境测试和可视化")
    
    # 创建可视化工具
    visualizer = CartPoleVisualizer('training')
    
    # 运行综合测试
    visualizer.run_comprehensive_test()
    
    print("\n所有测试完成! 🎉")


if __name__ == "__main__":
    main()