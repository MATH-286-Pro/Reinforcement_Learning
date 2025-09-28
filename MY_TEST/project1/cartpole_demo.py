"""
简单的 CartPole 环境使用示例
演示如何使用 MuJoCo Playground CartPole 环境
"""
import os
import sys
import jax
import jax.numpy as jp
import numpy as np

# 添加环境路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'env'))

from env.mujoco_playground_cartpole_env import create_cartpole_env
from env.cartpole_config import get_config


def demo_basic_usage():
    """演示基本使用方法"""
    print("=== CartPole 环境基本使用演示 ===")
    
    # 1. 创建环境
    config = get_config('default')
    env = create_cartpole_env(**config)
    
    print(f"环境创建完成!")
    print(f"观测空间维度: {env.observation_size}")
    print(f"动作空间维度: {env.action_size}")
    
    # 2. 重置环境
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print(f"\n初始状态:")
    print(f"观测: {state.obs}")
    print(f"奖励: {state.reward}")
    print(f"完成: {state.done}")
    
    # 3. 执行几步
    print(f"\n执行 5 步:")
    for step in range(5):
        # 简单的控制策略：基于摆杆角度
        x, x_dot, cos_theta, sin_theta, theta_dot = state.obs
        theta = jp.arctan2(sin_theta, cos_theta)
        
        # PID 控制
        force = -10.0 * theta - 2.0 * theta_dot - 0.5 * x - 0.5 * x_dot
        action = jp.clip(jp.array([force / 20.0]), -1.0, 1.0)
        
        # 执行动作
        state = env.step(state, action)
        
        print(f"步骤 {step + 1}: 动作={float(action[0]):.3f}, "
              f"奖励={float(state.reward):.3f}, "
              f"角度={float(theta):.3f} rad, "
              f"完成={state.done}")
        
        if state.done:
            print("环境终止!")
            break


def demo_vectorized_envs():
    """演示向量化环境使用"""
    print("\n=== 向量化环境演示 ===")
    
    # 创建环境
    config = get_config('default')
    env = create_cartpole_env(**config)
    
    # 向量化重置和步进
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step)
    
    # 批量重置多个环境
    num_envs = 10
    rngs = jax.random.split(jax.random.PRNGKey(0), num_envs)
    states = vmap_reset(rngs)
    
    print(f"批量创建 {num_envs} 个环境")
    print(f"观测形状: {states.obs.shape}")
    print(f"奖励形状: {states.reward.shape}")
    
    # 批量执行动作
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), 
        shape=(num_envs, 1), 
        minval=-1.0, 
        maxval=1.0
    )
    
    states = vmap_step(states, actions)
    
    print(f"批量步进后:")
    print(f"平均奖励: {jp.mean(states.reward):.3f}")
    print(f"完成环境数: {jp.sum(states.done)}")


def demo_performance_test():
    """演示性能测试"""
    print("\n=== 性能测试演示 ===")
    
    import time
    
    # 创建环境
    config = get_config('default')
    env = create_cartpole_env(**config)
    
    # JIT 编译函数
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step))
    
    # 单环境性能测试
    print("单环境性能测试...")
    rng = jax.random.PRNGKey(42)
    
    # 预热
    state = jit_reset(rng)
    action = jp.array([0.0])
    state = jit_step(state, action)
    
    # 计时
    num_steps = 1000
    start_time = time.time()
    
    for i in range(num_steps):
        action = jax.random.uniform(
            jax.random.PRNGKey(i), 
            shape=(1,), 
            minval=-1.0, 
            maxval=1.0
        )
        state = jit_step(state, action)
    
    single_env_time = time.time() - start_time
    single_env_fps = num_steps / single_env_time
    
    print(f"单环境: {single_env_fps:.0f} 步/秒")
    
    # 向量化环境性能测试
    print("向量化环境性能测试...")
    num_envs = 100
    rngs = jax.random.split(jax.random.PRNGKey(0), num_envs)
    
    # 预热
    states = vmap_reset(rngs)
    actions = jp.zeros((num_envs, 1))
    states = vmap_step(states, actions)
    
    # 计时
    start_time = time.time()
    
    for i in range(num_steps):
        actions = jax.random.uniform(
            jax.random.PRNGKey(i), 
            shape=(num_envs, 1), 
            minval=-1.0, 
            maxval=1.0
        )
        states = vmap_step(states, actions)
    
    vector_env_time = time.time() - start_time
    vector_env_fps = (num_steps * num_envs) / vector_env_time
    
    print(f"向量化环境 ({num_envs} 个): {vector_env_fps:.0f} 步/秒")
    print(f"加速比: {vector_env_fps / single_env_fps:.1f}x")


def main():
    """主函数"""
    print("MuJoCo Playground CartPole 环境使用示例")
    print("=" * 50)
    
    # 检查 JAX 设备
    print(f"JAX 设备: {jax.devices()}")
    print(f"默认后端: {jax.default_backend()}")
    
    # 运行演示
    demo_basic_usage()
    demo_vectorized_envs()
    demo_performance_test()
    
    print("\n演示完成! 🎉")
    print("\n接下来你可以:")
    print("1. 运行 cartpole_parallel_train.py 进行大规模训练")
    print("2. 运行 cartpole_test_visualize.py 进行详细测试")
    print("3. 查看 README.md 了解更多使用方法")


if __name__ == "__main__":
    main()