"""
简化版 MuJoCo 倒立摆环境
基于 Gymnasium 接口的简单实现，去除复杂的 domain randomization
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class SimplePendulumEnv(gym.Env):
    """
    简化版 MuJoCo 倒立摆环境
    观测: [x, x_dot, cos(theta), sin(theta), theta_dot]
    动作: 单个连续值 [-1, 1]，表示施加在小车上的力
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # 环境参数
        self.max_episode_steps = 500
        self.dt = 0.02  # 时间步长
        self.force_mag = 10.0  # 力的缩放因子
        
        # 终止条件
        self.x_threshold = 2.4  # 小车位置限制
        self.theta_threshold = 12 * 2 * np.pi / 360  # 角度限制（弧度）
        
        # 加载 MuJoCo 模型
        xml_path = r"c:\Users\Jiajun Hu\Desktop\Code\Reinforcement_Learning\MY_TEST\project1\env\inverted_pendulum.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取关节和执行器的索引
        self.slide_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide")
        self.hinge_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_force")
        
        # 定义观测和动作空间
        high = np.array([
            np.finfo(np.float32).max,  # x
            np.finfo(np.float32).max,  # x_dot
            1.0,  # cos(theta)
            1.0,  # sin(theta)
            np.finfo(np.float32).max,  # theta_dot
        ])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 渲染相关
        self.viewer = None
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # 重置 MuJoCo 数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 添加小的随机初始状态
        if seed is not None:
            np.random.seed(seed)
            
        # 小车位置和速度的小随机扰动
        self.data.qpos[self.slide_joint_id] = np.random.uniform(-0.05, 0.05)
        self.data.qvel[self.slide_joint_id] = np.random.uniform(-0.05, 0.05)
        
        # 摆杆角度和角速度的小随机扰动
        self.data.qpos[self.hinge_joint_id] = np.random.uniform(-0.05, 0.05)
        self.data.qvel[self.hinge_joint_id] = np.random.uniform(-0.05, 0.05)
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    def step(self, action):
        # 确保动作是 numpy 数组
        if np.isscalar(action):
            action = np.array([action])
        
        # 限制动作范围并应用力
        action = np.clip(action, -1.0, 1.0)
        force = action[0] * self.force_mag
        
        # 设置控制输入
        self.data.ctrl[self.actuator_id] = force
        
        # 执行物理步进
        mujoco.mj_step(self.model, self.data)
        
        # 获取观测
        observation = self._get_obs()
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        self.step_count += 1
        
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """获取观测值"""
        x = self.data.qpos[self.slide_joint_id]
        x_dot = self.data.qvel[self.slide_joint_id]
        theta = self.data.qpos[self.hinge_joint_id]
        theta_dot = self.data.qvel[self.hinge_joint_id]
        
        return np.array([
            x,
            x_dot,
            np.cos(theta),
            np.sin(theta),
            theta_dot
        ], dtype=np.float32)
    
    def _compute_reward(self):
        """计算奖励"""
        x = self.data.qpos[self.slide_joint_id]
        theta = self.data.qpos[self.hinge_joint_id]
        
        # 基础奖励：保持摆杆竖直
        reward = np.cos(theta)
        
        # 惩罚小车偏离中心
        reward -= 0.1 * (x ** 2)
        
        # 如果摆杆保持竖直，给予额外奖励
        if abs(theta) < 0.1:  # 约 5.7 度
            reward += 1.0
            
        return reward
    
    def _is_terminated(self):
        """检查是否终止"""
        x = self.data.qpos[self.slide_joint_id]
        theta = self.data.qpos[self.hinge_joint_id]
        
        # 小车超出边界或摆杆倾倒过多
        if abs(x) > self.x_threshold:
            return True
        if abs(theta) > self.theta_threshold:
            return True
            
        return False
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except Exception as e:
                    print(f"Failed to launch viewer: {e}")
                    return None
            
            if self.viewer is not None:
                self.viewer.sync()
                
        elif self.render_mode == "rgb_array":
            # 离屏渲染
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            renderer.close()
            return pixels
            
        return None
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

###########################################################

def test_visualization():
    """测试可视化功能"""
    print("测试MuJoCo倒立摆可视化...")
    
    # 创建环境并启用人类可视化
    env = SimplePendulumEnv(render_mode="human")
    
    # 重置环境
    obs, _ = env.reset()
    print("环境已重置")
    
    # 运行仿真
    total_reward = 0
    
    for step in range(200):  # 运行200步
        # 简单的平衡策略
        x, x_dot, cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        
        # PID控制
        action = -3.0 * theta - 1.0 * theta_dot - 0.2 * x - 0.1 * x_dot
        action = np.clip([action], -1.0, 1.0)
        
        # 执行动作
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # 渲染环境
        env.render()
        
        # 打印状态信息
        if step % 50 == 0:
            print(f"步骤 {step}: 角度={theta:.3f}rad, 位置={x:.3f}m, 奖励={reward:.3f}")
        
        # 如果倒下就重置
        if terminated or truncated:
            print("摆杆倒下，重置环境")
            obs, _ = env.reset()
            total_reward = 0
        
        # 稍微延迟以便观察
        time.sleep(0.02)
    
    print(f"仿真完成，总奖励: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    test_visualization()