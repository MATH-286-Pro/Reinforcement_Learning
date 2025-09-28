"""
MuJoCo Playground 版本的 CartPole 环境
针对大规模并行训练优化，使用 JAX/MJX 后端
"""
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from flax import struct
from brax.envs.base import PipelineEnv, State
from brax.base import Base, Motion, System
from brax.mjx.base import State as MjxState


@struct.dataclass
class CartPoleState:
    """CartPole 环境状态"""
    pipeline_state: MjxState
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class CartPoleEnv(PipelineEnv):
    """
    MuJoCo Playground CartPole 环境
    
    观测空间:
        - x: 小车位置 (m)
        - x_dot: 小车速度 (m/s)  
        - cos_theta: 摆杆角度余弦值
        - sin_theta: 摆杆角度正弦值
        - theta_dot: 摆杆角速度 (rad/s)
    
    动作空间:
        - force: 施加在小车上的力 [-force_limit, force_limit] (N)
    """
    
    def __init__(
        self,
        xml_path: Optional[str] = None,
        force_limit: float = 20.0,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.002,
        episode_length: int = 1000,
        x_threshold: float = 2.4,
        theta_threshold: float = 0.2094,  # 12 degrees in radians
        healthy_reward: float = 1.0,
        ctrl_cost_weight: float = 0.001,
        forward_reward_weight: float = 0.0,
        backend: str = 'mjx',
        **kwargs
    ):
        # 设置默认 XML 路径
        if xml_path is None:
            xml_path = r"c:\Users\Jiajun Hu\Desktop\Code\Reinforcement_Learning\MY_TEST\project1\env\inverted_pendulum.xml"
        
        # 加载 MuJoCo 模型
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 环境参数
        self._force_limit = force_limit
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._x_threshold = x_threshold
        self._theta_threshold = theta_threshold
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._forward_reward_weight = forward_reward_weight
        
        # 获取关节和执行器索引
        self._slide_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "slide")
        self._hinge_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self._actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_force")
        
        # 创建系统
        sys = mjx.put_model(mj_model)
        
        n_frames = int(self._ctrl_dt / self._sim_dt)
        super().__init__(sys, backend=backend, n_frames=n_frames)
    
    def reset(self, rng: jp.ndarray) -> State:
        """重置环境"""
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        
        # 获取默认的初始位置和速度
        qpos = jp.zeros(self.sys.nq)
        qvel = jp.zeros(self.sys.nv)
        
        # 随机初始化位置和速度
        qpos = qpos.at[self._slide_joint_id].set(
            jax.random.uniform(rng1, (), minval=-0.05, maxval=0.05)
        )
        qpos = qpos.at[self._hinge_joint_id].set(
            jax.random.uniform(rng2, (), minval=-0.05, maxval=0.05)
        )
        
        qvel = qvel.at[self._slide_joint_id].set(
            jax.random.uniform(rng3, (), minval=-0.05, maxval=0.05)
        )
        qvel = qvel.at[self._hinge_joint_id].set(
            jax.random.uniform(rng4, (), minval=-0.05, maxval=0.05)
        )
        
        # 创建初始状态
        pipeline_state = self.pipeline_init(qpos, qvel)
        
        obs = self._get_obs(pipeline_state)
        reward, done = jp.array(0.0), jp.array(False)
        metrics = {
            'x_position': pipeline_state.q[self._slide_joint_id],
            'theta': pipeline_state.q[self._hinge_joint_id],
            'x_velocity': pipeline_state.qd[self._slide_joint_id],
            'theta_velocity': pipeline_state.qd[self._hinge_joint_id],
        }
        
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jp.ndarray) -> State:
        """执行一步"""
        # 限制动作范围
        force = jp.clip(action, -1.0, 1.0) * self._force_limit
        
        # 执行动作
        pipeline_state = self.pipeline_step(state.pipeline_state, force)
        
        # 获取观测
        obs = self._get_obs(pipeline_state)
        
        # 计算奖励和终止条件
        reward = self._get_reward(pipeline_state, action)
        done = self._get_done(pipeline_state, state.info.get('step', 0))
        
        # 更新指标
        metrics = {
            'x_position': pipeline_state.q[self._slide_joint_id],
            'theta': pipeline_state.q[self._hinge_joint_id],
            'x_velocity': pipeline_state.qd[self._slide_joint_id],
            'theta_velocity': pipeline_state.qd[self._hinge_joint_id],
            'reward_healthy': self._healthy_reward * (1 - done),
            'reward_ctrl': -self._ctrl_cost_weight * jp.sum(jp.square(action)),
            'reward_total': reward,
        }
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics
        )
    
    def _get_obs(self, pipeline_state: MjxState) -> jp.ndarray:
        """获取观测"""
        x = pipeline_state.q[self._slide_joint_id]
        x_dot = pipeline_state.qd[self._slide_joint_id]
        theta = pipeline_state.q[self._hinge_joint_id] 
        theta_dot = pipeline_state.qd[self._hinge_joint_id]
        
        return jp.array([
            x,
            x_dot,
            jp.cos(theta),
            jp.sin(theta),
            theta_dot
        ])
    
    def _get_reward(self, pipeline_state: MjxState, action: jp.ndarray) -> jp.ndarray:
        """计算奖励"""
        # 基础健康奖励（保持摆杆竖直）
        theta = pipeline_state.q[self._hinge_joint_id]
        healthy_reward = self._healthy_reward
        
        # 位置奖励（鼓励小车保持在中心）
        x = pipeline_state.q[self._slide_joint_id]
        position_reward = -0.1 * jp.square(x)
        
        # 角度奖励（鼓励摆杆保持竖直）
        angle_reward = jp.cos(theta)
        
        # 控制成本
        ctrl_cost = -self._ctrl_cost_weight * jp.sum(jp.square(action))
        
        # 前向奖励（如果需要）
        forward_reward = self._forward_reward_weight * pipeline_state.qd[self._slide_joint_id]
        
        total_reward = healthy_reward + position_reward + angle_reward + ctrl_cost + forward_reward
        
        return total_reward
    
    def _get_done(self, pipeline_state: MjxState, step: int) -> jp.ndarray:
        """检查终止条件"""
        # 小车位置超出边界
        x = pipeline_state.q[self._slide_joint_id]
        x_done = jp.abs(x) > self._x_threshold
        
        # 摆杆角度过大
        theta = pipeline_state.q[self._hinge_joint_id]
        theta_done = jp.abs(theta) > self._theta_threshold
        
        # 时间步数达到上限
        step_done = step >= self._episode_length
        
        return x_done | theta_done | step_done
    
    @property
    def action_size(self) -> int:
        """动作空间维度"""
        return 1
    
    @property
    def observation_size(self) -> int:
        """观测空间维度"""
        return 5


def create_cartpole_env(
    xml_path: Optional[str] = None,
    force_limit: float = 20.0,
    ctrl_dt: float = 0.02,
    sim_dt: float = 0.002,
    episode_length: int = 1000,
    backend: str = 'mjx',
    **kwargs
) -> CartPoleEnv:
    """创建 CartPole 环境的便捷函数"""
    return CartPoleEnv(
        xml_path=xml_path,
        force_limit=force_limit,
        ctrl_dt=ctrl_dt,
        sim_dt=sim_dt,
        episode_length=episode_length,
        backend=backend,
        **kwargs
    )


# 注册环境（如果使用 registry）
def register_cartpole_env():
    """注册 CartPole 环境到 registry"""
    try:
        from mujoco_playground import registry
        registry.register(
            'CartPolePlayground',
            create_cartpole_env,
            default_config={
                'force_limit': 20.0,
                'ctrl_dt': 0.02,
                'sim_dt': 0.002,
                'episode_length': 1000,
                'x_threshold': 2.4,
                'theta_threshold': 0.2094,
                'healthy_reward': 1.0,
                'ctrl_cost_weight': 0.001,
                'forward_reward_weight': 0.0,
            }
        )
        print("CartPole environment registered successfully!")
    except ImportError:
        print("MuJoCo Playground not available for registration")


if __name__ == "__main__":
    # 测试环境创建
    print("创建 CartPole 环境...")
    env = create_cartpole_env()
    
    print(f"观测空间维度: {env.observation_size}")
    print(f"动作空间维度: {env.action_size}")
    
    # 测试环境重置和步进
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    print(f"初始观测: {state.obs}")
    
    # 执行随机动作
    action = jax.random.uniform(jax.random.PRNGKey(1), shape=(1,), minval=-1.0, maxval=1.0)
    state = env.step(state, action)
    print(f"步进后观测: {state.obs}")
    print(f"奖励: {state.reward}")
    print(f"完成: {state.done}")
    
    print("环境测试完成！")
