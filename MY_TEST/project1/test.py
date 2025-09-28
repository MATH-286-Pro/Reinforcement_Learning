from __future__ import annotations
import os
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium     def _apply_domain_randomization(self):
        """
        可选：应用随机化（可以在模型构建后改变质量、尺寸等）
        注意：真实项目中请确保动量一致性（mass/pos）等，或在 reset 前后 mj_forward。
        """
        # 获取 body 的 ID 而不是 geom 的 ID
        cart_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
        pole_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")

        cart_mass = self.np_random.uniform(*self.cfg.dr_mass_cart_range)
        tip_mass = self.np_random.uniform(*self.cfg.dr_tip_mass_range)

        # 修改 body mass (MuJoCo 中质量存储在 body 中)
        if cart_bid >= 0:
            self.model.body_mass[cart_bid] = cart_mass
        if pole_bid >= 0:
            self.model.body_mass[pole_bid] = tip_mass

        mujoco.mj_forward(self.model, self.data)import mujoco


@dataclass
class PendulumConfig:
    # 控制与步进 / Control & stepping
    frame_skip: int = 10                # 每步调用 mj_step 的次数（控制保持），effective dt = model.dt * frame_skip
    action_scale: float = 10.0          # 将 [-1,1] 动作缩放为实际牛顿力的比例
    force_limit: float = 20.0           # 再次 clip 力，确保与 actuator ctrlrange 一致或更紧

    # 终止条件 / Termination thresholds
    x_threshold: float = 2.4            # 轨道边界（和 XML 一致或略小）
    theta_threshold_radians: float = 36 * math.pi / 180.0  # 允许 36° 偏差

    # 奖励权重 / Reward shaping
    w_upright: float = 5.0              # 竖直（cos(theta)）奖励权重
    w_center: float = 0.5               # 靠近原点奖励权重
    w_vel: float = 0.01                 # 速度惩罚（x_dot、theta_dot）
    w_action: float = 0.001             # 动作 L2 惩罚
    alive_bonus: float = 0.2            # 存活奖励，鼓励长时间不倒

    # 重置与随机化 / Reset & DR
    init_x_std: float = 0.05
    init_xdot_std: float = 0.05
    init_theta_std: float = 5 * math.pi / 180.0
    init_thetadot_std: float = 0.05

    # 可选 domain randomization（每次 reset 应用）
    dr_mass_cart_range: Tuple[float, float] = (0.4, 0.8)   # 仅示例：用作几何质量缩放
    dr_tip_mass_range: Tuple[float, float] = (0.15, 0.30)

    terminate_on_fall: bool = True      # True: 越界/倒下即 done；False: 只给负奖励，不结束（部分算法更稳）
    use_truncation: bool = False        # True: 用 truncated 标志代替 terminated（更接近 gym-MuJoCo 的做法）


class InvertedPendulumMuJoCoEnv(gym.Env):
    """
    Gymnasium-compatible MuJoCo inverted pendulum environment.

    观测 (obs): np.array([x, x_dot, cos(theta), sin(theta), theta_dot])
    动作 (act): 标量 in [-1, 1]，内部缩放为牛顿力并 clip 在 [-force_limit, +force_limit]
    """
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 1.0 / 0.002  # 不太重要，仅供参考
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 cfg: Optional[PendulumConfig] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.cfg = cfg or PendulumConfig()

        # 构建模型与数据 / Build MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path("./MY_TEST/project1/inverted_pendulum.xml")
        self.data = mujoco.MjData(self.model)

        # 渲染器（可选）/ Renderer (optional)
        self._viewer = None
        if self.render_mode == "human":
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print("[WARN] Failed to launch viewer:", e)
                self._viewer = None

        # 空间定义 / Spaces
        high = np.array([np.inf] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 随机种子 / Seeding
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # 便捷索引 / Indices
        self._j_slide = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide")
        self._j_hinge = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self._act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart_force")

        # 预计算 dt
        self.dt = self.model.opt.timestep * self.cfg.frame_skip

    # ---------- Gym API ----------
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Domain Randomization（可选）
        self._apply_domain_randomization()

        # 初始状态随机化
        x0 = self.np_random.normal(0.0, self.cfg.init_x_std)
        xdot0 = self.np_random.normal(0.0, self.cfg.init_xdot_std)
        th0 = self.np_random.normal(0.0, self.cfg.init_theta_std)  # 0 = 竖直向上
        thdot0 = self.np_random.normal(0.0, self.cfg.init_thetadot_std)

        # 设置关节坐标与速度
        self.data.qpos[self._j_slide] = x0
        self.data.qvel[self._j_slide] = xdot0
        self.data.qpos[self._j_hinge] = th0
        self.data.qvel[self._j_hinge] = thdot0

        # 将控制置零
        self.data.ctrl[self._act_id] = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = {"dt": self.dt}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1) 动作缩放并 clip 到 actuator 范围
        a = float(np.clip(action[0], -1.0, 1.0)) * self.cfg.action_scale
        a = float(np.clip(a, -self.cfg.force_limit, self.cfg.force_limit))
        self.data.ctrl[self._act_id] = a

        # 2) 多子步积分（frame_skip）
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 3) 计算观测、奖励、终止
        obs = self._get_obs()
        reward = self._calc_reward(obs, a)
        terminated, truncated = self._check_done(obs)

        # 4) 渲染（可选）
        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # 用内置渲染器获取像素（简单做法：创建一个隐式的渲染上下文）
            width, height = 640, 480
            with mujoco.Renderer(self.model, width, height) as renderer:
                renderer.update_scene(self.data, camera=None)
                rgb = renderer.render()
            return rgb
        # human 模式已在 step 内 sync
        return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ---------- Helpers ----------
    def _get_obs(self) -> np.ndarray:
        x = float(self.data.qpos[self._j_slide])
        x_dot = float(self.data.qvel[self._j_slide])
        th = float(self.data.qpos[self._j_hinge])         # hinge angle (0 = upright)
        th_dot = float(self.data.qvel[self._j_hinge])

        obs = np.array([x, x_dot, math.cos(th), math.sin(th), th_dot], dtype=np.float32)
        return obs

    def _calc_reward(self, obs: np.ndarray, action_force: float) -> float:
        """
        奖励设计 / Reward shaping:
          + w_upright * cos(theta)           (越接近1越好)
          - w_center  * x^2
          - w_vel     * (x_dot^2 + theta_dot^2)
          - w_action  * (u^2)
          + alive_bonus
        """
        x, x_dot, cth, sth, th_dot = obs
        theta = math.atan2(sth, cth)

        r = 0.0
        r += self.cfg.w_upright * cth
        r -= self.cfg.w_center * (x ** 2)
        r -= self.cfg.w_vel * (x_dot ** 2 + th_dot ** 2)
        r -= self.cfg.w_action * (action_force ** 2)
        r += self.cfg.alive_bonus
        return float(r)

    def _check_done(self, obs: np.ndarray) -> Tuple[bool, bool]:
        x = float(obs[0])
        theta = math.atan2(float(obs[3]), float(obs[2]))

        out_of_bound = (abs(x) > self.cfg.x_threshold) or (abs(theta) > self.cfg.theta_threshold_radians)

        if self.cfg.terminate_on_fall:
            # 传统 CartPole 风格：越界即 terminated
            terminated = bool(out_of_bound)
            truncated = False
        else:
            # 不结束，但可选设置成 truncated（更接近 MuJoCo 一些基准的“时间截断”风格）
            terminated = False
            truncated = bool(out_of_bound) if self.cfg.use_truncation else False

        return terminated, truncated

    def _apply_domain_randomization(self):
        """
        简单的 DR：随机化 cart 和 tip 的质量（通过修改 geom mass）。
        注意：真实项目中请确保动量一致性（mass/pos）等，或在 reset 前后 mj_forward。
        """
        # cart geom
        cart_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cart_geom")
        tip_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tip")

        cart_mass = self.np_random.uniform(*self.cfg.dr_mass_cart_range)
        tip_mass = self.np_random.uniform(*self.cfg.dr_tip_mass_range)

        # 修改 geom mass
        self.model.geom_mass[cart_gid] = cart_mass
        self.model.geom_mass[tip_gid] = tip_mass

        mujoco.mj_forward(self.model, self.data)


# ---------------------------
# 2) Gym registration helper
# ---------------------------
def register_env_with_gym():
    """
    可选：注册到 Gymnasium，使其可以通过 make("InvertedPendulumMuJoCo-v0") 创建。
    """
    gym.register(
        id="InvertedPendulumMuJoCo-v0",
        entry_point=f"{__name__}:InvertedPendulumMuJoCoEnv",
        kwargs={"render_mode": None},
        max_episode_steps=1000,  # 你可根据需要调整
    )


# ---------------------------
# 3) Quick self-test
# ---------------------------
def _quick_test():
    print("Creating env ...")
    env = InvertedPendulumMuJoCoEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    print("Obs dim:", env.observation_space.shape, "| Act dim:", env.action_space.shape, "| dt =", info["dt"])

    total_r = 0.0
    for t in range(500):
        # 随机动作（仅自测）
        act = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(act)
        total_r += r
        if terminated or truncated:
            print(f"Episode finished at step {t}, return={total_r:.2f}, terminated={terminated}, truncated={truncated}")
            obs, _ = env.reset()
            total_r = 0.0

    env.close()
    print("Self-test done.")


if __name__ == "__main__":
    # 可选：注册到 Gymnasium 全局
    register_env_with_gym()
    # 运行一次自检
    _quick_test()