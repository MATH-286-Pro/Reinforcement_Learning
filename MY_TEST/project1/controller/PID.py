import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import env.cartpole_env as cartpole_env


def main():
    env = cartpole_env.SimplePendulumEnv(render_mode="human")

    # 模型信息
    pole_length = 0.6

    obs, _ = env.reset()
    terminated = False

    while not terminated:
        x, x_dot, cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        # 速度环 PID 控制
        VELOCITY_KP = 0.6
        VELOCITY_KD = 1.0   # 这里因为没有 acc 的信息所以条这个参数没用
        target_vel = 0
        current_vel = x_dot + pole_length * theta_dot  # 考虑小车速度和摆杆角速度的组合

        error_vel     =  target_vel - current_vel  # 目标速度
        error_vel_dot = 0                          # 目标加速度为0

        vel_loop_output = error_vel * VELOCITY_KP + error_vel_dot * VELOCITY_KD


        # 直立环 PID 控制
        ANG_KP = -10.0
        ANG_KD = -0.1
        target_angle  = vel_loop_output
        current_angle = theta

        error_angle     = target_angle - current_angle
        error_angle_dot = - theta_dot

        action = error_angle * ANG_KP + error_angle_dot * ANG_KD  # PID 控制器

        # 执行动作
        obs, _ , terminated, truncated, _ = env.step([action])

        # 渲染环境
        env.render()

        # 
        time.sleep(0.002)   

    env.close()

if __name__ == "__main__":
    main()