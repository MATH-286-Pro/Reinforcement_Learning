import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import env.cartpole_env as cartpole_env


def main():
    env = cartpole_env.SimplePendulumEnv(render_mode="human")

    obs, _ = env.reset()
    terminated = False

    while not terminated:
        x, x_dot, cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        # 直立环 PID 控制
        KP = 10.0
        KD = 0.1
        action = theta * KP + theta_dot * KD  # PID 控制器

        # 执行动作
        obs, _ , terminated, truncated, _ = env.step([action])

        # 渲染环境
        env.render()

        # 
        time.sleep(0.002)   

    env.close()

if __name__ == "__main__":
    main()