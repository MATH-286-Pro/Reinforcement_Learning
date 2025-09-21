import gymnasium as gym
import MY_TEST.project1.inverted_pendulum as ip

env = ip.InvertedPendulumMuJoCoEnv(render_mode=None)  # or "human"
# 之后直接丢给你的 RL 训练器即可
