import torch
import torch.nn as nn
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore

import RL.USER_DATATYPE as USER_DATATYPE
import RL.network as network
import env.cartpole_env as cartpole_env

#########################
# 注册配置结构到 Hydra ConfigStore
# 否则只会加载 .ymal 中有的变量
cs = ConfigStore.instance()
cs.store(name="config", node=USER_DATATYPE.CONFIG)

def reward_fn(obs):
    return 1.0

@hydra.main(config_path='./', config_name='config', version_base="1.1")
def main(cfg: USER_DATATYPE.CONFIG):
    print(cfg)

    # 初始化

    # 神经网络
    actor = network.Actor(cfg.archi)
    critic = network.Critic(cfg.archi)

    # 优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.archi.actor.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.archi.critic.lr)

    print(actor)
    print(critic)

    # 仿真环境初始化
    env = cartpole_env.SimplePendulumEnv(render_mode="human")
    obs, _ = env.reset()



if __name__ == "__main__":
    main()