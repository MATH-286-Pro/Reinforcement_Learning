import torch
import torch.nn as nn
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from . import USER_DATATYPE

############ 基础函数 #############
class Actor(nn.Module):
    def __init__(self, cfg: USER_DATATYPE.ArchiConfig):
        super().__init__()

        seq = [] 
        seq.append([nn.Linear(cfg.actor.obs_dim, cfg.actor.hidden_dim, device=cfg.device),        # 输入层
                    nn.LayerNorm(cfg.actor.hidden_dim, device=cfg.device),                        # 归一化层
                    nn.ReLU()])                                                                   # 激活函数

        for _ in range(cfg.actor.hidden_layers - 1):                                              # 隐藏层
            seq.append([nn.Linear(cfg.actor.hidden_dim, cfg.actor.hidden_dim, device=cfg.device),
                        nn.LayerNorm(cfg.actor.hidden_dim, device=cfg.device),
                        nn.ReLU()])
            
        seq.append([nn.Linear(cfg.actor.hidden_dim, cfg.actor.action_dim, device=cfg.device),     # 输出层
                    nn.Tanh()])                                                                   # 激活函数   # Tanh() 保证输出 action 为 [-1,+1]
        
        self.model = nn.Sequential(*[layer for block in seq for layer in block])                  # 展开列表并传入 Sequential

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class Critic(nn.Module):
    def __init__(self, cfg: USER_DATATYPE.ArchiConfig):
        super().__init__()

        seq = []
        seq.append([nn.Linear(cfg.critic.obs_dim, cfg.critic.hidden_dim, device=cfg.device),      # 输入层
                    nn.LayerNorm(cfg.critic.hidden_dim, device=cfg.device),                       # 归一化层
                    nn.ReLU()])                                                                   # 激活函数

        for _ in range(cfg.critic.hidden_layers - 1):                                             # 隐藏层
            seq.append([nn.Linear(cfg.critic.hidden_dim, cfg.critic.hidden_dim, device=cfg.device), 
                        nn.LayerNorm(cfg.critic.hidden_dim, device=cfg.device),
                        nn.ReLU()])
            
        seq.append([nn.Linear(cfg.critic.hidden_dim, cfg.critic.out_dim, device=cfg.device),      # 输出层
                    # nn.ReLU()                                                                   # 不使用激活函数，因为 Critic 的输出可以是任意实数 (正负皆可
                    ])                                                                  

        self.model = nn.Sequential(*[layer for block in seq for layer in block])
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

############ 高级函数 #############



#########################
# 注册配置结构到 Hydra ConfigStore
# 否则只会加载 .ymal 中有的变量
cs = ConfigStore.instance()
cs.store(name="config", node=USER_DATATYPE.CONFIG)

@hydra.main(config_path='./', config_name='config', version_base="1.1")
def main(cfg: USER_DATATYPE.CONFIG):
    print(cfg)

    # 初始化

    # 仿真环境初始化

    # 神经网络
    actor = Actor(cfg.archi)
    critic = Critic(cfg.archi)

    # 优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.archi.actor.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.archi.critic.lr)

    print(actor)
    print(critic)


if __name__ == "__main__":
    main()