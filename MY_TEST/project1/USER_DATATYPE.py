import torch
import dataclasses

##### 配置数据类型定义 #####
@dataclasses.dataclass
class WandbConfig:
    entity:  str = "math-286-pro-epfl"
    project: str = "learning"

@dataclasses.dataclass
class ActorConfig:
    obs_dim       : int = 10
    action_dim    : int = 2
    hidden_layers : int = 2
    hidden_dim    : int = 256
    lr            : float = 1e-4

@dataclasses.dataclass
class CriticConfig:
    obs_dim       : int = 10
    out_dim       : int = 2
    hidden_layers : int = 2
    hidden_dim    : int = 256
    lr            : float = 1e-4

@dataclasses.dataclass
class ArchiConfig:
    actor: ActorConfig = dataclasses.field(default_factory=ActorConfig)     # 不使用 ActorConfig() 的原因是为了避免共享变量 BUG 比如 List, Dict (详见 Code_Learning/notebook.ipynb)
    critic: CriticConfig = dataclasses.field(default_factory=CriticConfig) 
    device: str = "cpu"                                                     # "cuda" or "cpu"

@dataclasses.dataclass
class CONFIG:
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)  # wandb 账户设置
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)  # 神经网络参数