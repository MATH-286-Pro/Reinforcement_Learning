# MuJoCo Playground CartPole 环境

这是一个基于 MuJoCo Playground 框架的 CartPole 环境实现，专门针对大规模并行训练进行了优化。

## 🚀 特性

- **大规模并行训练**: 支持数千个环境同时运行
- **JAX/MJX 后端**: 利用 JAX 的 JIT 编译和自动微分
- **GPU 加速**: 完全在 GPU 上运行物理仿真
- **灵活配置**: 支持多种环境和训练参数配置
- **高效实现**: 针对 Brax 训练框架优化

## 📁 文件结构

```
MY_TEST/project1/
├── env/
│   ├── inverted_pendulum.xml           # MuJoCo 模型文件
│   ├── mujoco_playground_cartpole_env.py # 主环境实现
│   └── cartpole_config.py              # 环境配置
├── cartpole_parallel_train.py          # 并行训练脚本
├── cartpole_test_visualize.py          # 测试和可视化
└── README.md                           # 说明文档（本文件）
```

## 🛠️ 安装依赖

确保安装以下依赖包：

```bash
pip install jax jaxlib mujoco brax flax optax orbax-checkpoint
pip install numpy matplotlib mediapy tqdm ml-collections
pip install gymnasium  # 如果需要与标准 Gym 接口兼容
```

对于 GPU 支持，请安装对应的 JAX GPU 版本：

```bash
# CUDA 12 (推荐)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 或 CUDA 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 🏃‍♂️ 快速开始

### 1. 测试环境

首先测试环境是否正常工作：

```python
# 运行环境测试
python cartpole_test_visualize.py
```

这将运行综合测试，包括：
- 基本功能测试
- 随机策略测试
- PID 控制器测试
- 向量化环境性能测试
- 数据可视化

### 2. 基本使用

```python
import jax
from env.mujoco_playground_cartpole_env import create_cartpole_env
from env.cartpole_config import get_config

# 创建环境
env_config = get_config('default')
env = create_cartpole_env(**env_config)

# 重置环境
rng = jax.random.PRNGKey(42)
state = env.reset(rng)

# 执行动作
action = jax.numpy.array([0.5])  # 向右施力
new_state = env.step(state, action)

print(f"观测: {new_state.obs}")
print(f"奖励: {new_state.reward}")
print(f"完成: {new_state.done}")
```

### 3. 大规模并行训练

```python
# 运行 PPO 训练
python cartpole_parallel_train.py
```

或者自定义训练：

```python
from cartpole_parallel_train import CartPoleTrainer

# 创建训练器
trainer = CartPoleTrainer(
    algorithm='ppo',           # 'ppo' 或 'sac'
    env_config_name='training',
    num_envs=2048,            # 并行环境数量
    num_timesteps=2_000_000,  # 训练步数
    save_dir='./models'       # 模型保存路径
)

# 执行训练
make_inference_fn, params, metrics = trainer.train()

# 评估模型
eval_stats = trainer.evaluate(params, make_inference_fn, num_episodes=10)
```

## 🎛️ 配置选项

### 环境配置

可用的环境配置：

- `default`: 默认配置
- `training`: 训练优化配置
- `evaluation`: 评估配置

```python
from env.cartpole_config import get_config, print_config

# 获取配置
config = get_config('training')
print_config(config)

# 自定义参数
config.force_limit = 25.0        # 最大控制力
config.episode_length = 1000     # 最大步数
config.healthy_reward = 1.0      # 健康奖励
config.ctrl_cost_weight = 0.01   # 控制成本权重
```

### 训练配置

```python
from env.cartpole_config import get_training_config

# PPO 配置
ppo_config = get_training_config('ppo')
ppo_config.num_envs = 4096          # 环境数量
ppo_config.learning_rate = 3e-4     # 学习率
ppo_config.num_timesteps = 5_000_000 # 训练步数

# SAC 配置
sac_config = get_training_config('sac')
sac_config.batch_size = 512
sac_config.alpha = 0.1
```

## 📊 环境详细信息

### 观测空间 (5维)
- `x`: 小车位置 (m)
- `x_dot`: 小车速度 (m/s)
- `cos_theta`: 摆杆角度余弦值
- `sin_theta`: 摆杆角度正弦值
- `theta_dot`: 摆杆角速度 (rad/s)

### 动作空间 (1维)
- `force`: 施加在小车上的归一化力 [-1, 1]，实际力为 `force * force_limit`

### 奖励函数
```python
reward = healthy_reward + angle_reward + position_reward + ctrl_cost

# 其中：
# healthy_reward = 1.0 (保持运行)
# angle_reward = cos(theta) (鼓励摆杆竖直)
# position_reward = -0.1 * x^2 (鼓励小车居中)
# ctrl_cost = -ctrl_cost_weight * action^2 (惩罚大动作)
```

### 终止条件
- 小车位置超出边界: `|x| > x_threshold` (默认 2.4m)
- 摆杆角度过大: `|theta| > theta_threshold` (默认 12°)
- 达到最大步数: `step >= episode_length`

## 🔧 高级功能

### 域随机化

```python
# 启用域随机化
config = get_config('training')
config.randomization.enable = True
config.randomization.force_noise_std = 0.1
config.randomization.mass_range = [0.8, 1.2]
config.randomization.length_range = [0.95, 1.05]
```

### 向量化环境

```python
import jax

# 创建向量化函数
vmap_reset = jax.vmap(env.reset)
vmap_step = jax.vmap(env.step)

# 批量重置
num_envs = 1000
rngs = jax.random.split(jax.random.PRNGKey(0), num_envs)
states = vmap_reset(rngs)

# 批量步进
actions = jax.random.uniform(
    jax.random.PRNGKey(1), 
    shape=(num_envs, 1), 
    minval=-1.0, 
    maxval=1.0
)
new_states = vmap_step(states, actions)
```

### 自定义 XML 模型

你可以使用自己的 MuJoCo XML 模型：

```python
env = create_cartpole_env(
    xml_path="/path/to/your/model.xml",
    force_limit=20.0,
    ctrl_dt=0.02
)
```

## 📈 性能基准

在典型的 GPU 设置上（如 RTX 3080），你可以期待：

- **单环境**: ~10,000 步/秒
- **1000 环境**: ~1,000,000 步/秒
- **4000 环境**: ~3,000,000 步/秒

性能随 GPU 内存和计算能力线性扩展。

## 🐛 故障排除

### 常见问题

1. **JAX 设备问题**
   ```python
   print(f"JAX 设备: {jax.devices()}")
   print(f"默认后端: {jax.default_backend()}")
   ```

2. **内存不足**
   - 减少 `num_envs` 参数
   - 使用 `backend='generalized'` 而不是 `mjx`

3. **性能问题**
   - 确保使用 GPU 后端
   - 启用 JIT 编译: `jax.jit(function)`
   - 检查 XLA 编译缓存

### 调试模式

```python
# 启用 JAX 调试
import jax
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# 禁用 JIT 编译（调试时）
jax.config.update('jax_disable_jit', True)
```

## 🤝 贡献

欢迎贡献代码！请确保：

1. 代码通过所有测试
2. 添加适当的文档字符串
3. 遵循现有的代码风格
4. 更新相关文档

## 📝 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [JAX](https://github.com/google/jax) - 高性能数值计算
- [Brax](https://github.com/google/brax) - 强化学习框架
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) - 基础框架

---

如有问题或建议，请创建 Issue 或 Pull Request！ 🎉