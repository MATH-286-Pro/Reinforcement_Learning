# **MuJoCo**

## **Conda**
```bash
# windows 上在 conda activate 前先运行以下代码
source ~/miniconda3/etc/profile.d/conda.sh
```

## **MuJoCo**

- [总网站](https://mujoco.org/)
- [GitHub 网站](https://github.com/google-deepmind/mujoco)
- [函数介绍网站](https://mujoco.readthedocs.io/en/stable/overview.html)

MuJoCo 支持 C++ 和 Python

| Colab | 描述 |
|-------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb) | **入门教程** - 教授 MuJoCo 基础知识 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb) | **模型编辑** - 演示如何程序化创建和编辑模型 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb) | **Rollout 模块** - 展示如何使用多线程 rollout 模块 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb) | **LQR 控制器** - 合成线性二次控制器，让人形机器人单腿平衡 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb) | **最小二乘法** - 解释如何使用基于 Python 的非线性最小二乘求解器 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb) | **MJX 教程** - 提供 MuJoCo XLA 的使用示例（用 JAX 编写的 MuJoCo 分支） |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb) | **可微物理** - 使用从 MuJoCo 物理步骤自动导出的解析梯度训练运动策略 |

## **MuJoCo Playground**
- [Github 网站](https://github.com/google-deepmind/mujoco_playground/)

| Colab | 描述 | 类型 |
|-------|------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb) | Playground 与 DM Control Suite 入门介绍 | 基础教程 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb) | 足式机器人 locomotion 环境教程 | 基础教程 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb) | 机械臂 manipulation 环境教程 | 基础教程 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1_t4.ipynb) | 基于视觉的 CartPole 训练 (T4 实例) | 视觉训练 (GPU) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb) | 基于视觉的 CartPole 训练 | 本地运行* |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb) | 基于视觉的机器人操控 | 本地运行* |








## 使用指南
```bash
# 安装相关依赖
./install/install.sh

# 进入 examples 查看 example 1-3 案例
```

```bash
# 运行 MY_TEST/project1/controller/PID.py 效果
```
https://github.com/user-attachments/assets/2a4f6d82-7bc8-468b-b190-20ac2ae98538




## MuJoCo 可视化
```bash
python3 view_go2.py

# 双击选中零件
# ctrl + 左键 添加力矩
# ctrl + 右键 添加力
```

## MuJoCo 和 MuJoCo Playground?
- MuJoCo 不支持 GPU 训练，并行训练需要自己写线程
- MuJoCo Playground 支持 GPU 大规模并行训练



## 更新日志
- 2025.9.21 周日
  - 在 windows 上测试可以使用 view_go2.py 使用 MuJoCo GUI
  - 在 windows 上测试可以 Project 1 可以使用 hydra 
  - 成功部署 串级PID 控制 cartpole
