pip install mujoco
pip install mujoco_mjx     
pip install brax           # 安装 Google 的 JAX 版可微物理引擎
pip install playground     # 安装 mujoco playground

pip uninstall jax jaxlib   # 卸载 CPU 版本 Jax
pip install jax[cuda12]    # 安装 Cuda 版本 Jax

pip install mediapy        # 安装 mediapy
pip install wandb
pip install hydra-core --upgrade
pip install matplotlib