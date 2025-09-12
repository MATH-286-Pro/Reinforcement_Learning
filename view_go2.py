import mujoco, mujoco.viewer

# m = mujoco.MjModel.from_xml_path("go2.xml")
m = mujoco.MjModel.from_xml_path("./models/unitree_robots/go2/scene.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as v:
    # 让相机距离与模型尺度匹配
    v.cam.distance = m.stat.extent * 2.0
    v.cam.lookat[:] = m.stat.center  # 对准模型中心
    # 若模型被隐藏在地面以下，试着把视角抬高一点
    v.cam.elevation = -15
    v.cam.azimuth = 120

    # 保险起见：把所有几何分组“并到0组”，并去掉透明
    try:
        m.geom_group[:] = 0
        rgba = m.geom_rgba
        if rgba is not None and len(rgba):
            rgba[:,3] = 1.0  # 统一不透明
    except Exception:
        pass

    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
