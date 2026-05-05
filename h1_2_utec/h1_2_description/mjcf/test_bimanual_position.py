import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_stationary_pos.xml")
data = mujoco.MjData(model)

act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

def set_ctrl(name, value):
    if name in act:
        data.ctrl[act[name]] = value
    else:
        print(f"[WARN] No existe actuador: {name}")

def home_pose(t):
    data.ctrl[:] = 0.0

    set_ctrl("torso_joint", 0.0)

    # Postura conservadora
    set_ctrl("left_shoulder_pitch_joint", 0.0)
    set_ctrl("left_shoulder_roll_joint", 0.20 + 0.06*np.sin(t))
    set_ctrl("left_shoulder_yaw_joint", 0.0)
    set_ctrl("left_elbow_joint", 0.45)
    set_ctrl("left_wrist_roll_joint", 0.0)
    set_ctrl("left_wrist_pitch_joint", 0.0)
    set_ctrl("left_wrist_yaw_joint", 0.0)

    set_ctrl("right_shoulder_pitch_joint", 0.0)
    set_ctrl("right_shoulder_roll_joint", -0.20 - 0.06*np.sin(t))
    set_ctrl("right_shoulder_yaw_joint", 0.0)
    set_ctrl("right_elbow_joint", 0.45)
    set_ctrl("right_wrist_roll_joint", 0.0)
    set_ctrl("right_wrist_pitch_joint", 0.0)
    set_ctrl("right_wrist_yaw_joint", 0.0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        home_pose(t)

        mujoco.mj_step(model, data)
        viewer.sync()
