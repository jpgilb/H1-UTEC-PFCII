import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_part_sequencing_upper_fixed_v2.xml")
data = mujoco.MjData(model)

act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

def set_ctrl(name, value):
    if name in act:
        data.ctrl[act[name]] = value

def set_qpos(name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = value

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

# Home: brazos arriba de la mesa, sin tocarla
home_pose = {
    "left_shoulder_pitch_joint": 0.00,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.18,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.18,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

# Aproximación: más hombro, menos dependencia del codo.
# No intenta agarrar ni tocar piezas.
approach_pose = {
    "left_shoulder_pitch_joint": -0.22,
    "left_shoulder_roll_joint": 0.30,
    "left_shoulder_yaw_joint": 0.10,
    "left_elbow_joint": 0.32,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.05,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": -0.22,
    "right_shoulder_roll_joint": -0.30,
    "right_shoulder_yaw_joint": -0.10,
    "right_elbow_joint": 0.32,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.05,
    "right_wrist_yaw_joint": 0.00,
}

# Inicializar qpos exactamente en home_pose para evitar caída/rebote inicial
for name, value in home_pose.items():
    set_qpos(name, value)
    set_ctrl(name, value)

mujoco.mj_forward(model, data)

cmd = dict(home_pose)

def target_pose(t):
    # 0-3 s: mantener home
    # 3-33 s: aproximar muy lento
    if t < 3.0:
        return home_pose

    alpha = smoothstep((t - 3.0) / 30.0)
    pose = {}
    for name in home_pose:
        pose[name] = home_pose[name] + alpha * (approach_pose[name] - home_pose[name])
    return pose

def update_cmd(target):
    dt = model.opt.timestep

    # Lento pero visible
    max_rate = 0.035  # rad/s

    for name, target_value in target.items():
        delta = target_value - cmd[name]
        delta = np.clip(delta, -max_rate * dt, max_rate * dt)
        cmd[name] += delta

def apply_cmd():
    for name, value in cmd.items():
        set_ctrl(name, value)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        desired = target_pose(t)
        update_cmd(desired)
        apply_cmd()

        mujoco.mj_step(model, data)
        viewer.sync()
