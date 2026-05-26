import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_part_sequencing_upper_fixed.xml")
data = mujoco.MjData(model)

act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

controlled_joints = list(act.keys())

def set_ctrl(name, value):
    if name in act:
        data.ctrl[act[name]] = value

def joint_position(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jid]
    return float(data.qpos[qadr])

mujoco.mj_forward(model, data)

# Inicializar comando desde la posición real actual
cmd = {}
for name in controlled_joints:
    cmd[name] = joint_position(name)

home_pose = {
    "left_shoulder_pitch_joint": 0.00,
    "left_shoulder_roll_joint": 0.10,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.22,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint": -0.10,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.22,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

approach_pose = {
    "left_shoulder_pitch_joint": -0.03,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_yaw_joint": 0.02,
    "left_elbow_joint": 0.30,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": -0.03,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_yaw_joint": -0.02,
    "right_elbow_joint": 0.30,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def desired_pose(t):
    # 0-5 s: transición muy lenta hacia home
    # 5-35 s: aproximación lenta
    if t < 5.0:
        return home_pose

    alpha = smoothstep((t - 5.0) / 30.0)
    pose = {}
    for name in home_pose:
        pose[name] = home_pose[name] + alpha * (approach_pose[name] - home_pose[name])
    return pose

def update_cmd(target):
    dt = model.opt.timestep

    # Muy lento: 0.025 rad/s
    max_rate = 0.025

    for name, target_value in target.items():
        if name not in cmd:
            continue

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

        target = desired_pose(t)
        update_cmd(target)
        apply_cmd()

        mujoco.mj_step(model, data)
        viewer.sync()
