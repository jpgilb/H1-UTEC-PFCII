import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_part_sequencing.xml")
data = mujoco.MjData(model)

# Mapeo de actuadores por nombre
act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

def set_ctrl(name, value):
    if name in act:
        data.ctrl[act[name]] = value

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def interpolate_pose(q0, q1, alpha):
    s = smoothstep(alpha)
    return {k: q0[k] + s * (q1[k] - q0[k]) for k in q0.keys()}

# Pose inicial: brazos cerca del cuerpo, estable
home_pose = {
    "torso_joint": 0.0,

    "left_shoulder_pitch_joint": 0.00,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.35,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.35,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

# Pose de aproximación: brazos más adelante, pero sin tocar mesa ni piezas
approach_pose = {
    "torso_joint": 0.0,

    # Izquierdo: se abre y avanza suavemente
    "left_shoulder_pitch_joint": -0.12,
    "left_shoulder_roll_joint": 0.28,
    "left_shoulder_yaw_joint": 0.10,
    "left_elbow_joint": 0.55,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": -0.05,
    "left_wrist_yaw_joint": 0.00,

    # Derecho: simétrico
    "right_shoulder_pitch_joint": -0.12,
    "right_shoulder_roll_joint": -0.28,
    "right_shoulder_yaw_joint": -0.10,
    "right_elbow_joint": 0.55,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": -0.05,
    "right_wrist_yaw_joint": 0.00,
}

def apply_pose(pose):
    data.ctrl[:] = 0.0
    for joint, value in pose.items():
        set_ctrl(joint, value)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        # 0-2 s: mantener home
        # 2-10 s: interpolar suavemente hacia approach
        # >10 s: mantener approach
        if t < 2.0:
            pose = home_pose
        elif t < 10.0:
            alpha = (t - 2.0) / 8.0
            pose = interpolate_pose(home_pose, approach_pose, alpha)
        else:
            pose = approach_pose

        apply_pose(pose)

        mujoco.mj_step(model, data)
        viewer.sync()
