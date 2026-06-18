import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_part_sequencing_handless_smooth.xml")
data = mujoco.MjData(model)

act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

controlled_joints = [
    "torso_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

def joint_position(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jid]
    return data.qpos[qadr]

# Inicializar referencias desde la postura real actual del modelo.
# Esto evita el primer salto brusco.
mujoco.mj_forward(model, data)
cmd = {}
for name in controlled_joints:
    if name in act:
        cmd[name] = float(joint_position(name))

# Pose estable inicial, todavía cerca del cuerpo.
home_pose = {
    "torso_joint": 0.0,

    "left_shoulder_pitch_joint": 0.00,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.25,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.25,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

# Pose de pre-aproximación: leve avance, sin bajar hacia la mesa.
# No intenta alcanzar ni tocar piezas.
approach_pose = {
    "torso_joint": 0.0,

    "left_shoulder_pitch_joint": -0.05,
    "left_shoulder_roll_joint": 0.20,
    "left_shoulder_yaw_joint": 0.04,
    "left_elbow_joint": 0.38,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": -0.05,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": -0.04,
    "right_elbow_joint": 0.38,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def desired_pose(t):
    # 0-4 s: ir lentamente hacia home
    # 4-24 s: aproximación lenta hacia objetos
    # >24 s: mantener aproximación
    if t < 4.0:
        return home_pose

    alpha = smoothstep((t - 4.0) / 20.0)
    pose = {}
    for name in home_pose:
        pose[name] = home_pose[name] + alpha * (approach_pose[name] - home_pose[name])
    return pose

def update_rate_limited_command(target):
    dt = model.opt.timestep

    # Velocidad máxima de referencia articular.
    # Muy baja para evitar movimientos bruscos.
    max_rate_default = 0.08  # rad/s

    for name, target_value in target.items():
        if name not in cmd:
            continue

        max_rate = max_rate_default

        # Muñecas todavía más lentas
        if "wrist" in name:
            max_rate = 0.04

        # Torso casi fijo
        if "torso" in name:
            max_rate = 0.03

        delta = target_value - cmd[name]
        delta = np.clip(delta, -max_rate * dt, max_rate * dt)
        cmd[name] += delta

def apply_command():
    data.ctrl[:] = 0.0
    for name, value in cmd.items():
        if name in act:
            data.ctrl[act[name]] = value

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        target = desired_pose(t)
        update_rate_limited_command(target)
        apply_command()

        mujoco.mj_step(model, data)
        viewer.sync()
