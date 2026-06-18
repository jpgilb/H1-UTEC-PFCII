import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "scene_part_sequencing.xml"


def set_joint_qpos(model, data, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        print(f"[WARN] Joint no encontrado: {joint_name}")
        return

    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = value


def build_ctrl_reference_from_qpos(model, data):
    ctrl_ref = np.zeros(model.nu)

    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]

        if jid < 0:
            continue

        qadr = model.jnt_qposadr[jid]
        ctrl_value = data.qpos[qadr]

        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            ctrl_value = np.clip(ctrl_value, lo, hi)

        ctrl_ref[aid] = ctrl_value

    return ctrl_ref


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Modelo cargado correctamente")
print("Bodies:", model.nbody)
print("Geoms:", model.ngeom)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Cameras:", model.ncam)

if model.nu == 0:
    raise RuntimeError("El modelo no tiene actuadores. Revisa h1_2_stationary_controlled.xml")

mujoco.mj_resetData(model, data)

# Postura inicial conservadora.
# Ajusta estos valores si una articulación queda incómoda.
initial_pose = {
    # Piernas: postura estable visual, aunque la base esté fija.
    "left_hip_pitch_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "left_knee_joint": -0.15,
    "right_knee_joint": -0.15,
    "left_ankle_pitch_joint": 0.15,
    "right_ankle_pitch_joint": 0.15,

    # Torso fijo.
    "torso_joint": 0.0,

    # Brazos en postura de espera, hacia el frente.
    "left_shoulder_pitch_joint": -0.30,
    "right_shoulder_pitch_joint": -0.30,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.55,
    "right_elbow_joint": 0.55,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

for joint_name, value in initial_pose.items():
    set_joint_qpos(model, data, joint_name, value)

mujoco.mj_forward(model, data)

ctrl_ref = build_ctrl_reference_from_qpos(model, data)

print("Referencia de control construida.")
print("Abriendo viewer...")

last_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Mantener postura inicial con actuadores de posición
        data.ctrl[:] = ctrl_ref

        mujoco.mj_step(model, data)

        now = time.time()
        if now - last_time > 2.0:
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            pelvis_pos = data.xpos[pelvis_id]
            print(f"pelvis pos: {pelvis_pos}")
            last_time = now

        viewer.sync()
