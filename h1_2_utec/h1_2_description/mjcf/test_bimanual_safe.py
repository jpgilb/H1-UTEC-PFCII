import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_stationary.xml")
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
        print(f"[WARN] Actuador no encontrado: {name}")

def hold_neutral():
    data.ctrl[:] = 0.0

    # Torso
    set_ctrl("torso_joint", 0.0)

    # Piernas en postura base. Aunque la base esté fija, esto evita movimientos raros.
    for name in [
        "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]:
        set_ctrl(name, 0.0)

    # Brazos cerca de una postura neutra, sin abrirlos tanto
    set_ctrl("left_shoulder_pitch_joint", 0.0)
    set_ctrl("left_shoulder_roll_joint", 0.15)
    set_ctrl("left_shoulder_yaw_joint", 0.0)
    set_ctrl("left_elbow_joint", 0.30)
    set_ctrl("left_wrist_roll_joint", 0.0)
    set_ctrl("left_wrist_pitch_joint", 0.0)
    set_ctrl("left_wrist_yaw_joint", 0.0)

    set_ctrl("right_shoulder_pitch_joint", 0.0)
    set_ctrl("right_shoulder_roll_joint", -0.15)
    set_ctrl("right_shoulder_yaw_joint", 0.0)
    set_ctrl("right_elbow_joint", 0.30)
    set_ctrl("right_wrist_roll_joint", 0.0)
    set_ctrl("right_wrist_pitch_joint", 0.0)
    set_ctrl("right_wrist_yaw_joint", 0.0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        hold_neutral()

        # Movimiento MUY pequeño, solo para confirmar respuesta sin autocolisión
        amp = 0.08
        set_ctrl("left_shoulder_roll_joint", 0.15 + amp * np.sin(t))
        set_ctrl("right_shoulder_roll_joint", -0.15 - amp * np.sin(t))

        mujoco.mj_step(model, data)
        viewer.sync()
