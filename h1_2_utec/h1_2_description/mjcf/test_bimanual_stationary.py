import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_stationary.xml")
data = mujoco.MjData(model)

# Crear mapa: nombre_actuador -> índice
act = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    act[name] = i
    print(i, name)

def set_ctrl(name, value):
    if name not in act:
        print(f"[WARN] Actuador no encontrado: {name}")
        return
    data.ctrl[act[name]] = value

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0

        # Mantener todo inicialmente en 0
        data.ctrl[:] = 0.0

        # Torso fijo
        set_ctrl("torso_joint", 0.0)

        # Brazo izquierdo: postura extendida controlada
        set_ctrl("left_shoulder_pitch_joint", -0.45)
        set_ctrl("left_shoulder_roll_joint", 0.55 + 0.15 * np.sin(t))
        set_ctrl("left_shoulder_yaw_joint", 0.00)
        set_ctrl("left_elbow_joint", 1.20)
        set_ctrl("left_wrist_roll_joint", 0.00)
        set_ctrl("left_wrist_pitch_joint", 0.00)
        set_ctrl("left_wrist_yaw_joint", 0.00)

        # Brazo derecho: postura simétrica
        set_ctrl("right_shoulder_pitch_joint", -0.45)
        set_ctrl("right_shoulder_roll_joint", -0.55 + 0.15 * np.sin(t))
        set_ctrl("right_shoulder_yaw_joint", 0.00)
        set_ctrl("right_elbow_joint", 1.20)
        set_ctrl("right_wrist_roll_joint", 0.00)
        set_ctrl("right_wrist_pitch_joint", 0.00)
        set_ctrl("right_wrist_yaw_joint", 0.00)

        mujoco.mj_step(model, data)
        viewer.sync()
