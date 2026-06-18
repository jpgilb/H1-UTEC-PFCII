import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_part_sequencing.xml")
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

def home_pose():
    data.ctrl[:] = 0.0

    set_ctrl("torso_joint", 0.0)

    # Brazo izquierdo: más alto y menos flexionado
    set_ctrl("left_shoulder_pitch_joint", -0.25)
    set_ctrl("left_shoulder_roll_joint", 0.35)
    set_ctrl("left_shoulder_yaw_joint", 0.15)
    set_ctrl("left_elbow_joint", 0.80)
    set_ctrl("left_wrist_roll_joint", 0.00)
    set_ctrl("left_wrist_pitch_joint", -0.15)
    set_ctrl("left_wrist_yaw_joint", 0.00)

    # Brazo derecho: simétrico
    set_ctrl("right_shoulder_pitch_joint", -0.25)
    set_ctrl("right_shoulder_roll_joint", -0.35)
    set_ctrl("right_shoulder_yaw_joint", -0.15)
    set_ctrl("right_elbow_joint", 0.80)
    set_ctrl("right_wrist_roll_joint", 0.00)
    set_ctrl("right_wrist_pitch_joint", -0.15)
    set_ctrl("right_wrist_yaw_joint", 0.00)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        home_pose()
        mujoco.mj_step(model, data)
        viewer.sync()
