import mujoco
import mujoco.viewer

print("1. Cargando modelo...")
model = mujoco.MjModel.from_xml_path("scene_part_sequencing.xml")
data = mujoco.MjData(model)
print("2. Modelo cargado")
print("3. Abriendo viewer...")

mujoco.viewer.launch(model, data)

print("4. Viewer cerrado")
