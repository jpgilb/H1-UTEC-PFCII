from pathlib import Path
import xml.etree.ElementTree as ET

SRC = Path("h1_2.xml")
DST = Path("h1_2_stationary_controlled.xml")

tree = ET.parse(SRC)
root = tree.getroot()

# 1. Eliminar base flotante
removed = False
for parent in root.iter():
    for child in list(parent):
        if child.tag == "joint" and child.get("name") == "floating_base_joint":
            parent.remove(child)
            removed = True

print(f"floating_base_joint eliminado: {removed}")

# 2. Eliminar actuadores existentes para crear un set limpio de posición
for child in list(root):
    if child.tag == "actuator":
        root.remove(child)

# 3. Recolectar joints escalares
joint_names = []

for joint in root.iter("joint"):
    name = joint.get("name")
    joint_type = joint.get("type", "hinge")

    if not name:
        continue

    if joint_type == "free":
        continue

    joint_names.append(name)

    # Añadir amortiguamiento para reducir oscilaciones numéricas
    if "damping" not in joint.attrib:
        if any(k in name for k in ["shoulder", "elbow"]):
            joint.set("damping", "3.0")
        elif "wrist" in name:
            joint.set("damping", "1.5")
        elif any(k in name for k in ["hip", "knee", "ankle", "torso"]):
            joint.set("damping", "5.0")
        else:
            joint.set("damping", "0.5")

    if "armature" not in joint.attrib:
        joint.set("armature", "0.01")

# 4. Crear actuadores de posición
actuator = ET.SubElement(root, "actuator")

for name in joint_names:
    # Usar rango del joint si existe
    joint_elem = None
    for j in root.iter("joint"):
        if j.get("name") == name:
            joint_elem = j
            break

    ctrlrange = "-3.14 3.14"
    if joint_elem is not None and joint_elem.get("range"):
        ctrlrange = joint_elem.get("range")

    # Ganancias y límites por grupo
    if any(k in name for k in ["hip", "knee", "ankle"]):
        kp = "120"
        forcerange = "-250 250"
    elif "torso" in name:
        kp = "100"
        forcerange = "-150 150"
    elif any(k in name for k in ["shoulder", "elbow"]):
        kp = "70"
        forcerange = "-75 75"
    elif "wrist" in name:
        kp = "35"
        forcerange = "-35 35"
    else:
        # dedos / joints pequeños de mano
        kp = "12"
        forcerange = "-12 12"

    ET.SubElement(
        actuator,
        "position",
        {
            "name": f"{name}_pos",
            "joint": name,
            "kp": kp,
            "ctrlrange": ctrlrange,
            "forcelimited": "true",
            "forcerange": forcerange,
        },
    )

ET.indent(tree, space="  ")
tree.write(DST, encoding="utf-8", xml_declaration=False)

print(f"Modelo generado: {DST}")
print(f"Joints actuados: {len(joint_names)}")
