#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_torso_hold.py
Patches only torso_joint and its position actuator in h1_2_mujoco_model.xml
"""

import argparse
import datetime
import os
import re
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Ajusta kp, damping y armature para torso_joint en MuJoCo."
    )
    parser.add_argument("--kp", type=float, help="Nuevo valor de kp para el actuador de posición")
    parser.add_argument("--damping", type=float, help="Nuevo valor de damping para torso_joint")
    parser.add_argument("--armature", type=float, help="Nuevo valor de armature para torso_joint")

    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The xml is located at: ../description/h1_2_mujoco_model.xml
    xml_path = os.path.abspath(
        os.path.join(script_dir, "..", "description", "h1_2_mujoco_model.xml")
    )

    if not os.path.exists(xml_path):
        print(f"Error: No se encontró el archivo XML en {xml_path}", file=sys.stderr)
        sys.exit(1)

    # Read original content
    with open(xml_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Apply patches in memory
    modified_content = original_content
    updated_joint_tag = None
    updated_actuator_tag = None

    try:
        # Patch joint (damping, armature)
        if args.damping is not None or args.armature is not None:
            pattern_joint = r'(<joint\b[^>]*\bname="torso_joint"[^>]*>)'
            match_joint = re.search(pattern_joint, modified_content)
            if not match_joint:
                raise ValueError("No se encontró la etiqueta <joint> con name=\"torso_joint\"")

            tag_joint = match_joint.group(1)

            if args.damping is not None:
                if 'damping="' in tag_joint:
                    tag_joint = re.sub(r'damping="[^"]*"', f'damping="{args.damping}"', tag_joint)
                else:
                    if tag_joint.endswith('/>'):
                        tag_joint = tag_joint[:-2] + f' damping="{args.damping}"/>'
                    else:
                        tag_joint = tag_joint[:-1] + f' damping="{args.damping}">'

            if args.armature is not None:
                if 'armature="' in tag_joint:
                    tag_joint = re.sub(r'armature="[^"]*"', f'armature="{args.armature}"', tag_joint)
                else:
                    if tag_joint.endswith('/>'):
                        tag_joint = tag_joint[:-2] + f' armature="{args.armature}"/>'
                    else:
                        tag_joint = tag_joint[:-1] + f' armature="{args.armature}">'

            modified_content = modified_content.replace(match_joint.group(1), tag_joint)
            updated_joint_tag = tag_joint
        else:
            match_joint = re.search(r'(<joint\b[^>]*\bname="torso_joint"[^>]*>)', modified_content)
            if match_joint:
                updated_joint_tag = match_joint.group(1)

        # Patch actuator position (kp)
        if args.kp is not None:
            pattern_act = r'(<position\b[^>]*\bjoint="torso_joint"[^>]*>)'
            match_act = re.search(pattern_act, modified_content)
            if not match_act:
                raise ValueError("No se encontró la etiqueta <position> con joint=\"torso_joint\"")

            tag_act = match_act.group(1)
            if 'kp="' in tag_act:
                tag_act = re.sub(r'kp="[^"]*"', f'kp="{args.kp}"', tag_act)
            else:
                if tag_act.endswith('/>'):
                    tag_act = tag_act[:-2] + f' kp="{args.kp}"/>'
                else:
                    tag_act = tag_act[:-1] + f' kp="{args.kp}">'

            modified_content = modified_content.replace(match_act.group(1), tag_act)
            updated_actuator_tag = tag_act
        else:
            match_act = re.search(r'(<position\b[^>]*\bjoint="torso_joint"[^>]*>)', modified_content)
            if match_act:
                updated_actuator_tag = match_act.group(1)

    except Exception as e:
        print(f"Error al procesar el parche: {e}", file=sys.stderr)
        sys.exit(1)

    # If no changes were requested, just print and exit
    if args.kp is None and args.damping is None and args.armature is None:
        print("No se especificaron cambios (--kp, --damping o --armature).")
        print(f"Joint actual: {updated_joint_tag}")
        print(f"Actuador actual: {updated_actuator_tag}")
        return

    # Create timestamped backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{xml_path}.{timestamp}.bak"
    try:
        shutil.copy2(xml_path, backup_path)
        print(f"Backup creado en: {backup_path}")
    except Exception as e:
        print(f"Error al crear el backup: {e}", file=sys.stderr)
        sys.exit(1)

    # Write modified content
    try:
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
    except Exception as e:
        print(f"Error al escribir el archivo XML: {e}", file=sys.stderr)
        # Restore backup
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, xml_path)
        sys.exit(1)

    # Validate with MuJoCo
    try:
        import mujoco
        mujoco.MjModel.from_xml_path(xml_path)
        print("Validación exitosa con mujoco.MjModel.from_xml_path().")
    except Exception as e:
        print(f"Error de validación con MuJoCo: {e}", file=sys.stderr)
        print("Restaurando backup original...", file=sys.stderr)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, xml_path)
        sys.exit(1)

    print("\nModificaciones realizadas con éxito:")
    print(f"Joint modificado: {updated_joint_tag}")
    print(f"Actuador modificado: {updated_actuator_tag}")


if __name__ == "__main__":
    main()
