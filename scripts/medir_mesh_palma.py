#!/usr/bin/env python3
import argparse
import os
import sys

try:
    import trimesh
except ImportError:
    print("ERROR: falta instalar trimesh.")
    print("Instala con: python3 -m pip install trimesh")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Mide bounds, extents y centroide de un mesh STL de la mano."
    )
    parser.add_argument(
        "mesh_path",
        help="Ruta absoluta al archivo STL, por ejemplo: /home/sebas/ros2_ws/src/.../L_hand_base_link.STL"
    )
    args = parser.parse_args()

    mesh_path = os.path.expanduser(args.mesh_path)

    if not os.path.isfile(mesh_path):
        print(f"ERROR: no existe el archivo:\n{mesh_path}")
        sys.exit(1)

    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.is_empty:
        print("ERROR: el mesh está vacío o no se pudo cargar correctamente.")
        sys.exit(1)

    bounds = mesh.bounds
    extents = mesh.extents
    centroid = mesh.centroid
    center_bounds = (bounds[0] + bounds[1]) / 2.0

    print("\n=== MEDICIÓN DEL MESH ===")
    print(f"Archivo: {mesh_path}")

    print("\nBounds [min, max] en frame local del mesh:")
    print(f"X: {bounds[0][0]: .6f}  a  {bounds[1][0]: .6f}")
    print(f"Y: {bounds[0][1]: .6f}  a  {bounds[1][1]: .6f}")
    print(f"Z: {bounds[0][2]: .6f}  a  {bounds[1][2]: .6f}")

    print("\nDimensiones del mesh:")
    print(f"Ancho X: {extents[0]: .6f}")
    print(f"Largo Y: {extents[1]: .6f}")
    print(f"Alto  Z: {extents[2]: .6f}")

    print("\nCentro geométrico por bounds:")
    print(f"Xc: {center_bounds[0]: .6f}")
    print(f"Yc: {center_bounds[1]: .6f}")
    print(f"Zc: {center_bounds[2]: .6f}")

    print("\nCentroide del mesh:")
    print(f"X: {centroid[0]: .6f}")
    print(f"Y: {centroid[1]: .6f}")
    print(f"Z: {centroid[2]: .6f}")

    print("\n=== SUGERENCIA DE LECTURA ===")
    print("Si las dimensiones están alrededor de 0.05 a 0.20, probablemente están en metros.")
    print("Si aparecen alrededor de 50 a 200, probablemente están en milímetros.")
    print("Para calibrar L_palm_tcp, usa estos datos como referencia, pero valida visualmente en RViz.")
    print("No busques necesariamente el centro geométrico; busca el centro funcional del canal pulgar-dedos.\n")


if __name__ == "__main__":
    main()