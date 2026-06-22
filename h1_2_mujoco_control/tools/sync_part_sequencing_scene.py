#!/usr/bin/env python3

import math
from typing import Dict, Tuple

import mujoco
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from tf2_ros import StaticTransformBroadcaster


MODEL_PATH = "/home/jpgb/tesisws/src/h1_2_utec/h1_2_description/mjcf/scene_part_sequencing_calibrated_v12_55mm_cube.xml"
BASE_BODY = "pelvis"
BASE_FRAME = "pelvis"

# Objeto inicial para la prueba de secuenciación.
# Red cube está en el estante izquierdo, y=positivo.
OBJECT_BODY = "seq_cube_red"
OBJECT_FRAME = "objeto_cubo"
OBJECT_DIMENSION = 0.055

# Para cubos simétricos, este yaw define la cara frontal de agarre.
# +90° hace que approach_dir_world quede alineado con -X,
# coherente con shelf_out_dir_world = [-1, 0, 0].
OBJECT_YAW_OVERRIDE_RAD = math.pi / 2.0

# Mesa real de MuJoCo.
TABLE_BODY = "transfer_table"
TABLE_TOP_GEOM = "transfer_table_top"
ARUCO_FRAME = "aruco_mesa"

# Geometrías de entorno que se replican como cajas de colisión en MoveIt.
ENV_GEOMS = {
    # Mesa: solo tablero superior.
    "mjc_transfer_table_top": "transfer_table_top",

    # Estante izquierdo.
    "mjc_left_shelf_lower": "left_shelf_lower",
    "mjc_left_shelf_middle": "left_shelf_middle",
    "mjc_left_shelf_upper": "left_shelf_upper",
    "mjc_left_shelf_side_a": "left_shelf_side_a",
    "mjc_left_shelf_side_b": "left_shelf_side_b",

    # Estante derecho.
    "mjc_right_shelf_lower": "right_shelf_lower",
    "mjc_right_shelf_middle": "right_shelf_middle",
    "mjc_right_shelf_upper": "right_shelf_upper",
    "mjc_right_shelf_side_a": "right_shelf_side_a",
    "mjc_right_shelf_side_b": "right_shelf_side_b",
}



def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def quat_from_matrix(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convierte matriz de rotación 3x3 a quaternion xyzw."""
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return float(qx), float(qy), float(qz), float(qw)


class SceneSync(Node):
    def __init__(self):
        super().__init__("sync_part_sequencing_scene")

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.base_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            BASE_BODY,
        )

        if self.base_id < 0:
            raise RuntimeError(f"No se encontró body base '{BASE_BODY}' en el MJCF.")

        self.base_pos_w = np.array(self.data.xpos[self.base_id], dtype=float)
        self.base_R_w = np.array(self.data.xmat[self.base_id], dtype=float).reshape(3, 3)
        self.R_w_to_base = self.base_R_w.T

        self.get_logger().info(f"Modelo cargado: {MODEL_PATH}")
        self.get_logger().info(
            f"Pelvis MuJoCo en mundo: "
            f"[{self.base_pos_w[0]:.4f}, {self.base_pos_w[1]:.4f}, {self.base_pos_w[2]:.4f}]"
        )

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.apply_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        self.get_logger().info("Esperando /apply_planning_scene...")
        if not self.apply_scene_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("No está disponible /apply_planning_scene. Verifica que move_group esté corriendo.")

        self.publish_static_tfs()
        self.apply_scene()

        # No reaplicar periódicamente la PlanningScene.
        # La primera aplicación es suficiente mientras move_group siga vivo.
        # Mantener el nodo vivo conserva los TFs estáticos publicados.
        self.get_logger().info("Sincronización inicial completa. Nodo activo para mantener TFs.")

    def pos_world_to_base(self, pos_w: np.ndarray) -> np.ndarray:
        return self.R_w_to_base @ (pos_w - self.base_pos_w)

    def rot_world_to_base(self, R_w: np.ndarray) -> np.ndarray:
        return self.R_w_to_base @ R_w

    def body_pose_base(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"No existe body '{body_name}'")

        pos_w = np.array(self.data.xpos[body_id], dtype=float)
        R_w = np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3)

        return self.pos_world_to_base(pos_w), self.rot_world_to_base(R_w)

    def geom_pose_base(self, geom_name: str) -> Tuple[np.ndarray, np.ndarray]:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise RuntimeError(f"No existe geom '{geom_name}'")

        pos_w = np.array(self.data.geom_xpos[geom_id], dtype=float)
        R_w = np.array(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)

        return self.pos_world_to_base(pos_w), self.rot_world_to_base(R_w)

    def geom_box_full_size(self, geom_name: str) -> Tuple[float, float, float]:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise RuntimeError(f"No existe geom '{geom_name}'")

        geom_type = int(self.model.geom_type[geom_id])
        size = np.array(self.model.geom_size[geom_id], dtype=float)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float(2.0 * size[0]), float(2.0 * size[1]), float(2.0 * size[2])

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            d = float(2.0 * size[0])
            return d, d, d

        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return float(2.0 * size[0]), float(2.0 * size[0]), float(2.0 * size[1])

        # Fallback conservador.
        return 0.05, 0.05, 0.05

    def make_tf(self, child: str, pos: np.ndarray, R: np.ndarray) -> TransformStamped:
        qx, qy, qz, qw = quat_from_matrix(R)

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.child_frame_id = child

        msg.transform.translation.x = float(pos[0])
        msg.transform.translation.y = float(pos[1])
        msg.transform.translation.z = float(pos[2])

        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        return msg

    def publish_static_tfs(self):
        # TF del cubo real en MuJoCo.
        # Posición viene de MuJoCo; orientación se fija para definir
        # una cara de agarre coherente con la apertura frontal del estante.
        obj_pos, _obj_R_real = self.body_pose_base(OBJECT_BODY)
        obj_R = rotz(OBJECT_YAW_OVERRIDE_RAD)

        # TF del ArUco virtual: centro superior real de la mesa.
        table_top_pos, table_top_R = self.geom_pose_base(TABLE_TOP_GEOM)
        _, _, table_top_thickness = self.geom_box_full_size(TABLE_TOP_GEOM)

        # Convertimos la normal Z local de la mesa al frame pelvis.
        table_z_axis_base = table_top_R @ np.array([0.0, 0.0, 1.0])
        aruco_pos = table_top_pos + table_z_axis_base * (table_top_thickness / 2.0)
        aruco_R = table_top_R

        tfs = [self.make_tf(ARUCO_FRAME, aruco_pos, aruco_R)]

        self.tf_broadcaster.sendTransform(tfs)

        self.get_logger().info(
            f"TF {OBJECT_FRAME} será publicada dinámicamente por el bridge MuJoCo."
        )
        self.get_logger().info(
            f"TF {ARUCO_FRAME} desde superficie de {TABLE_TOP_GEOM}: "
            f"[{aruco_pos[0]:.4f}, {aruco_pos[1]:.4f}, {aruco_pos[2]:.4f}]"
        )
        self.get_logger().info(
            f"Dimensión del objeto para mover_brazo_single_aruco.launch.py: {OBJECT_DIMENSION:.3f} m"
        )

    def make_collision_box(self, object_id: str, geom_name: str) -> CollisionObject:
        pos, R = self.geom_pose_base(geom_name)
        sx, sy, sz = self.geom_box_full_size(geom_name)
        qx, qy, qz, qw = quat_from_matrix(R)

        obj = CollisionObject()
        obj.header.frame_id = BASE_FRAME
        obj.id = object_id
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [sx, sy, sz]

        pose = Pose()
        pose.position.x = float(pos[0])
        pose.position.y = float(pos[1])
        pose.position.z = float(pos[2])
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)

        return obj

    def apply_scene(self):
        scene = PlanningScene()
        scene.is_diff = True

        for object_id, geom_name in ENV_GEOMS.items():
            scene.world.collision_objects.append(
                self.make_collision_box(object_id, geom_name)
            )

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self.apply_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info("PlanningScene sincronizada con MuJoCo.")
        else:
            self.get_logger().error("No se pudo aplicar PlanningScene.")


def main():
    rclpy.init()
    node = SceneSync()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
