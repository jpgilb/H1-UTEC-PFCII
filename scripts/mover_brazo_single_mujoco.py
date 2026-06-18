#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mover_brazo_single_face_approach.py

Prueba monomanual de agarre con dedos para Unitree H1-2 en RViz/MoveIt 2.

Supuesto geométrico usado:
- L_palm_tcp / R_palm_tcp ya fue calibrado en el URDF/Xacro como TCP superficial
  de la palma, no como centro de la palma.
- El eje +X local del TCP apunta hacia el lado opuesto de la palma, es decir,
  hacia donde debe quedar el objeto.
- Por eso, el TCP NO se manda al centro del cubo/esfera. Se manda a la cara del
  objeto. El centro del objeto queda a:
      centro_objeto = TCP + R_tcp * [radio_objeto + margen_superficie, 0, 0]

Flujo:
1) preparar escena y abrir mano
2) preagarre elevado: desplazado en +Z y separado en -X local respecto a la cara del objeto
3) descenso vertical
4) aproximación frontal sobre +X local hacia la cara del objeto
5) cierre adaptativo de mano
6) validación opcional + attach
7) retirada vertical
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker


@dataclass
class ArmHandConfig:
    side: str
    arm_group: str
    hand_group: str
    ee_link: str
    hand_controller_action: str
    arm_joints: List[str]
    hand_joints: List[str]
    link_prefix: str


class SingleArmFaceApproachGraspNode(Node):
    def __init__(self) -> None:
        super().__init__('single_arm_face_approach_grasp_node')
        # ======================================================================
        # 1. PARÁMETROS GENERALES
        # ======================================================================
        self.declare_parameter('arm_side', 'left')                 # Lado del brazo a usar ('left' o 'right')
        self.declare_parameter('base_frame', 'pelvis')             # Frame base de referencia del robot
        self.declare_parameter('object_frame', 'objeto_cubo')      # Frame TF2 del objeto a manipular
        self.declare_parameter('object_type', 'cube')              # Tipo de objeto ('cube' o 'sphere')
        self.declare_parameter('object_dimension', 0.055)          # Lado del cubo o diámetro de la esfera [m]

        # ======================================================================
        # 2. PARÁMETROS DE ESCENA
        # ======================================================================
        self.declare_parameter('table_collision', True)            # Activar/desactivar colisión con la mesa de trabajo

        # ======================================================================
        # 3. PARÁMETROS DE PICK (AGARRE)
        # ======================================================================
        self.declare_parameter('hover_height', 0.120)              # Altura elevada sobre el objeto para preagarre [m]
        self.declare_parameter('approach_distance', 0.080)         # Distancia de aproximación desde atrás en el eje -X local [m]
        self.declare_parameter('surface_clearance', 0.008)         # Distancia mínima entre la palma y la cara del objeto [m]
        self.declare_parameter('lift_distance', 0.150)             # Altura de elevación vertical tras el agarre [m]
        self.declare_parameter('dz_offset', -0.020)                # Ajuste vertical del TCP (baja la mano para centrar dedos) [m]

        # ======================================================================
        # 4. PARÁMETROS DE PLACE (COLOCACIÓN)
        # ======================================================================
        self.declare_parameter('place_x', 0.50)                    # Coordenada X final del objeto en la mesa [m]
        self.declare_parameter('place_y', 0.00)                    # Coordenada Y final del objeto en la mesa [m]
        self.declare_parameter('place_object_z', -9999.0)          # Altura Z del objeto en mesa (si es -9999.0 usa la Z detectada) [m]
        self.declare_parameter('place_hover_height', 0.120)        # Altura elevada del TCP sobre la mesa para aproximarse al colocar [m]
        self.declare_parameter('post_place_retreat_height', 0.120) # Altura de retirada segura de la muñeca tras soltar [m]
        self.declare_parameter('place_z_margin', 0.002)            # Tolerancia de elevación Z para asentar el objeto [m]

        # ======================================================================
        # 5. PARÁMETROS DE MANO Y DEDOS (FRACCIONES DE FLEXIÓN)
        # ======================================================================
        self.declare_parameter('ring_finger_fraction', 0.0)        # Multiplicador flexión anular (0.0 = quieto)
        self.declare_parameter('pinky_finger_fraction', 0.0)       # Multiplicador flexión meñique (0.0 = quieto)
        self.declare_parameter('index_finger_fraction', 0.0)       # Multiplicador flexión índice [0.0 a 1.0]
        self.declare_parameter('middle_finger_fraction', 0.0)      # Multiplicador flexión medio [0.0 a 1.0]
        self.declare_parameter('thumb_pitch_fraction', 0.00)       # Multiplicador flexión pitch del pulgar [0.0 a 1.0]
        self.declare_parameter('thumb_yaw_fraction', 0.0)         # Multiplicador rotación yaw del pulgar (oposición) [0.0 a 1.0]

        # ======================================================================
        # 6. PARÁMETROS DE ESPEJO (CALIBRACIÓN YAW DEL TCP POR LADO)
        # ======================================================================
        self.declare_parameter('yaw_offset_left_deg', 90.0)        # Rotación base en grados para el brazo izquierdo
        self.declare_parameter('yaw_offset_right_deg', 90.0)     # Rotación base en grados para el brazo derecho (espejo)
        self.declare_parameter('pitch_offset_left_deg', 0.0)       # Pitch base en grados para el brazo izquierdo
        self.declare_parameter('pitch_offset_right_deg', -179.9)   # Pitch base en grados para el brazo derecho (espejo)
        self.declare_parameter('auto_select_arm_by_y', True)      # Seleccionar brazo automáticamente según la coordenada Y del objeto
        self.declare_parameter('enforce_place_side_consistency', False) # Forzar consistencia de lado de colocación en mesa

        self.declare_parameter('task_mode', 'bimanual_transfer')
        self.declare_parameter('table_transfer_y', 0.0)
        self.declare_parameter('use_initial_x_for_table', True)
        self.declare_parameter('use_initial_z_for_table', True)
        self.declare_parameter('final_mirror_y', True)
        self.declare_parameter('final_mirror_yaw', True)
        self.declare_parameter('table_object_yaw_deg', 0.0)
        self.declare_parameter('keep_final_x_from_initial', True)
        self.declare_parameter('keep_final_z_from_initial', True)

        # ======================================================================
        # 7. PARÁMETROS DE DIAGNÓSTICO Y SEGURIDAD
        # ======================================================================
        self.declare_parameter('publish_debug_markers', True)      # Publicar esferas y flechas visuales de depuración en RViz
        self.declare_parameter('diagnostic_ik_without_collisions', False) # Diagnosticar fallas cinemáticas desactivando colisiones en IK
        self.declare_parameter('allow_attach_without_effort', True)# Continuar cinemáticamente si no hay propiocepción
        self.declare_parameter('grasp_effort_threshold', 0.02)     # Umbral de esfuerzo para cierre táctil activo [N*m]
        self.declare_parameter('min_grasp_flexion', 0.10)          # Flexión mínima para ignorar transitorios de aceleración [rad]
        self.declare_parameter('penetration_limit', 0.002)         # Límite de penetración en el cubo para el sensor virtual [m]
        self.declare_parameter('touch_mode', 'permissive')         # Modo de touch links ('permissive' | 'fingers_only' | 'none')
        self.declare_parameter('use_inner_collision_object', True) # Activar caja de colisión interna para detección de presión

        # ======================================================================
        # 8. PARÁMETROS DE RETIRADA Y MOVIMIENTOS CARTESIANOS
        # ======================================================================
        self.declare_parameter('use_split_place_retreat', False)
        self.declare_parameter('use_cartesian_local_motions', False)

        # Modo de integración con MuJoCo:
        # - No espera controladores físicos de mano.
        # - Permite detener la secuencia tras una fase concreta.
        self.declare_parameter('mujoco_bridge_mode', True)
        self.declare_parameter('stop_after_phase', 4)
        self.declare_parameter('mujoco_disable_planning_collisions', True)
        self.declare_parameter('mujoco_split_approach', True)
        self.declare_parameter('mujoco_approach_steps', 3)
        self.declare_parameter('mujoco_lift_steps', 3)
        self.declare_parameter('mujoco_place_descent_steps', 5)
        self.declare_parameter('mujoco_wait_after_arm_motion_sec', 3.0)

        # --------------------------
        # CARGAR VALORES DE PARÁMETROS
        # --------------------------
        self.arm_side = str(self.get_parameter('arm_side').value).lower().strip()
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.object_frame = str(self.get_parameter('object_frame').value)
        self.object_type = str(self.get_parameter('object_type').value).lower().strip()
        self.object_dimension = float(self.get_parameter('object_dimension').value)
        self.use_table_collision = bool(self.get_parameter('table_collision').value)

        self.hover_height = float(self.get_parameter('hover_height').value)
        self.approach_distance = float(self.get_parameter('approach_distance').value)
        self.surface_clearance = float(self.get_parameter('surface_clearance').value)
        self.lift_distance = float(self.get_parameter('lift_distance').value)
        self.dz_offset = float(self.get_parameter('dz_offset').value)

        self.place_x = float(self.get_parameter('place_x').value)
        self.place_y = float(self.get_parameter('place_y').value)
        self.place_object_z = float(self.get_parameter('place_object_z').value)
        self.place_hover_height = float(self.get_parameter('place_hover_height').value)
        self.post_place_retreat_height = float(self.get_parameter('post_place_retreat_height').value)
        self.place_z_margin = float(self.get_parameter('place_z_margin').value)

        self.ring_finger_fraction = float(self.get_parameter('ring_finger_fraction').value)
        self.pinky_finger_fraction = float(self.get_parameter('pinky_finger_fraction').value)
        self.index_finger_fraction = float(self.get_parameter('index_finger_fraction').value)
        self.middle_finger_fraction = float(self.get_parameter('middle_finger_fraction').value)
        self.thumb_pitch_fraction = float(self.get_parameter('thumb_pitch_fraction').value)
        self.thumb_yaw_fraction = float(self.get_parameter('thumb_yaw_fraction').value)

        self.yaw_offset_left_deg = float(self.get_parameter('yaw_offset_left_deg').value)
        self.yaw_offset_right_deg = float(self.get_parameter('yaw_offset_right_deg').value)
        self.pitch_offset_left_deg = float(self.get_parameter('pitch_offset_left_deg').value)
        self.pitch_offset_right_deg = float(self.get_parameter('pitch_offset_right_deg').value)
        self.auto_select_arm_by_y = bool(self.get_parameter('auto_select_arm_by_y').value)
        self.enforce_place_side_consistency = bool(self.get_parameter('enforce_place_side_consistency').value)

        self.task_mode = str(self.get_parameter('task_mode').value).strip()
        self.table_transfer_y = float(self.get_parameter('table_transfer_y').value)
        self.use_initial_x_for_table = bool(self.get_parameter('use_initial_x_for_table').value)
        self.use_initial_z_for_table = bool(self.get_parameter('use_initial_z_for_table').value)
        self.final_mirror_y = bool(self.get_parameter('final_mirror_y').value)
        self.final_mirror_yaw = bool(self.get_parameter('final_mirror_yaw').value)
        self.table_object_yaw_deg = float(self.get_parameter('table_object_yaw_deg').value)
        self.keep_final_x_from_initial = bool(self.get_parameter('keep_final_x_from_initial').value)
        self.keep_final_z_from_initial = bool(self.get_parameter('keep_final_z_from_initial').value)

        self.publish_debug_markers = bool(self.get_parameter('publish_debug_markers').value)
        self.diagnostic_ik_without_collisions = bool(self.get_parameter('diagnostic_ik_without_collisions').value)
        self.allow_attach_without_effort = bool(self.get_parameter('allow_attach_without_effort').value)
        self.grasp_effort_threshold = float(self.get_parameter('grasp_effort_threshold').value)
        self.min_grasp_flexion = float(self.get_parameter('min_grasp_flexion').value)
        self.penetration_limit = float(self.get_parameter('penetration_limit').value)
        self.touch_mode = str(self.get_parameter('touch_mode').value).lower().strip()
        self.use_inner_collision_object = bool(self.get_parameter('use_inner_collision_object').value)
        self.use_split_place_retreat = bool(self.get_parameter('use_split_place_retreat').value)
        self.use_cartesian_local_motions = bool(self.get_parameter('use_cartesian_local_motions').value)
        self.mujoco_bridge_mode = bool(self.get_parameter('mujoco_bridge_mode').value)
        self.stop_after_phase = int(self.get_parameter('stop_after_phase').value)
        self.mujoco_disable_planning_collisions = bool(
            self.get_parameter('mujoco_disable_planning_collisions').value
        )
        self.mujoco_split_approach = bool(self.get_parameter('mujoco_split_approach').value)
        self.mujoco_approach_steps = int(self.get_parameter('mujoco_approach_steps').value)
        self.mujoco_lift_steps = int(self.get_parameter('mujoco_lift_steps').value)
        self.mujoco_place_descent_steps = int(self.get_parameter('mujoco_place_descent_steps').value)
        self.mujoco_wait_after_arm_motion_sec = float(
            self.get_parameter('mujoco_wait_after_arm_motion_sec').value
        )

        self.get_logger().info(
            f'Modo MuJoCo bridge: {self.mujoco_bridge_mode} | '
            f'stop_after_phase={self.stop_after_phase} | '
            f'mujoco_disable_planning_collisions={self.mujoco_disable_planning_collisions}'
        )

        if self.arm_side not in ('left', 'right'):
            raise ValueError('arm_side debe ser "left" o "right"')
        if self.object_type not in ('cube', 'sphere'):
            raise ValueError('object_type debe ser "cube" o "sphere"')
        if self.object_dimension <= 0.0:
            raise ValueError('object_dimension debe ser mayor a cero')

        self.cfg = self._build_arm_hand_config(self.arm_side)

        # --------------------------
        # Estado interno
        # --------------------------
        self.current_phase = 0
        self.object_position = np.zeros(3, dtype=float)
        self.object_yaw_rad = 0.0
        self.object_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # Variables de control para transferencia bimanual
        self.initial_object_position = np.zeros(3, dtype=float)
        self.initial_object_yaw_rad = 0.0
        self.initial_arm_side = 'left'
        self.second_arm_side = 'right'
        self.transfer_stage = 'source_to_table'
        self.target_object_yaw_rad = 0.0
        self.tcp_place_retreat = np.zeros(3, dtype=float)
        self.retreat_done = False

        self.tcp_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.tcp_pregrasp = np.zeros(3, dtype=float)
        self.tcp_ready = np.zeros(3, dtype=float)
        self.tcp_contact = np.zeros(3, dtype=float)
        self.approach_dir_world = np.array([1.0, 0.0, 0.0], dtype=float)

        # Variables de colocación (Place)
        self.place_object_center = np.zeros(3, dtype=float)
        self.tcp_place_above = np.zeros(3, dtype=float)
        self.tcp_place = np.zeros(3, dtype=float)
        self.tcp_quat_place = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # Estado para aproximación frontal segmentada en MuJoCo
        self.approach_substep_index = 0
        self.approach_substep_targets = []
        self.lift_substep_index = 0
        self.lift_substep_targets = []
        self.place_descent_substep_index = 0
        self.place_descent_substep_targets = []

        self.last_joint_state: Optional[JointState] = None
        self.pre_close_effort: Optional[Dict[str, float]] = None
        self.last_tick_effort: Optional[Dict[str, float]] = None
        self.effort_available = False

        # --------------------------
        # ROS 2
        # --------------------------
        self.cb_group = ReentrantCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_group_client = ActionClient(
            self, MoveGroup, 'move_action', callback_group=self.cb_group
        )
        self.ik_client = self.create_client(
            GetPositionIK, 'compute_ik', callback_group=self.cb_group
        )
        self.hand_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.cfg.hand_controller_action,
            callback_group=self.cb_group,
        )

        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.attach_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        self.marker_pub = self.create_publisher(Marker, '/adaptive_grasp_markers', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio /compute_ik...')

        self.get_logger().info(
            f'Nodo listo. Brazo={self.cfg.arm_group}, mano={self.cfg.hand_group}, '
            f'TCP={self.cfg.ee_link}, objeto={self.object_frame}, tipo={self.object_type}'
        )

        self.timer = self.create_timer(1.0, self.buscar_tf_y_ejecutar, callback_group=self.cb_group)

    # ======================================================================
    # Configuración
    # ======================================================================

    def _build_arm_hand_config(self, side: str) -> ArmHandConfig:
        if side == 'left':
            return ArmHandConfig(
                side='left',
                arm_group='left_arm',
                hand_group='left_hand',
                ee_link='L_palm_tcp',
                hand_controller_action='/left_hand_controller/follow_joint_trajectory',
                arm_joints=[
                    'left_shoulder_pitch_joint',
                    'left_shoulder_roll_joint',
                    'left_shoulder_yaw_joint',
                    'left_elbow_joint',
                    'left_wrist_roll_joint',
                    'left_wrist_pitch_joint',
                    'left_wrist_yaw_joint',
                ],
                hand_joints=[
                    'L_index_proximal_joint',
                    'L_middle_proximal_joint',
                    'L_pinky_proximal_joint',
                    'L_ring_proximal_joint',
                    'L_thumb_proximal_pitch_joint',
                    'L_thumb_proximal_yaw_joint',
                ],
                link_prefix='L_',
            )

        return ArmHandConfig(
            side='right',
            arm_group='right_arm',
            hand_group='right_hand',
            ee_link='R_palm_tcp',
            hand_controller_action='/right_hand_controller/follow_joint_trajectory',
            arm_joints=[
                'right_shoulder_pitch_joint',
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint',
                'right_elbow_joint',
                'right_wrist_roll_joint',
                'right_wrist_pitch_joint',
                'right_wrist_yaw_joint',
            ],
            hand_joints=[
                'R_index_proximal_joint',
                'R_middle_proximal_joint',
                'R_pinky_proximal_joint',
                'R_ring_proximal_joint',
                'R_thumb_proximal_pitch_joint',
                'R_thumb_proximal_yaw_joint',
            ],
            link_prefix='R_',
        )

    def configurar_brazo_activo(self, side: str) -> None:
        self.arm_side = side
        self.cfg = self._build_arm_hand_config(side)
        self.hand_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.cfg.hand_controller_action,
            callback_group=self.cb_group,
        )
        self.get_logger().info(
            f'[Brazo Activo] Brazo: {self.arm_side.upper()} | '
            f'Grupo MoveIt: {self.cfg.arm_group} | '
            f'TCP Link: {self.cfg.ee_link} | '
            f'Controlador Mano: {self.cfg.hand_controller_action}'
        )

    def calcular_destino_mesa(self) -> np.ndarray:
        x = self.initial_object_position[0] if self.use_initial_x_for_table else self.place_x
        y = self.table_transfer_y
        if self.use_initial_z_for_table:
            z = self.initial_object_position[2]
        else:
            z = self.initial_object_position[2] if self.place_object_z == -9999.0 else self.place_object_z
        z += self.place_z_margin
        return np.array([x, y, z], dtype=float)

    def calcular_destino_final_espejado(self) -> np.ndarray:
        x = self.initial_object_position[0] if self.keep_final_x_from_initial else self.place_x
        y = -self.initial_object_position[1] if self.final_mirror_y else self.place_y
        if self.keep_final_z_from_initial:
            z = self.initial_object_position[2]
        else:
            z = self.initial_object_position[2] if self.place_object_z == -9999.0 else self.place_object_z
        z += self.place_z_margin
        return np.array([x, y, z], dtype=float)

    def reiniciar_ciclo_para_siguiente_etapa(self) -> None:
        self.current_phase = 0
        self.pre_close_effort = None
        self.last_tick_effort = None
        self.retreat_done = False
        self.tcp_place_retreat = np.zeros(3, dtype=float)
        self.tcp_place = np.zeros(3, dtype=float)
        self.tcp_place_above = np.zeros(3, dtype=float)

        self.object_position = self.place_object_center.copy()
        self.object_yaw_rad = self.target_object_yaw_rad
        self.transfer_stage = 'table_to_target'

        self.get_logger().info(f'[Transición] Pose actual del objeto en mesa: {self.object_position.tolist()}')
        self.get_logger().info(f'[Transición] Yaw actual del objeto en mesa: {math.degrees(self.object_yaw_rad):.2f}°')
        self.get_logger().info(f'[Transición] Brazo opuesto seleccionado: {self.second_arm_side.upper()}')

        dest_final = self.calcular_destino_final_espejado()
        yaw_final = -self.initial_object_yaw_rad if self.final_mirror_yaw else self.initial_object_yaw_rad
        self.get_logger().info(f'[Transición] Destino final espejado: {dest_final.tolist()}')
        self.get_logger().info(f'[Transición] Yaw final espejado: {math.degrees(yaw_final):.2f}°')

        self.configurar_brazo_activo(self.second_arm_side)

        self.ejecutar_fase_1_preparar_escena_y_abrir_mano()

    # ======================================================================
    # Joint states / propiocepción
    # ======================================================================

    def joint_state_callback(self, msg: JointState) -> None:
        self.last_joint_state = msg
        if len(msg.effort) == len(msg.name) and len(msg.effort) > 0:
            self.effort_available = any(abs(e) > 1.0e-6 for e in msg.effort)

    def snapshot_effort(self) -> Optional[Dict[str, float]]:
        if self.last_joint_state is None:
            return None
        if len(self.last_joint_state.effort) != len(self.last_joint_state.name):
            return None
        return {
            name: float(effort)
            for name, effort in zip(self.last_joint_state.name, self.last_joint_state.effort)
            if name in self.cfg.hand_joints
        }

    def evaluar_agarre_propioceptivo(self) -> Tuple[bool, str]:
        if not self.effort_available:
            if self.allow_attach_without_effort:
                return True, 'Sin effort físico disponible; se continúa en validación cinemática RViz/MoveIt.'
            return False, 'Sin effort físico disponible y allow_attach_without_effort=False.'

        post = self.snapshot_effort()
        if not self.pre_close_effort or not post:
            return self.allow_attach_without_effort, 'No se pudo calcular línea base de effort.'

        deltas = [
            abs(post[j] - self.pre_close_effort[j])
            for j in self.cfg.hand_joints
            if j in post and j in self.pre_close_effort
        ]
        if not deltas:
            return self.allow_attach_without_effort, 'Sin joints de mano comparables en effort.'

        mean_delta = float(np.mean(deltas))
        if mean_delta > self.grasp_effort_threshold:
            return True, f'Agarre probable por incremento de effort medio: {mean_delta:.4f} N*m.'
 
        return self.allow_attach_without_effort, f'Incremento de effort bajo: {mean_delta:.4f} N*m.'

    # ======================================================================
    # TF y marcadores
    # ======================================================================

    def buscar_tf_y_ejecutar(self) -> None:
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.object_frame,
                rclpy.time.Time(),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info(f'Esperando TF2 de "{self.object_frame}" hacia "{self.base_frame}"...')
            return

        self.timer.cancel()
        self.retreat_done = False

        self.object_position = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ], dtype=float)

        q = trans.transform.rotation
        self.object_quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=float)
        self.object_yaw_rad = float(
            R.from_quat(self.object_quat_xyzw).as_euler('xyz', degrees=False)[2]
        )

        self.get_logger().info(
            f'TF encontrada. Posición objeto={self.object_position.tolist()}, '
            f'yaw={math.degrees(self.object_yaw_rad):.2f} deg'
        )

        # Copias al inicio
        self.initial_object_position = np.copy(self.object_position)
        self.initial_object_yaw_rad = self.object_yaw_rad

        # Logs de inicialización
        self.get_logger().info(f'Modo de tarea: {self.task_mode}')
        self.get_logger().info(f'Pose inicial del objeto: {self.initial_object_position.tolist()}')
        self.get_logger().info(f'Yaw inicial del objeto: {math.degrees(self.initial_object_yaw_rad):.2f} deg')

        if self.task_mode == 'bimanual_transfer':
            self.transfer_stage = 'source_to_table'
            if self.initial_object_position[1] >= 0.0:
                self.initial_arm_side = 'left'
                self.second_arm_side = 'right'
            else:
                self.initial_arm_side = 'right'
                self.second_arm_side = 'left'
            self.arm_side = self.initial_arm_side

            self.get_logger().info(f'Primer brazo seleccionado: {self.initial_arm_side.upper()}')
            self.get_logger().info(f'Segundo brazo seleccionado: {self.second_arm_side.upper()}')

            dest_mesa = self.calcular_destino_mesa()
            self.get_logger().info(f'Destino de mesa calculado: {dest_mesa.tolist()}')

            dest_final = self.calcular_destino_final_espejado()
            self.get_logger().info(f'Destino final espejado calculado: {dest_final.tolist()}')
        else:
            # Switch inteligente de brazo basado en la Y del objeto o respeto de arm_side manual
            if self.auto_select_arm_by_y:
                if self.object_position[1] >= 0.0:
                    self.arm_side = 'left'
                else:
                    self.arm_side = 'right'
                self.get_logger().info(
                    f'Selección AUTOMÁTICA de brazo (Y_objeto = {self.object_position[1]:.4f} m): '
                    f'se seleccionó el brazo {self.arm_side.upper()}.'
                )
            else:
                self.get_logger().info(
                    f'Selección MANUAL de brazo respetada: '
                    f'brazo {self.arm_side.upper()}.'
                )

        # Validación de consistencia para la colocación en mesa
        if self.enforce_place_side_consistency:
            if self.arm_side == 'left' and self.place_y < 0.0:
                self.get_logger().error(
                    f'Error de consistencia: Brazo IZQUIERDO seleccionado pero place_y es negativo ({self.place_y:.4f} m). '
                    'Abortando para evitar trayectoria cruzada.'
                )
                rclpy.shutdown()
                return
            elif self.arm_side == 'right' and self.place_y > 0.0:
                self.get_logger().error(
                    f'Error de consistencia: Brazo DERECHO seleccionado pero place_y es positivo ({self.place_y:.4f} m). '
                    'Abortando para evitar trayectoria cruzada.'
                )
                rclpy.shutdown()
                return

        # Reconfigurar de forma dinámica
        self.configurar_brazo_activo(self.arm_side)

        if self.publish_debug_markers:
            self.publicar_marker(self.object_position, 0, 'objeto', [0.1, 0.8, 0.1, 0.70], 0.045)

        self.ejecutar_fase_1_preparar_escena_y_abrir_mano()

    def publicar_marker(
        self,
        position: np.ndarray,
        marker_id: int,
        namespace: str,
        rgba: List[float],
        escala: float = 0.04,
        marker_type: int = Marker.SPHERE,
        quat: Optional[np.ndarray] = None,
        scale_xyz: Optional[List[float]] = None,
    ) -> None:
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])

        if quat is not None:
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])
        else:
            marker.pose.orientation.w = 1.0

        if scale_xyz is not None:
            marker.scale.x = float(scale_xyz[0])
            marker.scale.y = float(scale_xyz[1])
            marker.scale.z = float(scale_xyz[2])
        else:
            marker.scale.x = float(escala)
            marker.scale.y = float(escala)
            marker.scale.z = float(escala)

        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        self.marker_pub.publish(marker)

    # ======================================================================
    # Escena de planificación
    # ======================================================================

    def remover_objetos_mundo(self) -> None:
        for obj_id in ['objeto_manipulado', 'objeto_interno_colision']:
            co = CollisionObject()
            co.header.frame_id = self.base_frame
            co.id = obj_id
            co.operation = CollisionObject.REMOVE
            self.collision_pub.publish(co)

    def publicar_objeto_mundo(self, posicion: np.ndarray, interno: bool = False) -> None:
        if interno and not self.use_inner_collision_object:
            return
        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.id = 'objeto_interno_colision' if interno else 'objeto_manipulado'
        obj.operation = CollisionObject.ADD
        obj.primitives.append(self.crear_primitiva_objeto(interno=interno))
        obj.primitive_poses.append(self.crear_pose_objeto_en(posicion))
        self.collision_pub.publish(obj)

    def remover_objetos_adjuntos(self) -> None:
        for obj_id in ['objeto_manipulado', 'objeto_interno_colision']:
            aco = AttachedCollisionObject()
            aco.link_name = self.cfg.ee_link
            aco.object.id = obj_id
            aco.object.operation = CollisionObject.REMOVE
            self.attach_pub.publish(aco)

    def adjuntar_objeto(self, obj_id: str, interno: bool = False, touch_links: List[str] = None) -> None:
        if interno and not self.use_inner_collision_object:
            return
        aco = AttachedCollisionObject()
        aco.link_name = self.cfg.ee_link
        if touch_links is not None:
            aco.touch_links = touch_links

        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.id = obj_id
        obj.operation = CollisionObject.ADD
        obj.primitives.append(self.crear_primitiva_objeto(interno=interno))
        obj.primitive_poses.append(self.crear_pose_objeto_mundo())
        aco.object = obj
        self.attach_pub.publish(aco)

    def limpiar_escena(self) -> None:
        self.get_logger().info('Limpiando escena de planificación...')
        self.remover_objetos_adjuntos()
        self.remover_objetos_mundo()

        # Remover mesa_trabajo
        co = CollisionObject()
        co.header.frame_id = self.base_frame
        co.id = 'mesa_trabajo'
        co.operation = CollisionObject.REMOVE
        self.collision_pub.publish(co)

        time.sleep(0.5)

    def añadir_mesa_colision(self) -> None:
        ancho_mesa = 1.2
        profundidad_mesa = 0.8
        grosor_mesa = 0.05
        dist_robot_mesa = 0.15
        z_superficie_mesa = -0.04

        mesa = CollisionObject()
        mesa.header.frame_id = self.base_frame
        mesa.id = 'mesa_trabajo'
        mesa.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [profundidad_mesa, ancho_mesa, grosor_mesa]

        pose = Pose()
        pose.position.x = dist_robot_mesa + profundidad_mesa / 2.0
        pose.position.y = 0.0
        pose.position.z = z_superficie_mesa - grosor_mesa / 2.0
        pose.orientation.w = 1.0

        mesa.primitives.append(box)
        mesa.primitive_poses.append(pose)
        self.collision_pub.publish(mesa)
        self.get_logger().info('Mesa registrada como geometría de colisión.')
        time.sleep(0.5)

    def crear_primitiva_objeto(self, interno: bool = False) -> SolidPrimitive:
        primitive = SolidPrimitive()
        dim = self.object_dimension
        if interno:
            dim = max(0.005, self.object_dimension - 2.0 * self.penetration_limit)

        if self.object_type == 'cube':
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [dim, dim, dim]
        else:
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [dim / 2.0]
        return primitive

    def crear_pose_objeto_en(self, posicion_objeto: np.ndarray, yaw_objeto_rad: Optional[float] = None) -> Pose:
        pose = Pose()
        pose.position.x = float(posicion_objeto[0])
        pose.position.y = float(posicion_objeto[1])
        pose.position.z = float(posicion_objeto[2])

        if self.object_type == 'cube':
            y_rad = yaw_objeto_rad if yaw_objeto_rad is not None else self.object_yaw_rad
            quat = R.from_euler('xyz', [0.0, 0.0, y_rad], degrees=False).as_quat()
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    def crear_pose_objeto_mundo(self) -> Pose:
        return self.crear_pose_objeto_en(self.object_position)

    def añadir_objeto_colision_mundo(self) -> None:
        self.publicar_objeto_mundo(self.object_position, interno=False)
        self.publicar_objeto_mundo(self.object_position, interno=True)
        self.get_logger().info('Objeto principal e interno registrados como geometrías de colisión.')
        time.sleep(0.5)

    def remover_objeto_mundo(self) -> None:
        self.remover_objetos_mundo()
        time.sleep(0.2)

    def touch_links_dedos(self) -> List[str]:
        if self.touch_mode == 'none':
            return []

        p = self.cfg.link_prefix

        dedos = [
            f'{p}thumb_proximal_base',
            f'{p}thumb_proximal',
            f'{p}thumb_intermediate',
            f'{p}thumb_distal',
            f'{p}index_proximal',
            f'{p}index_intermediate',
            f'{p}index_distal',
            f'{p}middle_proximal',
            f'{p}middle_intermediate',
            f'{p}middle_distal',
        ]

        if self.touch_mode == 'fingers_only':
            return dedos

        # Por defecto es permissive
        return [self.cfg.ee_link, f'{p}hand_base_link'] + dedos

    def adjuntar_objeto_al_tcp(self) -> None:
        self.remover_objetos_mundo()
        time.sleep(0.2)

        # 1. Adjuntar Objeto Principal
        self.adjuntar_objeto('objeto_manipulado', interno=False, touch_links=self.touch_links_dedos())

        # 2. Adjuntar Objeto Interno
        p = self.cfg.link_prefix
        self.adjuntar_objeto('objeto_interno_colision', interno=True, touch_links=[self.cfg.ee_link, f'{p}hand_base_link'])

        self.get_logger().info(
            'Objeto principal e interno adjuntados al TCP. '
            'El objeto interno carece de touch_links en los dedos para simular sensor propioceptivo en RViz.'
        )
        time.sleep(0.8)

    def registrar_objeto_en_destino(self) -> None:
        self.get_logger().info('Registrando objeto en el destino final/mesa...')
        self.remover_objetos_adjuntos()

        self.object_position = self.place_object_center.copy()
        self.object_yaw_rad = self.target_object_yaw_rad

        self.publicar_objeto_mundo(self.object_position, interno=False)
        self.publicar_objeto_mundo(self.object_position, interno=True)

        self.get_logger().info(
            f'Confirmación: Objeto registrado como colisión del mundo en destino. '
            f'Posición: {self.object_position.tolist()} | '
            f'Yaw: {math.degrees(self.object_yaw_rad):.2f}°'
        )
        time.sleep(0.5)

    # ======================================================================
    # Geometría de agarre
    # ======================================================================

    def object_contact_radius(self) -> float:
        # Para cubo: distancia desde centro a cara. Para esfera: radio.
        return self.object_dimension / 2.0

    def orientacion_tcp_para_objeto(self) -> np.ndarray:
        yaw_objeto = self.object_yaw_rad if self.object_type == 'cube' else 0.0
        if self.cfg.side == 'left':
            pitch_deg = self.pitch_offset_left_deg
            yaw_offset_deg = self.yaw_offset_left_deg
        else:
            pitch_deg = self.pitch_offset_right_deg
            yaw_offset_deg = self.yaw_offset_right_deg

        yaw_final_deg = math.degrees(yaw_objeto) + yaw_offset_deg

        self.get_logger().info(
            f'[Orientación TCP Pick] Lado: {self.cfg.side.upper()} | '
            f'pitch_deg: {pitch_deg:.2f} | yaw_final_deg: {yaw_final_deg:.2f} | '
            f'yaw_objeto_deg: {math.degrees(yaw_objeto):.2f}'
        )

        return R.from_euler('xyz', [0.0, pitch_deg, yaw_final_deg], degrees=True).as_quat()

    def calcular_poses_tcp(self) -> None:
        """
        Calcula tres poses:
        - tcp_pregrasp: separado en -X local y elevado en +Z mundo.
        - tcp_ready: separado en -X local a la altura del objeto.
        - tcp_contact: frente a la cara del objeto, sin penetrarla.
        """
        self.tcp_quat = self.orientacion_tcp_para_objeto()
        r_tcp = R.from_quat(self.tcp_quat).as_matrix()

        # +X local apunta hacia el objeto.
        self.approach_dir_world = r_tcp.dot(np.array([1.0, 0.0, 0.0], dtype=float))
        self.approach_dir_world = self.approach_dir_world / np.linalg.norm(self.approach_dir_world)

        # Determinación del offset de aproximación:
        # Tanto para el brazo izquierdo como derecho, el TCP se aproxima de forma
        # frontal (en la dirección del eje local +X) hacia la cara del objeto.
        contact_offset = self.object_contact_radius() + self.surface_clearance

        self.tcp_contact = self.object_position + self.approach_dir_world * contact_offset
        self.tcp_contact[2] += self.dz_offset

        # Posición separada, todavía sin tocar objeto.
        self.tcp_ready = self.tcp_contact + self.approach_dir_world * self.approach_distance

        # Preagarre elevado en Z mundo, manteniendo la separación lateral.
        self.tcp_pregrasp = self.tcp_ready + np.array([0.0, 0.0, self.hover_height], dtype=float)

        self.get_logger().info(
            f'[Diagnóstico Poses TCP] Lado: {self.cfg.side.upper()} | '
            f'ee_link: {self.cfg.ee_link} | '
            f'approach_dir_world: {np.round(self.approach_dir_world, 4).tolist()} | '
            f'tcp_pregrasp: {np.round(self.tcp_pregrasp, 4).tolist()} | '
            f'tcp_ready: {np.round(self.tcp_ready, 4).tolist()} | '
            f'tcp_contact: {np.round(self.tcp_contact, 4).tolist()}'
        )

        self.get_logger().info(
            'Geometría de agarre: '
            f'radio={self.object_contact_radius():.4f}, clearance={self.surface_clearance:.4f}, '
            f'approach_dir_world={[round(v, 4) for v in self.approach_dir_world.tolist()]}'
        )
        self.get_logger().info(
            f'tcp_pregrasp={np.round(self.tcp_pregrasp, 4).tolist()} | '
            f'tcp_ready={np.round(self.tcp_ready, 4).tolist()} | '
            f'tcp_contact={np.round(self.tcp_contact, 4).tolist()}'
        )

        if self.publish_debug_markers:
            self.publicar_marker(self.tcp_pregrasp, 1, 'tcp_pregrasp', [0.1, 0.2, 1.0, 0.65], 0.030)
            self.publicar_marker(self.tcp_ready, 2, 'tcp_ready', [0.2, 0.9, 0.9, 0.65], 0.030)
            self.publicar_marker(self.tcp_contact, 3, 'tcp_contact', [1.0, 0.4, 0.0, 0.75], 0.030)
            self.publicar_marker(
                position=self.tcp_contact,
                marker_id=4,
                namespace='tcp_approach_arrow',
                rgba=[1.0, 0.0, 0.0, 0.9],
                marker_type=Marker.ARROW,
                quat=self.tcp_quat,
                scale_xyz=[0.06, 0.01, 0.01]
            )

    def calcular_poses_place_mesa(self) -> None:
        """
        Calcula las poses para la colocación del cubo:
        - place_object_center: el centro objetivo sobre la mesa (con el margen Z).
        - tcp_place: pose del TCP para colocarlo usando la orientación de colocación neutra (yaw = 0) y dz_offset del pick.
        - tcp_place_above: pose elevada para aproximación y retirada.
        """
        if self.task_mode == 'bimanual_transfer':
            if self.transfer_stage == 'source_to_table':
                self.place_object_center = self.calcular_destino_mesa()
                self.target_object_yaw_rad = math.radians(self.table_object_yaw_deg)
            else:  # table_to_target
                self.place_object_center = self.calcular_destino_final_espejado()
                if self.final_mirror_yaw:
                    self.target_object_yaw_rad = -self.initial_object_yaw_rad
                else:
                    self.target_object_yaw_rad = self.initial_object_yaw_rad
        else:
            # single_pick_place
            if self.place_object_z == -9999.0:
                z_val = self.object_position[2]
            else:
                z_val = self.place_object_z
            self.place_object_center = np.array([self.place_x, self.place_y, z_val + self.place_z_margin], dtype=float)
            self.target_object_yaw_rad = 0.0

        # Calcular orientación de colocación simétrica y neutra (con respecto al target_object_yaw_rad del cubo colocado)
        if self.cfg.side == 'left':
            pitch_deg = self.pitch_offset_left_deg
            yaw_offset_deg = self.yaw_offset_left_deg
        else:
            pitch_deg = self.pitch_offset_right_deg
            yaw_offset_deg = self.yaw_offset_right_deg

        yaw_final_place_deg = math.degrees(self.target_object_yaw_rad) + yaw_offset_deg

        self.get_logger().info(
            f'[Orientación TCP Place] Lado: {self.cfg.side.upper()} | '
            f'pitch_deg: {pitch_deg:.2f} | yaw_final_place_deg: {yaw_final_place_deg:.2f}'
        )

        self.tcp_quat_place = R.from_euler('xyz', [0.0, pitch_deg, yaw_final_place_deg], degrees=True).as_quat()

        # Calcular dirección de aproximación para colocación usando la nueva orientación
        r_tcp_place = R.from_quat(self.tcp_quat_place).as_matrix()
        approach_dir_world_place = r_tcp_place.dot(np.array([1.0, 0.0, 0.0], dtype=float))
        approach_dir_world_place = approach_dir_world_place / np.linalg.norm(approach_dir_world_place)

        # Tanto para el brazo izquierdo como derecho, el TCP se sitúa de forma
        # frontal (en la dirección del eje local +X de colocación) respecto al centro.
        contact_offset = self.object_contact_radius() + self.surface_clearance

        self.tcp_place = self.place_object_center + approach_dir_world_place * contact_offset
        self.tcp_place[2] += self.dz_offset

        # tcp_place_above = tcp_place + [0, 0, place_hover_height]
        self.tcp_place_above = self.tcp_place + np.array([0.0, 0.0, self.place_hover_height], dtype=float)

        # tcp_place_retreat = tcp_place + approach_dir_world_place * approach_distance (retirada lateral)
        self.tcp_place_retreat = self.tcp_place + approach_dir_world_place * self.approach_distance

        self.get_logger().info(
            'Poses de Place calculadas (Alineación Yaw Neutro): '
            f'place_object_center={np.round(self.place_object_center, 4).tolist()} | '
            f'tcp_place={np.round(self.tcp_place, 4).tolist()} | '
            f'tcp_place_above={np.round(self.tcp_place_above, 4).tolist()} | '
            f'tcp_place_retreat={np.round(self.tcp_place_retreat, 4).tolist()}'
        )

        if self.publish_debug_markers:
            self.publicar_marker(self.place_object_center, 10, 'place_object_center', [0.8, 0.1, 0.8, 0.70], self.object_dimension)
            self.publicar_marker(self.tcp_place, 11, 'tcp_place', [1.0, 0.0, 1.0, 0.75], 0.030)
            self.publicar_marker(self.tcp_place_above, 12, 'tcp_place_above', [0.5, 0.0, 0.5, 0.65], 0.030)
            self.publicar_marker(self.tcp_place_retreat, 14, 'tcp_place_retreat', [0.0, 1.0, 0.0, 0.65], 0.030)
            self.publicar_marker(
                position=self.tcp_place,
                marker_id=13,
                namespace='tcp_place_arrow',
                rgba=[1.0, 0.0, 1.0, 0.9],
                marker_type=Marker.ARROW,
                quat=self.tcp_quat_place,
                scale_xyz=[0.06, 0.01, 0.01]
            )

    # ======================================================================
    # IK y planificación
    # ======================================================================

    def get_ik(self, pos: np.ndarray, quat_xyzw: np.ndarray, avoid_collisions: bool = True):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.cfg.arm_group
        req.ik_request.ik_link_name = self.cfg.ee_link
        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.pose.position.x = float(pos[0])
        req.ik_request.pose_stamped.pose.position.y = float(pos[1])
        req.ik_request.pose_stamped.pose.position.z = float(pos[2])
        req.ik_request.pose_stamped.pose.orientation.x = float(quat_xyzw[0])
        req.ik_request.pose_stamped.pose.orientation.y = float(quat_xyzw[1])
        req.ik_request.pose_stamped.pose.orientation.z = float(quat_xyzw[2])
        req.ik_request.pose_stamped.pose.orientation.w = float(quat_xyzw[3])
        req.ik_request.avoid_collisions = bool(avoid_collisions)
        return self.ik_client.call(req)

    def filtrar_joints_brazo(self, ik_response) -> Tuple[List[str], List[float]]:
        joint_names = list(ik_response.solution.joint_state.name)
        joint_positions = list(ik_response.solution.joint_state.position)

        target_names, target_positions = [], []
        for name, pos in zip(joint_names, joint_positions):
            if name in self.cfg.arm_joints:
                target_names.append(name)
                target_positions.append(float(pos))

        if len(target_names) != len(self.cfg.arm_joints):
            self.get_logger().warn(
                f'IK devolvió {len(target_names)} joints del brazo; se esperaban {len(self.cfg.arm_joints)}.'
            )
        return target_names, target_positions

    def enviar_meta_articular_brazo(self, names: List[str], positions: List[float], phase: int) -> None:
        self.current_phase = phase
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Servidor MoveGroup no disponible.')
            rclpy.shutdown()
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.cfg.arm_group
        req.num_planning_attempts = 35
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.04
        req.max_acceleration_scaling_factor = 0.04

        constraints = Constraints()
        for name, pos in zip(names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)
        goal_msg.request = req

        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.move_group_goal_response_callback)

    def planificar_a_pose(self, pos: np.ndarray, quat: np.ndarray, phase: int, label: str) -> None:
        self.get_logger().info(f'Planificando fase {phase}: {label}')
        res_ik = self.get_ik(pos, quat, avoid_collisions=True)

        if res_ik.error_code.val != 1:
            self.get_logger().error(
                f'IK con colisiones falló en fase {phase} ({label}). Código: {res_ik.error_code.val}.'
            )

            usar_fallback_mujoco = (
                self.mujoco_bridge_mode
                and self.mujoco_disable_planning_collisions
                and phase in [7, 8, 70, 9, 90]
            )

            if usar_fallback_mujoco:
                self.get_logger().warn(
                    f'[MuJoCo bridge] Fallback IK sin colisiones habilitado para fase {phase} '
                    f'({label}). Esta ruta se usa solo para validación cinemática en MuJoCo.'
                )
                res_ik = self.get_ik(pos, quat, avoid_collisions=False)

                if res_ik.error_code.val != 1:
                    self.get_logger().error(
                        f'IK fallback sin colisiones también falló en fase {phase} ({label}). '
                        f'Código: {res_ik.error_code.val}.'
                    )
                    rclpy.shutdown()
                    return
            else:
                self.get_logger().error(
                    'No se ejecuta fallback sin colisiones para evitar atravesar el objeto.'
                )

                if self.diagnostic_ik_without_collisions:
                    diag = self.get_ik(pos, quat, avoid_collisions=False)
                    self.get_logger().warn(
                        f'Diagnóstico IK sin colisiones: código {diag.error_code.val}. '
                        'Si aquí funciona, el problema es geométrico/colisión, no cinemático.'
                    )

                rclpy.shutdown()
                return

        names, positions = self.filtrar_joints_brazo(res_ik)
        self.enviar_meta_articular_brazo(names, positions, phase=phase)

    def move_group_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Planificación fase {self.current_phase} rechazada.')
            rclpy.shutdown()
            return

        self.get_logger().info(f'Planificación fase {self.current_phase} aceptada. Ejecutando...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.move_group_result_callback)

    def move_group_result_callback(self, future) -> None:
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info(f'Fase {self.current_phase} completada correctamente.')

            if self.mujoco_bridge_mode and self.mujoco_wait_after_arm_motion_sec > 0.0:
                self.get_logger().info(
                    f'[MuJoCo bridge] Esperando {self.mujoco_wait_after_arm_motion_sec:.2f}s '
                    'para asentamiento dinámico antes de la siguiente fase.'
                )
                time.sleep(self.mujoco_wait_after_arm_motion_sec)

            if self.current_phase == 40:
                self.ejecutar_siguiente_subpaso_aproximacion()
                return

            if self.current_phase == 70:
                self.ejecutar_siguiente_subpaso_lift()
                return

            if self.current_phase == 90:
                self.ejecutar_siguiente_subpaso_place_descent()
                return

            if self.current_phase == 12 and self.use_split_place_retreat and not self.retreat_done:
                self.retreat_done = True
                self.ejecutar_fase_12_retirada_segura()
            else:
                self.ejecutar_siguiente_fase()
        else:
            self.get_logger().error(f'Error MoveIt en fase {self.current_phase}: {result.error_code.val}')
            rclpy.shutdown()

    # ======================================================================
    # Mano
    # ======================================================================

    def hand_open_positions(self) -> List[float]:
        return [0.0 for _ in self.cfg.hand_joints]

    def calcular_configuracion_mano(self) -> List[float]:
        d_open = 0.120
        d_closed = 0.012
        theta_finger_max = 1.45
        dimension = max(0.015, float(self.object_dimension))

        target_gap = max(d_closed, dimension - 2.0 * self.penetration_limit)
        theta = (d_open - target_gap) / (d_open - d_closed) * theta_finger_max
        theta = float(np.clip(theta, 0.0, theta_finger_max))

        # Cálculo manual/adaptativo parametrizado:
        # Permite cambiar dinámicamente las fracciones de flexión para ajustarse a esferas u otros objetos.
        # Por defecto, anular y meñique se fijan en 0.0 para no moverlos (agarrar solo con índice, medio y pulgar).
        index = theta * self.index_finger_fraction
        middle = theta * self.middle_finger_fraction
        ring = theta * self.ring_finger_fraction
        pinky = theta * self.pinky_finger_fraction

        # Modulación del pulgar (pitch y yaw) basada en las fracciones declaradas
        thumb_pitch = float(np.clip(0.40 * self.thumb_pitch_fraction + 0.18 * (theta / theta_finger_max), -0.1, 0.58)) if self.thumb_pitch_fraction > 0.0 else 0.0
        thumb_yaw = float(np.clip(0.85 * self.thumb_yaw_fraction + 0.35 * (theta / theta_finger_max), -0.1, 1.25)) if self.thumb_yaw_fraction > 0.0 else 0.0

        positions = [index, middle, pinky, ring, thumb_pitch, thumb_yaw]
        self.get_logger().info(
            f'Cierre mano (Pinch): index={index:.3f}, middle={middle:.3f}, ring={ring:.3f}, pinky={pinky:.3f}, '
            f'thumb_pitch={thumb_pitch:.3f}, thumb_yaw={thumb_yaw:.3f}'
        )
        return [float(p) for p in positions]

    def enviar_comando_mano(self, positions: List[float], phase: int) -> None:
        self.current_phase = phase

        if self.mujoco_bridge_mode:
            self.get_logger().info(
                f'[MuJoCo bridge] Fase {phase}: comando de mano omitido. '
                'Se continúa con validación cinemática del brazo.'
            )
            time.sleep(0.25)
            self.ejecutar_siguiente_fase()
            return

        if not self.hand_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'Controlador de mano no disponible: {self.cfg.hand_controller_action}')
            rclpy.shutdown()
            return

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.cfg.hand_joints

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=1, nanosec=200_000_000)

        trajectory.points.append(point)
        goal_msg.trajectory = trajectory

        future = self.hand_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.hand_goal_response_callback)

    def hand_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Comando de mano fase {self.current_phase} rechazado.')
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.hand_result_callback)

    def hand_result_callback(self, future) -> None:
        error_code = future.result().result.error_code
        if error_code == 0:
            self.get_logger().info(f'Movimiento de mano fase {self.current_phase} completado.')
            self.ejecutar_siguiente_fase()
        else:
            self.get_logger().error(f'Fallo en controlador de mano fase {self.current_phase}. Código: {error_code}')
            rclpy.shutdown()

    # ======================================================================
    # CIERRE TACTIL ACTIVO GRADUAL (MONITOREO INDIVIDUAL POR DEDO - MUJOCO)
    # ======================================================================

    def iniciar_tactile_grasp(self) -> None:
        self.get_logger().info('=== FASE 5: Iniciando Cierre Táctil Activo Gradual (MuJoCo) ===')
        
        # Calcular los límites de flexión máximos adaptativos para no machacar el objeto
        positions_max = self.calcular_configuracion_mano()
        self.target_index_max = positions_max[0]
        self.target_middle_max = positions_max[1]
        self.target_thumb_pitch_max = positions_max[4]
        self.target_thumb_yaw = positions_max[5]
        
        # Inicializar el estado de contacto de cada dedo que debe tocar
        self.finger_contact = {'index': False, 'middle': False, 'thumb': False}
        self.current_index_pos = 0.0
        self.current_middle_pos = 0.0
        self.current_thumb_pitch_pos = 0.0
        
        # Posicionar primero el pulgar en su ángulo de oposición (yaw) para encarar las caras
        self.enviar_comando_mano_tactico([0.0, 0.0, 0.0, 0.0, 0.0, self.target_thumb_yaw])
        time.sleep(0.4)

        # Guardar la línea base del torque (estática) después de posicionar el pulgar y que todo esté quieto
        self.pre_close_effort = self.snapshot_effort()
        if self.pre_close_effort:
            self.last_tick_effort = self.pre_close_effort.copy()
        else:
            self.last_tick_effort = None
            self.get_logger().warn('No se pudo inicializar last_tick_effort (snapshot_effort devolvió None)')

        self.grasp_step = 0.08  # Paso de flexión angular en cada tick [rad]
        
        # Lanzar el bucle táctil periódico a 5 Hz (cada 200 ms)
        self.grasp_timer = self.create_timer(0.20, self.tactile_grasp_tick, callback_group=self.cb_group)

    def tactile_grasp_tick(self) -> None:
        post = self.snapshot_effort()
        if not post or not self.pre_close_effort:
            self.get_logger().error('Pérdida de propiocepción en el bucle táctil.')
            self.grasp_timer.cancel()
            rclpy.shutdown()
            return
            
        p = self.cfg.link_prefix

        # Asegurar que last_tick_effort está inicializado para el primer ciclo
        if not self.last_tick_effort:
            self.last_tick_effort = self.pre_close_effort.copy()

        # 1. Dedo Índice
        if not self.finger_contact['index']:
            j = f'{p}index_proximal_joint'
            delta_base = abs(post.get(j, 0.0) - self.pre_close_effort.get(j, 0.0))
            delta_step = abs(post.get(j, 0.0) - self.last_tick_effort.get(j, 0.0))
            
            # Solo evaluar esfuerzo si el dedo ya pasó la flexión mínima
            if self.current_index_pos >= self.min_grasp_flexion and (delta_base > self.grasp_effort_threshold or delta_step > self.grasp_effort_threshold):
                self.finger_contact['index'] = True
                self.get_logger().info(f'[TACTIL] ¡Contacto en Dedo Índice! Pos={self.current_index_pos:.3f} rad, DeltaBase={delta_base:.4f} N*m, DeltaStep={delta_step:.4f} N*m')
            else:
                self.current_index_pos = min(self.target_index_max, self.current_index_pos + self.grasp_step)

        # 2. Dedo Medio
        if not self.finger_contact['middle']:
            j = f'{p}middle_proximal_joint'
            delta_base = abs(post.get(j, 0.0) - self.pre_close_effort.get(j, 0.0))
            delta_step = abs(post.get(j, 0.0) - self.last_tick_effort.get(j, 0.0))
            
            # Solo evaluar esfuerzo si el dedo ya pasó la flexión mínima
            if self.current_middle_pos >= self.min_grasp_flexion and (delta_base > self.grasp_effort_threshold or delta_step > self.grasp_effort_threshold):
                self.finger_contact['middle'] = True
                self.get_logger().info(f'[TACTIL] ¡Contacto en Dedo Medio! Pos={self.current_middle_pos:.3f} rad, DeltaBase={delta_base:.4f} N*m, DeltaStep={delta_step:.4f} N*m')
            else:
                self.current_middle_pos = min(self.target_middle_max, self.current_middle_pos + self.grasp_step)

        # 3. Dedo Pulgar (Clamping Pitch)
        if not self.finger_contact['thumb']:
            j = f'{p}thumb_proximal_pitch_joint'
            delta_base = abs(post.get(j, 0.0) - self.pre_close_effort.get(j, 0.0))
            delta_step = abs(post.get(j, 0.0) - self.last_tick_effort.get(j, 0.0))
            
            # Solo evaluar esfuerzo si el dedo ya pasó la flexión mínima
            if self.current_thumb_pitch_pos >= self.min_grasp_flexion and (delta_base > self.grasp_effort_threshold or delta_step > self.grasp_effort_threshold):
                self.finger_contact['thumb'] = True
                self.get_logger().info(f'[TACTIL] ¡Contacto en Pulgar! Pos={self.current_thumb_pitch_pos:.3f} rad, DeltaBase={delta_base:.4f} N*m, DeltaStep={delta_step:.4f} N*m')
            else:
                self.current_thumb_pitch_pos = min(self.target_thumb_pitch_max, self.current_thumb_pitch_pos + self.grasp_step)

        # Guardar el esfuerzo de este ciclo como referencia para el siguiente
        self.last_tick_effort = post.copy()

        # Publicar trayectorias tácticas incrementales
        positions = [
            self.current_index_pos,
            self.current_middle_pos,
            0.0,  # Anular inactivo (pinch)
            0.0,  # Meñique inactivo (pinch)
            self.current_thumb_pitch_pos,
            self.target_thumb_yaw
        ]
        self.enviar_comando_mano_tactico(positions)

        # Condiciones de parada: contacto establecido o límite físico alcanzado
        index_done = self.finger_contact['index'] or (self.current_index_pos >= self.target_index_max)
        middle_done = self.finger_contact['middle'] or (self.current_middle_pos >= self.target_middle_max)
        thumb_done = self.finger_contact['thumb'] or (self.current_thumb_pitch_pos >= self.target_thumb_pitch_max)

        if index_done and middle_done and thumb_done:
            self.get_logger().info('=== [TACTIL] Agarre exitoso. Todos los dedos activos en contacto o límite. ===')
            self.grasp_timer.cancel()
            self.current_phase = 5
            self.ejecutar_siguiente_fase()

    def enviar_comando_mano_tactico(self, positions: List[float]) -> None:
        """Comando de mano asíncrono y de baja latencia para el bucle cerrado táctil."""
        if self.mujoco_bridge_mode:
            return

        if not self.hand_action_client.wait_for_server(timeout_sec=1.0):
            return
        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.cfg.hand_joints
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=0, nanosec=150_000_000) # Reacción en 150ms
        trajectory.points.append(point)
        goal_msg.trajectory = trajectory
        self.hand_action_client.send_goal_async(goal_msg)

    # ======================================================================
    # Fases
    # ======================================================================

    def ejecutar_fase_1_preparar_escena_y_abrir_mano(self) -> None:
        self.get_logger().info('=== FASE 1: Preparar escena y abrir mano ===')
        self.limpiar_escena()

        if self.mujoco_bridge_mode and self.mujoco_disable_planning_collisions:
            self.get_logger().warn(
                '[MuJoCo bridge] Colisiones de mesa/objeto OMITIDAS en PlanningScene '
                'para aislar validación MoveIt→MuJoCo.'
            )
        else:
            if self.use_table_collision:
                self.añadir_mesa_colision()
            self.añadir_objeto_colision_mundo()

        self.enviar_comando_mano(self.hand_open_positions(), phase=1)

    def ejecutar_fase_2_preagarre_elevado(self) -> None:
        self.get_logger().info('=== FASE 2: Preagarre elevado ===')
        self.calcular_poses_tcp()
        self.planificar_a_pose(self.tcp_pregrasp, self.tcp_quat, phase=2, label='preagarre elevado')

    def ejecutar_fase_3_descenso_vertical(self) -> None:
        self.get_logger().info('=== FASE 3: Descenso vertical manteniendo separación lateral ===')
        self.planificar_a_pose(self.tcp_ready, self.tcp_quat, phase=3, label='descenso vertical')

    def ejecutar_fase_4_aproximacion_frontal(self) -> None:
        self.get_logger().info('=== FASE 4: Aproximación frontal hacia cara del objeto ===')

        if self.mujoco_bridge_mode and self.mujoco_split_approach:
            steps = max(1, int(self.mujoco_approach_steps))
            self.approach_substep_index = 0
            self.approach_substep_targets = []

            start = np.array(self.tcp_ready, dtype=float)
            end = np.array(self.tcp_contact, dtype=float)

            for i in range(1, steps + 1):
                alpha = i / steps
                target = (1.0 - alpha) * start + alpha * end
                self.approach_substep_targets.append(target)

            self.get_logger().info(
                f'[MuJoCo bridge] Aproximación segmentada activada: {steps} pasos. '
                f'Distancia total={np.linalg.norm(end - start):.4f} m'
            )

            self.ejecutar_siguiente_subpaso_aproximacion()
            return

        self.planificar_a_pose(self.tcp_contact, self.tcp_quat, phase=4, label='aproximación a cara del objeto')

    def ejecutar_siguiente_subpaso_aproximacion(self) -> None:
        if self.approach_substep_index >= len(self.approach_substep_targets):
            self.get_logger().info('[MuJoCo bridge] Aproximación segmentada completada.')
            self.current_phase = 4
            self.ejecutar_siguiente_fase()
            return

        target = self.approach_substep_targets[self.approach_substep_index]
        self.approach_substep_index += 1

        self.get_logger().info(
            f'[MuJoCo bridge] Aproximación subpaso '
            f'{self.approach_substep_index}/{len(self.approach_substep_targets)}: '
            f'{np.round(target, 4).tolist()}'
        )

        self.planificar_a_pose(
            target,
            self.tcp_quat,
            phase=40,
            label=f'aproximación frontal subpaso {self.approach_substep_index}'
        )

    def ejecutar_fase_5_cierre_adaptativo(self) -> None:
        self.get_logger().info('=== FASE 5: Iniciando cierre de mano ===')
        if self.effort_available:
            self.iniciar_tactile_grasp()
        else:
            self.get_logger().info('Sin propiocepción disponible en este entorno; ejecutando cierre cinemático.')
            self.enviar_comando_mano(self.calcular_configuracion_mano(), phase=5)

    def ejecutar_fase_6_validar_y_adjuntar(self) -> None:
        self.get_logger().info('=== FASE 6: Validar agarre y adjuntar objeto ===')
        ok, msg = self.evaluar_agarre_propioceptivo()
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)
            rclpy.shutdown()
            return

        if self.mujoco_bridge_mode and self.mujoco_disable_planning_collisions:
            self.get_logger().warn(
                '[MuJoCo bridge] Attach real en PlanningScene omitido. '
                'Se conserva solo validación cinemática para fases posteriores.'
            )
        else:
            self.adjuntar_objeto_al_tcp()

        self.current_phase = 6
        self.ejecutar_siguiente_fase()

    def ejecutar_fase_7_retirada_vertical(self) -> None:
        self.get_logger().info('=== FASE 7: Retirada vertical con objeto adjunto ===')

        start = np.array(self.tcp_contact, dtype=float)
        target = np.array(self.tcp_contact, dtype=float)
        target[2] += self.lift_distance

        if self.mujoco_bridge_mode and self.mujoco_lift_steps > 1:
            self.lift_substep_index = 0
            self.lift_substep_targets = []

            for i in range(1, self.mujoco_lift_steps + 1):
                alpha = i / float(self.mujoco_lift_steps)
                subtarget = start + alpha * (target - start)
                self.lift_substep_targets.append(subtarget)

            self.get_logger().info(
                f'[MuJoCo bridge] Retirada vertical segmentada activada: '
                f'{self.mujoco_lift_steps} pasos. Altura total={self.lift_distance:.4f} m'
            )

            self.ejecutar_siguiente_subpaso_lift()
            return

        self.planificar_a_pose(target, self.tcp_quat, phase=7, label='retirada vertical')

    def ejecutar_siguiente_subpaso_lift(self) -> None:
        if self.lift_substep_index >= len(self.lift_substep_targets):
            self.get_logger().info('[MuJoCo bridge] Retirada vertical segmentada completada.')
            self.current_phase = 7
            self.ejecutar_siguiente_fase()
            return

        target = self.lift_substep_targets[self.lift_substep_index]
        self.lift_substep_index += 1

        self.get_logger().info(
            f'[MuJoCo bridge] Retirada vertical subpaso '
            f'{self.lift_substep_index}/{len(self.lift_substep_targets)}: {target.tolist()}'
        )

        self.current_phase = 70
        self.planificar_a_pose(
            target,
            self.tcp_quat,
            phase=70,
            label=f'retirada vertical subpaso {self.lift_substep_index}'
        )

    def ejecutar_fase_8_traslado_sobre_mesa(self) -> None:
        self.get_logger().info('=== FASE 8: Traslado sobre la mesa ===')
        self.calcular_poses_place_mesa()
        self.get_logger().info(f'Pose TCP sobre mesa (pre-place): {self.tcp_place_above.tolist()}')
        self.planificar_a_pose(self.tcp_place_above, self.tcp_quat_place, phase=8, label='traslado sobre mesa')

    def ejecutar_fase_9_descenso_colocacion(self) -> None:
        self.get_logger().info('=== FASE 9: Descenso a pose de colocación ===')
        self.get_logger().info(f'Pose TCP de colocación (place): {self.tcp_place.tolist()}')

        start = np.array(self.tcp_place_above, dtype=float)
        target = np.array(self.tcp_place, dtype=float)

        if self.mujoco_bridge_mode and self.mujoco_place_descent_steps > 1:
            self.place_descent_substep_index = 0
            self.place_descent_substep_targets = []

            for i in range(1, self.mujoco_place_descent_steps + 1):
                alpha = i / float(self.mujoco_place_descent_steps)
                subtarget = start + alpha * (target - start)
                self.place_descent_substep_targets.append(subtarget)

            descent_distance = abs(float(start[2] - target[2]))
            self.get_logger().info(
                f'[MuJoCo bridge] Descenso de colocación segmentado activado: '
                f'{self.mujoco_place_descent_steps} pasos. Descenso total={descent_distance:.4f} m'
            )

            self.ejecutar_siguiente_subpaso_place_descent()
            return

        self.planificar_a_pose(self.tcp_place, self.tcp_quat_place, phase=9, label='descenso a colocación')

    def ejecutar_siguiente_subpaso_place_descent(self) -> None:
        if self.place_descent_substep_index >= len(self.place_descent_substep_targets):
            self.get_logger().info('[MuJoCo bridge] Descenso de colocación segmentado completado.')
            self.current_phase = 9
            self.ejecutar_siguiente_fase()
            return

        target = self.place_descent_substep_targets[self.place_descent_substep_index]
        self.place_descent_substep_index += 1

        self.get_logger().info(
            f'[MuJoCo bridge] Descenso de colocación subpaso '
            f'{self.place_descent_substep_index}/{len(self.place_descent_substep_targets)}: {target.tolist()}'
        )

        self.current_phase = 90
        self.planificar_a_pose(
            target,
            self.tcp_quat_place,
            phase=90,
            label=f'descenso de colocación subpaso {self.place_descent_substep_index}'
        )

    def ejecutar_fase_10_apertura_mano(self) -> None:
        self.get_logger().info('=== FASE 10: Apertura de mano para liberación ===')
        self.enviar_comando_mano(self.hand_open_positions(), phase=10)

    def ejecutar_fase_11_desadjuntar_y_registrar(self) -> None:
        self.get_logger().info('=== FASE 11: Desadjuntar y registrar objeto en destino/mesa ===')
        self.registrar_objeto_en_destino()
        if self.mujoco_bridge_mode and self.mujoco_disable_planning_collisions:
            self.get_logger().warn(
                '[MuJoCo bridge] PlanningScene limpiada después del registro lógico '
                'del objeto para no bloquear la retirada segura.'
            )
            self.limpiar_escena()

        self.current_phase = 11
        self.ejecutar_siguiente_fase()

    def ejecutar_retirada_cartesiana_local_desde_place(self) -> None:
        """
        Placeholder para la futura implementación de la retirada cartesiana local.
        
        Comentarios/Indicaciones de diseño:
        - Se usará luego para retirar la mano desde tcp_place hacia tcp_place_retreat en línea recta.
        - Será útil para movimientos cercanos al cubo y mesa.
        - Debe usar compute_cartesian_path o una interfaz equivalente de MoveIt (ej. invocar el servicio de planificación cartesiana).
        - No debe usarse para traslados largos.
        """
        self.get_logger().warn(
            "Movimiento cartesiano local aún no implementado; usando retirada por pose/home."
        )

    def ejecutar_fase_12_retirada_segura(self) -> None:
        self.get_logger().info('=== FASE 12: Retirada segura del brazo hacia posición de reposo ===')
        
        if self.current_phase != 12:
            dist = float(np.linalg.norm(self.tcp_place - self.tcp_place_retreat))
            self.get_logger().info('--- Diagnóstico de Fase 12 ---')
            self.get_logger().info(f'task_mode: {self.task_mode}')
            self.get_logger().info(f'transfer_stage: {self.transfer_stage}')
            self.get_logger().info(f'cfg.side: {self.cfg.side}')
            self.get_logger().info(f'object_position: {self.object_position.tolist()}')
            self.get_logger().info(f'object_yaw_rad: {self.object_yaw_rad:.4f} ({math.degrees(self.object_yaw_rad):.2f}°)')
            self.get_logger().info(f'tcp_place: {self.tcp_place.tolist()}')
            self.get_logger().info(f'tcp_place_retreat: {self.tcp_place_retreat.tolist()}')
            self.get_logger().info(f'distancia tcp_place a tcp_place_retreat: {dist:.4f} m')
            self.get_logger().info(f'use_split_place_retreat: {self.use_split_place_retreat}')
            self.get_logger().info(f'use_cartesian_local_motions: {self.use_cartesian_local_motions}')
            self.get_logger().info('------------------------------')
            
        if self.use_cartesian_local_motions:
            self.ejecutar_retirada_cartesiana_local_desde_place()
            
        if self.use_split_place_retreat:
            if not self.retreat_done:
                # Primero, retiramos lateralmente hacia afuera
                self.get_logger().info('Fase 12 - Parte 1: Retirada lateral hacia afuera (alejándose del objeto)...')
                self.planificar_a_pose(self.tcp_place_retreat, self.tcp_quat_place, phase=12, label='retirada lateral de colocación')
            else:
                # Segundo, vamos a home
                self.get_logger().info('Fase 12 - Parte 2: Moviendo a posición de reposo neutral...')
                names = self.cfg.arm_joints
                positions = [0.0] * len(self.cfg.arm_joints)
                self.get_logger().info(f'Planificando articulaciones hacia pose de reposo neutral ("home_{self.cfg.side}")...')
                self.enviar_meta_articular_brazo(names, positions, phase=12)
        else:
            # Ir directamente a home
            self.get_logger().info('Fase 12: Moviendo directamente a posición de reposo neutral (bypasseando retirada lateral)...')
            names = self.cfg.arm_joints
            positions = [0.0] * len(self.cfg.arm_joints)
            self.get_logger().info(f'Planificando articulaciones hacia pose de reposo neutral ("home_{self.cfg.side}")...')
            self.enviar_meta_articular_brazo(names, positions, phase=12)

    def ejecutar_siguiente_fase(self) -> None:
        if self.stop_after_phase > 0 and self.current_phase >= self.stop_after_phase:
            self.get_logger().info(
                f'[MuJoCo bridge] stop_after_phase={self.stop_after_phase} alcanzado. '
                'Secuencia detenida correctamente para validación.'
            )
            rclpy.shutdown()
            return

        if self.current_phase == 1:
            self.ejecutar_fase_2_preagarre_elevado()
        elif self.current_phase == 2:
            self.ejecutar_fase_3_descenso_vertical()
        elif self.current_phase == 3:
            self.ejecutar_fase_4_aproximacion_frontal()
        elif self.current_phase == 4:
            self.ejecutar_fase_5_cierre_adaptativo()
        elif self.current_phase == 5:
            self.ejecutar_fase_6_validar_y_adjuntar()
        elif self.current_phase == 6:
            self.ejecutar_fase_7_retirada_vertical()
        elif self.current_phase == 7:
            self.ejecutar_fase_8_traslado_sobre_mesa()
        elif self.current_phase == 8:
            self.ejecutar_fase_9_descenso_colocacion()
        elif self.current_phase == 9:
            self.ejecutar_fase_10_apertura_mano()
        elif self.current_phase == 10:
            self.ejecutar_fase_11_desadjuntar_y_registrar()
        elif self.current_phase == 11:
            self.ejecutar_fase_12_retirada_segura()
        elif self.current_phase == 12:
            if self.task_mode == 'bimanual_transfer' and self.transfer_stage == 'source_to_table':
                self.get_logger().info('Primera etapa completada: objeto colocado en mesa central.')
                self.get_logger().info('Cambiando al brazo opuesto para tomar objeto desde mesa.')
                self.reiniciar_ciclo_para_siguiente_etapa()
            elif self.task_mode == 'bimanual_transfer' and self.transfer_stage == 'table_to_target':
                self.get_logger().info(
                    'Transferencia bimanual completada: objeto movido desde lado inicial hacia lado opuesto mediante mesa central.'
                )
                self.get_logger().info(f'Pose final esperada: {self.object_position.tolist()} | Yaw final esperado: {math.degrees(self.object_yaw_rad):.2f}°')
                self.get_logger().info('Confirmación de objeto registrado en mundo.')
                rclpy.shutdown()
            else:
                self.get_logger().info('Pick and place parcial completado: objeto colocado en mesa y registrado en escena.')
                rclpy.shutdown()
        else:
            self.get_logger().error(f'Fase desconocida: {self.current_phase}')
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SingleArmFaceApproachGraspNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
