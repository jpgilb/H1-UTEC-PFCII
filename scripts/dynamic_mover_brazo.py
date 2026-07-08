#!/usr/bin/env python3
"""
Fase 1: Acercamiento Pre-Grasp Dinámico (Arquitectura Relativa al Objeto).
Ingeniería de Coordinación Bimanual para Unitree H1-2.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
import threading
import time
import numpy as np
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    CollisionObject
)
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R


class DynamicMoveItClient(Node):
    def __init__(self):
        super().__init__('dynamic_grasp_planner')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Clientes Action y Service ---
        self._mg_client = ActionClient(self, MoveGroup, 'move_action')
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self._left_ctrl = ActionClient(self, FollowJointTrajectory, '/left_arm_controller/follow_joint_trajectory')
        self._right_ctrl = ActionClient(self, FollowJointTrajectory, '/right_arm_controller/follow_joint_trajectory')

        self.marcador_pub = self.create_publisher(MarkerArray, 'meta_visual_robot_array', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.current_joint_state = None
        self.create_subscription(JointState, 'joint_states', self.joint_cb, 10)

        self.base_frame = 'pelvis'
        self.target_frame = 'objeto_cubo'
        self.arm_group = 'both_arms'

        # =====================================================================
        # PARÁMETROS DE PRE-GRASP (Ajuste empírico)
        # Todos los offsets se aplican en el frame LOCAL del objeto (r_c.apply).
        # =====================================================================
        # Offset lateral (Eje Y local del cubo): Separación a cada lado del cubo.
        # Aumentado a 22cm para evitar colisión entre manos (H1-2 mano ~10cm de ancho).
        self.offset_lateral = 0.22   # [m] 22 cm a cada lado del cubo

        # Offset frontal (Eje X local del cubo): Positivo = delante del cubo.
        # Acercamos el target hacia el robot para que el brazo se extienda frontalmente.
        # Valor negativo = detrás del cubo (causaba pose de "abrazar").
        self.offset_frontal = 0.00   # [m] 0 = mismo plano frontal que el cubo

        # Offset de altura (Z global): Relativo al centroide del cubo.
        # Ligeramente por debajo del centro para agarre lateral al nivel de la mesa.
        self.offset_altura = -0.02   # [m] 2 cm por debajo del centro del cubo

        # -----------------------------------------------------------------------
        # CORRECCIÓN DE CÁMARA INVERTIDA
        # Si la cámara está físicamente invertida, la Z del cubo detectada será
        # incorrecta (aparecerá a la altura del pecho en lugar de la mesa).
        # Activa 'usar_z_fija = True' y ajusta 'z_mesa_fija' a la altura real
        # del cubo respecto al frame 'pelvis' (aprox. -0.2 a 0.0 m para una mesa
        # que llega a la cadera del robot).
        self.usar_z_fija = True      # True mientras la cámara esté invertida
        self.z_mesa_fija = 0.25      # [m] altura del cubo en frame pelvis (ajustar a la mesa real)
        # -----------------------------------------------------------------------
        # =====================================================================

        self.movimiento_iniciado = False
        self.timer = self.create_timer(1.0, self.buscar_objeto_y_planear)

        self.get_logger().info("H1-2: Iniciando con Planeación Dinámica Relativa al Objeto.")

    def joint_cb(self, msg):
        self.current_joint_state = msg

    def generar_escena_inicial(self):
        mesa = CollisionObject()
        mesa.header.frame_id = self.base_frame
        mesa.header.stamp = self.get_clock().now().to_msg()
        mesa.id = 'mesa_trabajo'
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.6, 1.2, 0.05]
        pose = Pose()
        pose.position.x = 0.45 
        pose.position.y = 0.0
        pose.position.z = -0.15
        mesa.primitives.append(box)
        mesa.primitive_poses.append(pose)
        mesa.operation = CollisionObject.ADD
        self.collision_pub.publish(mesa)

    def dibujar_metas_dinamicas(self, p_izq, p_der, t_cubo):
        """
        Dibuja flechas SIMÉTRICAS que apuntan horizontalmente desde cada mano
        hacia el centro del cubo. Las flechas son espejo (Z normals opuestos),
        lo que visualmente representa la condición de 'palmas enfrentadas'.
        """
        marcador_array = MarkerArray()
        t_cubo_np = np.array(t_cubo)

        for idx, (pos, color) in enumerate([
            (p_izq, (0.0, 1.0, 1.0, 0.9)),   # Cian = Izquierdo
            (p_der, (1.0, 0.0, 1.0, 0.9)),   # Magenta = Derecho
        ]):
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'meta_dinamica'
            m.id = idx
            m.type = Marker.ARROW
            m.action = Marker.ADD

            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])

            # Dirección horizontal proyectada (sin componente Z).
            # Así ambas flechas son simétricas en el plano XY -> espejo visual.
            dir_vec = t_cubo_np - np.array(pos)
            dir_vec[2] = 0.0  # Proyección horizontal pura
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                dir_vec /= norm
                x_axis = np.array([1.0, 0.0, 0.0])
                cross = np.cross(x_axis, dir_vec)
                dot = np.dot(x_axis, dir_vec)
                cross_norm = np.linalg.norm(cross)
                if cross_norm > 1e-6:
                    q_orient = R.from_rotvec(
                        np.arccos(np.clip(dot, -1, 1)) * cross / cross_norm
                    ).as_quat()
                else:
                    q_orient = [0, 0, 0, 1] if dot > 0 else [0, 1, 0, 0]
            else:
                q_orient = [0, 0, 0, 1]

            m.pose.orientation.x = float(q_orient[0])
            m.pose.orientation.y = float(q_orient[1])
            m.pose.orientation.z = float(q_orient[2])
            m.pose.orientation.w = float(q_orient[3])

            m.scale.x, m.scale.y, m.scale.z = 0.12, 0.025, 0.025
            m.color.r, m.color.g, m.color.b, m.color.a = color
            marcador_array.markers.append(m)

        self.marcador_pub.publish(marcador_array)

    def buscar_objeto_y_planear(self):
        """
        Fase 1: Cálculo de Pre-Grasp Dinámico Relativo al Objeto.
        """
        self.generar_escena_inicial()
        if self.movimiento_iniciado or self.current_joint_state is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))

            t_c = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            q_c = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]

            r_c = R.from_quat(q_c)

            # --- CORRECCIÓN DE CÁMARA INVERTIDA ---
            # La cámara invertida corrompe TANTO la Z como la orientación del cubo.
            # Cuando usar_z_fija=True, usamos ejes globales puros para que el
            # pre-grasp sea siempre lateral en Y y frontal en X, sin importar
            # qué rotación reporta el TF del cubo.
            if self.usar_z_fija:
                t_c[2] = self.z_mesa_fija
                # Ejes globales: la mano izquierda va a +Y, la derecha a -Y
                vector_y_cubo = np.array([0.0, 1.0, 0.0])   # Lateral global
                vector_x_cubo = np.array([1.0, 0.0, 0.0])   # Frontal global
                self.get_logger().info(
                    f"Modo Z-Fija: cubo XY=({t_c[0]:.2f},{t_c[1]:.2f}) Z={t_c[2]:.2f} | Ejes GLOBALES")
            else:
                # Ejes del cubo reales (cuando la cámara está correcta)
                vector_x_cubo = r_c.apply([1, 0, 0])
                vector_y_cubo = r_c.apply([0, 1, 0])
                self.get_logger().info(
                    f"Modo dinámico: vec_Y_cubo=({vector_y_cubo[0]:.2f},{vector_y_cubo[1]:.2f},{vector_y_cubo[2]:.2f})")

            # Posiciones de pre-grasp con los 3 offsets parametrizados
            offset_global_z = np.array([0, 0, self.offset_altura])
            pos_izq = (t_c
                       + (vector_y_cubo * self.offset_lateral)
                       + (vector_x_cubo * self.offset_frontal)
                       + offset_global_z)
            pos_der = (t_c
                       - (vector_y_cubo * self.offset_lateral)
                       + (vector_x_cubo * self.offset_frontal)
                       + offset_global_z)

            self.get_logger().info(
                f"Metas — Izq: ({pos_izq[0]:.2f},{pos_izq[1]:.2f},{pos_izq[2]:.2f}) "
                f"Der: ({pos_der[0]:.2f},{pos_der[1]:.2f},{pos_der[2]:.2f})")

            # --- CALIBRACIÓN DE ORIENTACIÓN ---
            r_align_izq = R.from_euler('xyz', [90, 0, 90], degrees=True)
            r_align_der = R.from_euler('xyz', [-90, 0, 90], degrees=True)
            # ----------------------------------

            # Orientación visual (candidato A) solo para las flechas de RViz
            q_izq_visual = (r_c * r_align_izq).as_quat()
            q_der_visual = (r_c * r_align_der).as_quat()

            self.get_logger().info("Posición OK. Lanzando búsqueda sistemática de 4 configs × 4 rolls...")
            self.movimiento_iniciado = True
            self.timer.cancel()

            # Marcador visual provisional (Config 1: eje Y)
            pos_izq_vis = t_c + np.array([0.0, 1.0, 0.0]) * self.offset_lateral + np.array([0, 0, self.offset_altura])
            pos_der_vis = t_c - np.array([0.0, 1.0, 0.0]) * self.offset_lateral + np.array([0, 0, self.offset_altura])
            self.dibujar_metas_dinamicas(pos_izq_vis, pos_der_vis, t_c)

            # El hilo recibe t_c y q_c para calcular posiciones y orientaciones internamente
            threading.Thread(
                target=self.plan_bimanual_thread_dinamico,
                args=(t_c.copy(), q_c)
            ).start()

        except Exception:
            pass

    def wrist_orient_from_approach(self, approach_dir, finger_roll_deg=0):
        """
        Calcula el cuaternión de orientación del wrist tal que:
          - X_hand (normal de palma) apunte en dirección 'approach_dir'
          - El dedo se puede girar 'finger_roll_deg' grados alrededor del eje de aproximación.
        """
        d = np.array(approach_dir, dtype=float)
        d /= np.linalg.norm(d)
        x_axis = np.array([1.0, 0.0, 0.0])
        cross = np.cross(x_axis, d)
        dot   = float(np.dot(x_axis, d))
        cross_norm = np.linalg.norm(cross)

        if cross_norm > 1e-6:
            r_base = R.from_rotvec(np.arccos(np.clip(dot, -1.0, 1.0)) * cross / cross_norm)
        elif dot > 0:
            r_base = R.identity()
        else:
            r_base = R.from_euler('y', 180, degrees=True)

        # Giro adicional alrededor del eje de aproximación (orienta los dedos)
        r_roll = R.from_rotvec(np.deg2rad(finger_roll_deg) * d)
        return (r_roll * r_base).as_quat()

    def plan_bimanual_thread_dinamico(self, t_c, r_c_quat):
        """
        Prueba sistemáticamente las 4 configuraciones de pre-grasp bimanual.
        SANITIZACIÓN: Extrae solo el Yaw del cubo para garantizar simetría horizontal.
        """
        r_c_raw = R.from_quat(r_c_quat)
        
        # Extraemos solo el Yaw (giro sobre la mesa)
        # Ignoramos Pitch/Roll causados por la cámara invertida o ruido.
        euler = r_c_raw.as_euler('xyz', degrees=True)
        yaw_clean = euler[2]
        r_c_clean = R.from_euler('z', yaw_clean, degrees=True)

        # Calculamos los ejes locales del cubo usando solo el Yaw sanitizado
        # eje_x: apunta hacia adelante del cubo
        # eje_y: apunta hacia el lateral del cubo
        eje_x = r_c_clean.apply([1.0, 0.0, 0.0])
        eje_y = r_c_clean.apply([0.0, 1.0, 0.0])
        
        # Garantía extra de horizontalidad (aunque r_c_clean ya lo es)
        eje_x[2] = 0.0
        eje_y[2] = 0.0
        eje_x /= np.linalg.norm(eje_x)
        eje_y /= np.linalg.norm(eje_y)

        # ================================================================
        # 4 CONFIGURACIONES DE POSICIÓN
        # ================================================================
        configs_posicion = [
            ("Y+ Izq / Y- Der", eje_y,  +1),  # Lateral (caras Y)
            ("Y- Izq / Y+ Der", eje_y,  -1),  # Lateral intercambiado
            ("X+ Izq / X- Der", eje_x,  +1),  # Frontal (caras X)
            ("X- Izq / X+ Der", eje_x,  -1),  # Frontal intercambiado
        ]

        # Giros de dedos alrededor del eje de aproximación a probar
        finger_rolls = [0, 90, -90, 180]

        t_c_np = np.array(t_c)
        offset_z = np.array([0.0, 0.0, self.offset_altura])

        for config_nombre, eje, signo in configs_posicion:
            pos_izq = t_c_np + eje * signo * self.offset_lateral + offset_z
            pos_der = t_c_np - eje * signo * self.offset_lateral + offset_z

            # Dirección del wrist: X_hand apunta al DORSO. 
            # Para que PALMA mire al cubo, X_hand apunta LEJOS.
            dir_izq = eje * signo
            dir_der = -eje * signo  # Apunta desde t_c hacia pos_der (AFUERA del cubo)

            self.get_logger().info(f"Probando config: {config_nombre}")

            # --- Búsqueda INDEPENDIENTE por brazo ---
            # Seed para brazo izquierdo: codo recto y hombro extendido lateralmente.
            seed_izq = {
                'left_shoulder_pitch_joint':  0.0,
                'left_shoulder_roll_joint':   1.2,   # ~69° — brazo extendido a +Y
                'left_shoulder_yaw_joint':    0.0,   # Sin torsión
                'left_elbow_joint':           0.0,   # Codo recto
            }
            # Seed para brazo derecho: espejo del izquierdo.
            # Si el brazo se tuerce, incrementa shoulder_roll (más negativo = más extendido).
            seed_der = {
                'right_shoulder_pitch_joint':  0.0,
                'right_shoulder_roll_joint':  -1.5,  # ~86° — fuerza extensión lateral -Y
                'right_shoulder_yaw_joint':    0.0,  # Sin torsión — evita giro interno
                'right_elbow_joint':           0.0,  # Codo recto
            }

            sol_izq = None
            for roll in finger_rolls:
                q = self.wrist_orient_from_approach(dir_izq, roll)
                ni, pi = self.get_ik('left_arm', 'L_hand_base_link', pos_izq, q, seed_izq)
                if ni:
                    sol_izq = (ni, pi, roll)
                    self.get_logger().info(f"  Izq OK con Roll={roll}°")
                    break

            sol_der = None
            for roll in finger_rolls:
                q = self.wrist_orient_from_approach(dir_der, roll)
                nd, pd = self.get_ik('right_arm', 'R_hand_base_link', pos_der, q, seed_der)
                if nd:
                    sol_der = (nd, pd, roll)
                    self.get_logger().info(f"  Der OK con Roll={roll}°")
                    break

            if sol_izq and sol_der:
                self.get_logger().info(
                    f"✓ IK BIMANUAL OK: {config_nombre} | "
                    f"Roll Izq={sol_izq[2]}° Der={sol_der[2]}° | "
                    f"Pos Izq:({pos_izq[0]:.2f},{pos_izq[1]:.2f},{pos_izq[2]:.2f}) "
                    f"Der:({pos_der[0]:.2f},{pos_der[1]:.2f},{pos_der[2]:.2f})")
                self.dibujar_metas_dinamicas(pos_izq, pos_der, t_c)
                self.planear_y_ejecutar(sol_izq[0] + sol_der[0], sol_izq[1] + sol_der[1])
                return
            else:
                self.get_logger().warn(
                    f"  ✗ {config_nombre}: "
                    f"Izq={'OK' if sol_izq else 'X'}, Der={'OK' if sol_der else 'X'}")

        self.get_logger().error("Todas las configuraciones fallaron. Reintentando...")
        self.movimiento_iniciado = False
        self.timer = self.create_timer(1.0, self.buscar_objeto_y_planear)

    def get_ik(self, group, link, pos, quat, seed_overrides=None):
        """
        Llama al servicio compute_ik.
        seed_overrides: dict {joint_name: value_rad} para sesgar el solver
        hacia una configuración preferida (ej. codo recto).
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = link
        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.pose.position.x, req.ik_request.pose_stamped.pose.position.y, req.ik_request.pose_stamped.pose.position.z = pos
        req.ik_request.pose_stamped.pose.orientation.x, req.ik_request.pose_stamped.pose.orientation.y, req.ik_request.pose_stamped.pose.orientation.z, req.ik_request.pose_stamped.pose.orientation.w = quat
        req.ik_request.avoid_collisions = True

        # Construir el estado semilla: copia del estado actual + overrides opcionales
        from sensor_msgs.msg import JointState as JS
        import copy
        seed = copy.deepcopy(self.current_joint_state)
        if seed_overrides:
            for jname, jval in seed_overrides.items():
                if jname in seed.name:
                    idx = seed.name.index(jname)
                    seed.position = list(seed.position)
                    seed.position[idx] = jval
        req.ik_request.robot_state.joint_state = seed
        
        future = self.ik_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        res = future.result()
        if res and res.error_code.val == 1:
            prefix = group.split('_')[0]
            names = [f"{prefix}_{j}" for j in ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]]
            positions = [res.solution.joint_state.position[res.solution.joint_state.name.index(n)] for n in names]
            return names, positions
        return None, None

    def planear_y_ejecutar(self, joint_names, joint_positions):
        self._mg_client.wait_for_server()
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.arm_group
        req.num_planning_attempts = 30
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.2
        
        constraints = Constraints()
        for name, pos in zip(joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            # Tolerancia relajada: 0.08 rad (~4.6°) da más margen al planificador
            # para encontrar una trayectoria libre de colisiones sin ser muy restrictivo.
            jc.tolerance_above = 0.08
            jc.tolerance_below = 0.08
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)
        goal_msg.request = req
        goal_msg.planning_options.plan_only = True
        
        self._mg_client.send_goal_async(goal_msg).add_done_callback(self.cb_plan_result)

    def cb_plan_result(self, future):
        gh = future.result()
        if not gh.accepted: return
        gh.get_result_async().add_done_callback(self.cb_traj_ready)

    def cb_traj_ready(self, future):
        res = future.result().result
        if res.error_code.val != 1: return
            
        names = res.planned_trajectory.joint_trajectory.joint_names
        points = res.planned_trajectory.joint_trajectory.points
        
        idx_izq = [i for i, name in enumerate(names) if 'left_' in name]
        idx_der = [i for i, name in enumerate(names) if 'right_' in name]
        
        traj_izq, traj_der = JointTrajectory(), JointTrajectory()
        traj_izq.joint_names = [names[i] for i in idx_izq]
        traj_der.joint_names = [names[i] for i in idx_der]

        for p in points:
            p_izq, p_der = JointTrajectoryPoint(), JointTrajectoryPoint()
            p_izq.time_from_start, p_der.time_from_start = p.time_from_start, p.time_from_start
            p_izq.positions = [p.positions[i] for i in idx_izq]
            p_der.positions = [p.positions[i] for i in idx_der]
            traj_izq.points.append(p_izq)
            traj_der.points.append(p_der)

        self._left_ctrl.wait_for_server()
        self._right_ctrl.wait_for_server()
        self._left_ctrl.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj_izq))
        self._right_ctrl.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj_der))

def main(args=None):
    rclpy.init(args=args)
    nodo = DynamicMoveItClient()
    rclpy.spin(nodo)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
