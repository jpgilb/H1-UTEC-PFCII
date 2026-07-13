#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import sys
import time
import math
import tf2_ros
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    MotionPlanRequest, 
    Constraints, 
    JointConstraint,
    PositionIKRequest,
    CollisionObject,
    AttachedCollisionObject
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import numpy as np

class BimanualIKCheck(Node):
    """
    Script de Sanity Check Bimanual usando el patrón 'IK-Split'.
    Obtiene coordenadas dinámicamente de TF2 (One-Shot Listener).
    """
    def __init__(self):
        super().__init__('bimanual_ik_check')
        
        # --- Configuración del Grupo de Planificación ---
        self.arm_group = 'both_arms'
        self.base_frame = 'pelvis' 
        self.left_ee_link = 'L_palm_tcp'
        self.right_ee_link = 'R_palm_tcp'
        self.offset_tcp = 0.0 # 9 cm desde la muñeca hasta el centro de agarre en los dedos
        
        # --- Parámetros Geométricos del Agarre ---
        self.separacion_hover = 0.10  # Separación entre palmas durante el acercamiento/descenso
        self.separacion_grasp = 0.064 # Separación entre palmas durante el agarre (Squeeze/Lift)
        self.dimension_cubo = 0.055   # Dimensión (lado) del cubo manipulado
        
        # --- Control de Ejecución ---
        self.fase_objetivo = 4        # Máxima fase a ejecutar (1 al 5)
        
        # Nombres de los joints para filtrar (7 por brazo en H1-2)
        self.left_joints = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'
        ]
        self.right_joints = [
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Callback Group ---
        self.cb_group = ReentrantCallbackGroup()
        
        # --- Clientes y Publishers ---
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik', callback_group=self.cb_group)
        self.current_fase = 1
        
        self.marker_pub = self.create_publisher(Marker, 'centro_imaginario_marker', 10)
        self.meta_visual_pub = self.create_publisher(Marker, '/meta_visual_robot', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.attach_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando al servicio compute_ik...')
            
        # --- Timer One-Shot Listener ---
        self.get_logger().info('Iniciando búsqueda de TF "objeto_cubo"...')
        self.timer = self.create_timer(1.0, self.buscar_tf_y_ejecutar, callback_group=self.cb_group)

    def buscar_tf_y_ejecutar(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, 
                'objeto_cubo', 
                rclpy.time.Time()
            )
            
            # ¡TF Obtenida exitosamente! Cancelamos el temporizador
            self.timer.cancel()
            self.get_logger().info('¡Transformada encontrada! Cancelando bucle de búsqueda TF2.')
            
            # Extraer posición del objeto
            obj_x = trans.transform.translation.x
            obj_y = trans.transform.translation.y
            obj_z = trans.transform.translation.z
            
            # Extraer orientación (Yaw)
            quat = trans.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            euler = r.as_euler('xyz', degrees=False)
            yaw_rad = euler[2] # Ángulo Z (Yaw)
            
            # Visualizar el cubo detectado
            self.publicar_marcador_verde(obj_x, obj_y, obj_z)
            
            # Llamar a planificación con las coordenadas base y el yaw
            self.plan_and_execute(obj_x, obj_y, obj_z, yaw_rad)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info('Esperando transformada TF2 de "objeto_cubo"...')

    def publicar_marcador_verde(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "meta_visual"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 0.5 # Transparente
        marker.color.r = 0.0
        marker.color.g = 1.0 # Verde
        marker.color.b = 0.0
        self.meta_visual_pub.publish(marker)

    def limpiar_escena(self):
        """Limpia la escena de planificación de objetos previos para evitar colisiones fantasmas."""
        self.get_logger().info('Sanitizando escena de planificación...')
        
        # 1. Eliminar la mesa
        mesa_remove = CollisionObject()
        mesa_remove.header.frame_id = self.base_frame
        mesa_remove.id = 'mesa_trabajo'
        mesa_remove.operation = CollisionObject.REMOVE
        self.collision_pub.publish(mesa_remove)
        
        # 2. Desvincular el cubo de las manos
        aco_remove = AttachedCollisionObject()
        aco_remove.link_name = self.left_ee_link
        aco_remove.object.id = 'cubo_manipulado'
        aco_remove.object.operation = CollisionObject.REMOVE
        self.attach_pub.publish(aco_remove)
        
        # 3. Eliminar el cubo del mundo
        cubo_remove = CollisionObject()
        cubo_remove.header.frame_id = self.base_frame
        cubo_remove.id = 'cubo_manipulado'
        cubo_remove.operation = CollisionObject.REMOVE
        self.collision_pub.publish(cubo_remove)
        
        # Dar tiempo al servidor de MoveIt para procesar
        time.sleep(0.5)

    def añadir_mesa_colision(self):
        """Agrega un objeto de colisión (mesa) a la escena de MoveIt."""
        # --- Parametrización ---
        ancho_mesa = 1.2
        profundidad_mesa = 0.8
        grosor_mesa = 0.05
        dist_robot_mesa = 0.15  # 15 cm desde la pelvis al borde
        z_superficie_mesa = -0.04 # Altura de la superficie respecto a la pelvis

        # --- Creación del Objeto ---
        mesa = CollisionObject()
        mesa.header.frame_id = self.base_frame
        mesa.id = 'mesa_trabajo'
        mesa.operation = CollisionObject.ADD

        # Forma de la mesa (BOX)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [profundidad_mesa, ancho_mesa, grosor_mesa]

        # Posición de la mesa (Centrada en Y, desplazada en X y Z)
        pose = Pose()
        pose.position.x = dist_robot_mesa + (profundidad_mesa / 2.0)
        pose.position.y = 0.0
        pose.position.z = z_superficie_mesa - (grosor_mesa / 2.0)

        mesa.primitives.append(box)
        mesa.primitive_poses.append(pose)

        # Publicar y dar tiempo a MoveIt para actualizar
        self.get_logger().info('Inyectando mesa de colisión en la escena...')
        self.collision_pub.publish(mesa)
        time.sleep(1.0)

    def publicar_marker_centro(self, x, y, z):
        """Dibuja una esfera en RViz para representar el centro del Pre-Grasp."""
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "centro"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0 # Amarillo
        self.marker_pub.publish(marker)

    def get_ik(self, group, link, pos, quat):
        """Construye y envía una solicitud de IK para un grupo específico."""
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = link
        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.pose.position.x = float(pos[0])
        req.ik_request.pose_stamped.pose.position.y = float(pos[1])
        req.ik_request.pose_stamped.pose.position.z = float(pos[2])
        req.ik_request.pose_stamped.pose.orientation.x = float(quat[0])
        req.ik_request.pose_stamped.pose.orientation.y = float(quat[1])
        req.ik_request.pose_stamped.pose.orientation.z = float(quat[2])
        req.ik_request.pose_stamped.pose.orientation.w = float(quat[3])
        req.ik_request.avoid_collisions = True
        
        return self.ik_client.call(req)

    def plan_and_execute(self, obj_x, obj_y, obj_z, yaw_rad):
        # -1. Sanitizar escena
        self.limpiar_escena()
        
        # 0. Inyectar entorno de colisión
        self.añadir_mesa_colision()

        # =================================================================
        # 1. PARÁMETROS DE PRE-GRASP (Órbita 2.5D)
        # =================================================================
        centro_x = obj_x
        centro_y = obj_y
        centro_z = obj_z + 0.15 # Offset seguro para Pre-Grasp (+15 cm arriba del cubo)
        
        # Guardar estado para Fase 2 (Descenso Cartesiano)
        self.obj_x = obj_x
        self.obj_y = obj_y
        self.yaw_rad = yaw_rad
        self.centro_z_hover = centro_z
        
        separacion = self.separacion_hover + (2 * self.offset_tcp)
        
        # --- Cálculo de Órbita 2.5D ---
        d = separacion / 2.0
        
        pos_izq_x = centro_x - d * math.sin(yaw_rad)
        pos_izq_y = centro_y + d * math.cos(yaw_rad)
        pos_izq_z = centro_z
        
        pos_der_x = centro_x + d * math.sin(yaw_rad)
        pos_der_y = centro_y - d * math.cos(yaw_rad)
        pos_der_z = centro_z

        pos_izq = [pos_izq_x, pos_izq_y, pos_izq_z]
        pos_der = [pos_der_x, pos_der_y, pos_der_z]
        
        # --- Orientación de Muñecas (Yaw) ---
        yaw_deg = math.degrees(yaw_rad)
        euler_izq = [0.0, 0.0, 90.0 + yaw_deg] 
        euler_der = [0.0, -179.9, 90.0 + yaw_deg] 
        # =================================================================

        # Publicar marcador visual en RViz (Posición objetivo Pre-Grasp)
        self.publicar_marker_centro(centro_x, centro_y, centro_z)

        # Cálculo de cuaterniones finales
        quat_izq = R.from_euler('xyz', euler_izq, degrees=True).as_quat()
        quat_der = R.from_euler('xyz', euler_der, degrees=True).as_quat()
        
        self.get_logger().info(f'--- IK-SPLIT: Posicionando brazos a {separacion}m ---')
        
        # --- IK BRAZO IZQUIERDO ---
        res_izq = self.get_ik('left_arm', self.left_ee_link, pos_izq, quat_izq)
        if res_izq.error_code.val != 1:
            self.get_logger().error(f'Brazo IZQUIERDO fuera de alcance. Código: {res_izq.error_code.val}')
            return
            
        # --- IK BRAZO DERECHO ---
        res_der = self.get_ik('right_arm', self.right_ee_link, pos_der, quat_der)
        if res_der.error_code.val != 1:
            self.get_logger().error(f'Brazo DERECHO fuera de alcance. Código: {res_der.error_code.val}')
            return

        # Fusión de 14 joints
        names_izq, pos_izq_vals = list(res_izq.solution.joint_state.name), list(res_izq.solution.joint_state.position)
        names_der, pos_der_vals = list(res_der.solution.joint_state.name), list(res_der.solution.joint_state.position)
        
        target_names, target_positions = [], []
        
        for name, pos in zip(names_izq, pos_izq_vals):
            if name in self.left_joints:
                target_names.append(name)
                target_positions.append(pos)
        
        for name, pos in zip(names_der, pos_der_vals):
            if name in self.right_joints:
                target_names.append(name)
                target_positions.append(pos)

        if len(target_names) != 14:
            self.get_logger().error(f'Error en fusión: {len(target_names)}/14 joints detectados.')
            return

        self.enviar_meta_articular(target_names, target_positions)

    def enviar_meta_articular(self, names, positions, fase=1):
        self.current_fase = fase
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.arm_group
        req.num_planning_attempts = 20
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.1
        
        constraints = Constraints()
        for name, pos in zip(names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraints)
        goal_msg.request = req
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Planificación RECHAZADA.')
            return
        self.get_logger().info('Planificación ACEPTADA, ejecutando...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            z_final = self.centro_z_hover - 0.15 # Bajar 15 cm (hasta z original del objeto)
            
            if self.current_fase == 1:
                self.get_logger().info('¡ÉXITO! Fase 1 (Pre-Grasp) completada.')
                if self.fase_objetivo > 1:
                    self.iniciar_fase_descenso(self.obj_x, self.obj_y, z_final, self.yaw_rad)
                else:
                    self.get_logger().info('Fase objetivo alcanzada. Terminando ejecución.')
                    rclpy.shutdown()
                
            elif self.current_fase == 2:
                self.get_logger().info('¡ÉXITO! Fase 2 (Descenso) completada.')
                if self.fase_objetivo > 2:
                    self.get_logger().info('Iniciando Squeeze (Fase 3)...')
                    self.iniciar_fase_squeeze(self.obj_x, self.obj_y, z_final, self.yaw_rad)
                else:
                    self.get_logger().info('Fase objetivo alcanzada. Terminando ejecución.')
                    rclpy.shutdown()
                
            elif self.current_fase == 3:
                self.get_logger().info('¡ÉXITO! Fase 3 (Squeeze) completada.')
                if self.fase_objetivo > 3:
                    self.get_logger().info('Adjuntando objeto (Fase 4)...')
                    self.adjuntar_cubo(self.obj_x, self.obj_y, z_final, self.yaw_rad)
                else:
                    self.get_logger().info('Fase objetivo alcanzada. Terminando ejecución.')
                    rclpy.shutdown()
                
            elif self.current_fase == 5:
                self.get_logger().info('¡ÉXITO TOTAL! Fase 5 (Lift) completada. Pick finalizado.')
                rclpy.shutdown()
                
        else:
            self.get_logger().error(f'Error en ejecución Fase {self.current_fase}: {result.error_code.val}')
            rclpy.shutdown()

    def iniciar_fase_descenso(self, obj_x, obj_y, z_final, yaw_rad):
        self.get_logger().info('Iniciando cálculo de Descenso Articular (Fase 2)...')
        separacion = self.separacion_hover + (2 * self.offset_tcp)
        d = separacion / 2.0
        
        yaw_deg = math.degrees(yaw_rad)
        euler_izq = [0.0, 0.0, 90.0 + yaw_deg] 
        euler_der = [0.0, -179.9, 90.0 + yaw_deg] 
        quat_izq = R.from_euler('xyz', euler_izq, degrees=True).as_quat()
        quat_der = R.from_euler('xyz', euler_der, degrees=True).as_quat()

        # Posiciones XY constantes (Bajada vertical) a Z final
        pos_izq_x = obj_x - d * math.sin(yaw_rad)
        pos_izq_y = obj_y + d * math.cos(yaw_rad)
        pos_izq_z = z_final
        
        pos_der_x = obj_x + d * math.sin(yaw_rad)
        pos_der_y = obj_y - d * math.cos(yaw_rad)
        pos_der_z = z_final
        
        pos_izq = [pos_izq_x, pos_izq_y, pos_izq_z]
        pos_der = [pos_der_x, pos_der_y, pos_der_z]
        
        res_izq = self.get_ik('left_arm', self.left_ee_link, pos_izq, quat_izq)
        res_der = self.get_ik('right_arm', self.right_ee_link, pos_der, quat_der)
        
        if res_izq.error_code.val == 1 and res_der.error_code.val == 1:
            names_izq = list(res_izq.solution.joint_state.name)
            pos_izq_vals = list(res_izq.solution.joint_state.position)
            
            names_der = list(res_der.solution.joint_state.name)
            pos_der_vals = list(res_der.solution.joint_state.position)
            
            target_names, target_positions = [], []
            
            for name, pos in zip(names_izq, pos_izq_vals):
                if name in self.left_joints:
                    target_names.append(name)
                    target_positions.append(pos)
            
            for name, pos in zip(names_der, pos_der_vals):
                if name in self.right_joints:
                    target_names.append(name)
                    target_positions.append(pos)
                    
            if len(target_names) == 14:
                self.enviar_meta_articular(target_names, target_positions, fase=2)
            else:
                self.get_logger().error(f'Error en fusión Fase 2: {len(target_names)}/14 joints detectados.')
                rclpy.shutdown()
        else:
            self.get_logger().error('Error de IK en la Fase 2 (Descenso).')
            rclpy.shutdown()

    def iniciar_fase_squeeze(self, obj_x, obj_y, z_final, yaw_rad):
        self.get_logger().info('Iniciando cálculo de Squeeze Articular (Fase 3)...')
        separacion = self.separacion_grasp + (2 * self.offset_tcp)
        d = separacion / 2.0
        
        yaw_deg = math.degrees(yaw_rad)
        euler_izq = [0.0, 0.0, 90.0 + yaw_deg] 
        euler_der = [0.0, -179.9, 90.0 + yaw_deg] 
        quat_izq = R.from_euler('xyz', euler_izq, degrees=True).as_quat()
        quat_der = R.from_euler('xyz', euler_der, degrees=True).as_quat()

        # Posiciones XY constantes (Apretando hacia adentro)
        pos_izq_x = obj_x - d * math.sin(yaw_rad)
        pos_izq_y = obj_y + d * math.cos(yaw_rad)
        pos_izq_z = z_final
        
        pos_der_x = obj_x + d * math.sin(yaw_rad)
        pos_der_y = obj_y - d * math.cos(yaw_rad)
        pos_der_z = z_final
        
        pos_izq = [pos_izq_x, pos_izq_y, pos_izq_z]
        pos_der = [pos_der_x, pos_der_y, pos_der_z]
        
        res_izq = self.get_ik('left_arm', self.left_ee_link, pos_izq, quat_izq)
        res_der = self.get_ik('right_arm', self.right_ee_link, pos_der, quat_der)
        
        if res_izq.error_code.val == 1 and res_der.error_code.val == 1:
            names_izq = list(res_izq.solution.joint_state.name)
            pos_izq_vals = list(res_izq.solution.joint_state.position)
            
            names_der = list(res_der.solution.joint_state.name)
            pos_der_vals = list(res_der.solution.joint_state.position)
            
            target_names, target_positions = [], []
            
            for name, pos in zip(names_izq, pos_izq_vals):
                if name in self.left_joints:
                    target_names.append(name)
                    target_positions.append(pos)
            
            for name, pos in zip(names_der, pos_der_vals):
                if name in self.right_joints:
                    target_names.append(name)
                    target_positions.append(pos)
                    
            if len(target_names) == 14:
                self.enviar_meta_articular(target_names, target_positions, fase=3)
            else:
                self.get_logger().error(f'Error en fusión Fase 3: {len(target_names)}/14 joints detectados.')
                rclpy.shutdown()
        else:
            self.get_logger().error('Error de IK en la Fase 3 (Squeeze).')
            rclpy.shutdown()

    def adjuntar_cubo(self, obj_x, obj_y, z_final, yaw_rad):
        self.get_logger().info('Iniciando Fase 4 (Attach Object)...')
        
        aco = AttachedCollisionObject()
        aco.link_name = self.left_ee_link
        
        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.id = 'cubo_manipulado'
        obj.operation = CollisionObject.ADD
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [self.dimension_cubo, self.dimension_cubo, self.dimension_cubo]
        
        # Convertir Yaw a Cuaternión para la orientación del cubo
        quat = R.from_euler('xyz', [0, 0, yaw_rad], degrees=False).as_quat()
        
        pose = Pose()
        pose.position.x = float(obj_x)
        pose.position.y = float(obj_y)
        pose.position.z = float(z_final)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        
        obj.primitives.append(box)
        obj.primitive_poses.append(pose)
        
        aco.object = obj
        
        # --- Generación Dinámica de Touch Links ---
        # Es crucial ignorar las colisiones no solo del TCP virtual y la base,
        # sino también de TODA la malla física de los dedos para evitar fallos en Fase 5.
        touch_links_extendidos = [
            self.left_ee_link, self.right_ee_link,
            'L_hand_base_link', 'R_hand_base_link'
        ]
        
        prefijos = ['L_', 'R_']
        dedos = ['thumb', 'index', 'middle', 'ring', 'pinky']
        sub_links = ['_proximal', '_intermediate', '_distal', '_link', '_sh', '_un', '_1', '_2', '_3']
        
        for prefijo in prefijos:
            for dedo in dedos:
                for sub in sub_links:
                    touch_links_extendidos.append(prefijo + dedo + sub)
                    
        aco.touch_links = touch_links_extendidos
        
        self.attach_pub.publish(aco)
        
        self.get_logger().info('[INFO] Fase 4 (Attach) completada. El robot ha agarrado el cubo.')
        time.sleep(0.5)
        
        if self.fase_objetivo > 4:
            self.iniciar_fase_levantamiento(obj_x, obj_y, z_final, yaw_rad)
        else:
            self.get_logger().info('Fase objetivo alcanzada. Terminando ejecución.')
            rclpy.shutdown()

    def iniciar_fase_levantamiento(self, obj_x, obj_y, z_final, yaw_rad):
        self.get_logger().info('Iniciando cálculo de Levantamiento Articular (Fase 5)...')
        z_lift = z_final + 0.15
        separacion = self.separacion_grasp + (2 * self.offset_tcp)
        d = separacion / 2.0
        
        yaw_deg = math.degrees(yaw_rad)
        euler_izq = [0.0, 0.0, 90.0 + yaw_deg] 
        euler_der = [0.0, -179.9, 90.0 + yaw_deg] 
        quat_izq = R.from_euler('xyz', euler_izq, degrees=True).as_quat()
        quat_der = R.from_euler('xyz', euler_der, degrees=True).as_quat()

        # Posiciones XY constantes (Levantamiento vertical)
        pos_izq_x = obj_x - d * math.sin(yaw_rad)
        pos_izq_y = obj_y + d * math.cos(yaw_rad)
        pos_izq_z = z_lift
        
        pos_der_x = obj_x + d * math.sin(yaw_rad)
        pos_der_y = obj_y - d * math.cos(yaw_rad)
        pos_der_z = z_lift
        
        pos_izq = [pos_izq_x, pos_izq_y, pos_izq_z]
        pos_der = [pos_der_x, pos_der_y, pos_der_z]
        
        res_izq = self.get_ik('left_arm', self.left_ee_link, pos_izq, quat_izq)
        res_der = self.get_ik('right_arm', self.right_ee_link, pos_der, quat_der)
        
        if res_izq.error_code.val == 1 and res_der.error_code.val == 1:
            names_izq = list(res_izq.solution.joint_state.name)
            pos_izq_vals = list(res_izq.solution.joint_state.position)
            
            names_der = list(res_der.solution.joint_state.name)
            pos_der_vals = list(res_der.solution.joint_state.position)
            
            target_names, target_positions = [], []
            
            for name, pos in zip(names_izq, pos_izq_vals):
                if name in self.left_joints:
                    target_names.append(name)
                    target_positions.append(pos)
            
            for name, pos in zip(names_der, pos_der_vals):
                if name in self.right_joints:
                    target_names.append(name)
                    target_positions.append(pos)
                    
            if len(target_names) == 14:
                self.enviar_meta_articular(target_names, target_positions, fase=5)
            else:
                self.get_logger().error(f'Error en fusión Fase 5: {len(target_names)}/14 joints detectados.')
                rclpy.shutdown()
        else:
            self.get_logger().error('Error de IK en la Fase 5 (Lift).')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = BimanualIKCheck()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()
