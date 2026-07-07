#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_robot.py

Sanea la Planning Scene de MoveIt y devuelve los brazos a su posición inicial (Home).
Flujo secuencial:
1. Eliminar y desvincular el cubo (para liberar las manos).
2. Regresar a Home (esquivando la mesa, que permanece activa).
3. Eliminar la mesa (una vez que los brazos están retirados en Home).
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, 
    Constraints, 
    JointConstraint,
    CollisionObject,
    AttachedCollisionObject
)

class RobotResetter(Node):
    def __init__(self):
        super().__init__('robot_resetter')
        
        # --- Configuración ---
        self.base_frame = 'pelvis'
        self.left_ee_link = 'L_palm_tcp'
        self.right_ee_link = 'R_palm_tcp'
        
        # --- Clientes y Publishers ---
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.attach_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        
        self.get_logger().info('Esperando a MoveIt para iniciar reset secuencial...')
        self._action_client.wait_for_server()
        
        # Paso 1: Eliminar el cubo y desvincular de los TCPs
        self.eliminar_cubo()
        
        # Paso 2: Enviar los brazos a la posición inicial (Home)
        self.send_reset_goal()

    def eliminar_cubo(self):
        """Desvincula de las manos y elimina de la escena todos los posibles IDs del cubo."""
        self.get_logger().info('Paso 1: Desvinculando y eliminando cubo de la escena...')
        
        object_ids = ['objeto_manipulado', 'objeto_interno_colision', 'cubo_manipulado']
        ee_links = [self.left_ee_link, self.right_ee_link]

        # 1. Desvincular de ambos brazos (AttachedCollisionObject)
        for obj_id in object_ids:
            for ee_link in ee_links:
                aco_remove = AttachedCollisionObject()
                aco_remove.link_name = ee_link
                aco_remove.object.id = obj_id
                aco_remove.object.operation = CollisionObject.REMOVE
                self.attach_pub.publish(aco_remove)
        
        # 2. Eliminar del mundo (CollisionObject)
        for obj_id in object_ids:
            cubo_remove = CollisionObject()
            cubo_remove.header.frame_id = self.base_frame
            cubo_remove.id = obj_id
            cubo_remove.operation = CollisionObject.REMOVE
            self.collision_pub.publish(cubo_remove)
            
        # Pequeño retardo para que MoveIt procese la limpieza del cubo antes del movimiento
        time.sleep(0.5)

    def send_reset_goal(self):
        """Envía el comando de reset articular para ambos brazos (Joints -> 0.0)."""
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'both_arms'
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2  # Movimiento de reset tranquilo e impecable
        
        # 14 articulaciones principales del H1 (7 por brazo)
        joints = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        constraints = Constraints()
        for j in joints:
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = 0.0  # Posición inicial (Home)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraints)
        goal_msg.request = req
        
        self.get_logger().info('Paso 2: Enviando comando de movimiento a Home (Joints -> 0.0)...')
        self._action_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt rechazó la trayectoria articular a Home.')
            # En caso de rechazo, procedemos a limpiar la mesa igualmente por seguridad
            self.eliminar_mesa()
            rclpy.shutdown()
            return
        self.get_logger().info('Planificación a Home aceptada, ejecutando movimiento...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        error_code = future.result().result.error_code.val
        if error_code == 1:
            self.get_logger().info('¡Éxito! Brazos posicionados en Home.')
        else:
            self.get_logger().error(f'Fallo al mover a Home. Código de error MoveIt: {error_code}')
        
        # Paso 3: Eliminar la mesa una vez concluido el movimiento
        self.eliminar_mesa()
        rclpy.shutdown()

    def eliminar_mesa(self):
        """Elimina el obstáculo mesa de la escena de colisiones."""
        self.get_logger().info('Paso 3: Eliminando la mesa de trabajo de la escena...')
        mesa_remove = CollisionObject()
        mesa_remove.header.frame_id = self.base_frame
        mesa_remove.id = 'mesa_trabajo'
        mesa_remove.operation = CollisionObject.REMOVE
        self.collision_pub.publish(mesa_remove)
        
        # Pequeña pausa para asegurar la publicación del mensaje antes de la destrucción del nodo
        time.sleep(0.5)
        self.get_logger().info('Limpieza e inicialización completadas.')

def main():
    rclpy.init()
    node = RobotResetter()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
