#!/usr/bin/env python3
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
        self.left_ee_link = 'L_palm_tcp' # Usando los nuevos TCPs definidos
        self.right_ee_link = 'R_palm_tcp'
        
        # --- Clientes y Publishers ---
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.attach_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        
        self.get_logger().info('Esperando a MoveIt para resetear el robot y limpiar escena...')
        self._action_client.wait_for_server()
        
        # 1. Limpiar entorno
        self.limpiar_escena()
        
        # 2. Enviar a posición inicial (Joints a 0.0)
        self.send_reset_goal()

    def limpiar_escena(self):
        """Elimina todos los objetos de colisión y desvincula objetos adjuntos."""
        self.get_logger().info('Saneando la Planning Scene...')
        
        # 1. Eliminar la mesa
        mesa_remove = CollisionObject()
        mesa_remove.header.frame_id = self.base_frame
        mesa_remove.id = 'mesa_trabajo'
        mesa_remove.operation = CollisionObject.REMOVE
        self.collision_pub.publish(mesa_remove)
        
        # 2. Desvincular el cubo (Attached)
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
        
        # Pequeña pausa para que MoveIt procese la limpieza
        time.sleep(1.0)

    def send_reset_goal(self):
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'both_arms'
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2 # Movimiento de reset más tranquilo
        
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
            jc.position = 0.0 # Posición inicial (Home)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraints)
        goal_msg.request = req
        
        self.get_logger().info('Enviando comando de reset (Joints -> 0.0)...')
        self._action_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt rechazó el reset articular.')
            return
        self.get_logger().info('Plan de Reset aceptado, moviendo...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        error_code = future.result().result.error_code.val
        if error_code == 1:
            self.get_logger().info('¡ÉXITO! Robot en posición inicial y escena limpia.')
        else:
            self.get_logger().error(f'Fallo al resetear. Código: {error_code}')
        rclpy.shutdown()

def main():
    rclpy.init()
    node = RobotResetter()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
