#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class ReadyPose(Node):
    def __init__(self):
        super().__init__("send_ready_pose")

        self.left_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_arm_controller/follow_joint_trajectory",
        )
        self.right_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_arm_controller/follow_joint_trajectory",
        )
        self.left_hand_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_hand_controller/follow_joint_trajectory",
        )
        self.right_hand_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_hand_controller/follow_joint_trajectory",
        )

    def send_goal(self, client, joint_names, positions, duration=3.0):
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Servidor no disponible: {client._action_name}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        goal.trajectory.points.append(point)

        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError(f"Goal rechazado por {client._action_name}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Fallo en {client._action_name}: codigo={result.error_code}, "
                f"detalle={result.error_string}"
            )
        return result

    def run(self):
        self.get_logger().info("Enviando postura READY de brazos...")

        left_arm_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

        right_arm_joints = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # Postura de espera: codos flexionados, brazos separados del tablero,
        # muñecas neutras. Si visualmente queda demasiado abierta/cerrada,
        # se ajusta después de verla en RViz/MuJoCo.
        left_arm_ready = [-0.30, 0.20, 0.00, 1.50, 0.00, 0.00, 0.00]
        right_arm_ready = [-0.30, -0.20, 0.00, 1.50, 0.00, 0.00, 0.00]

        self.send_goal(self.left_arm_client, left_arm_joints, left_arm_ready, 3.0)
        self.send_goal(self.right_arm_client, right_arm_joints, right_arm_ready, 3.0)

        self.get_logger().info("Abriendo manos...")

        left_hand_joints = [
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_proximal_yaw_joint",
        ]

        right_hand_joints = [
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
        ]

        self.send_goal(self.left_hand_client, left_hand_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0)
        self.send_goal(self.right_hand_client, right_hand_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0)

        self.get_logger().info("READY completado.")


def main():
    rclpy.init()
    node = ReadyPose()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
