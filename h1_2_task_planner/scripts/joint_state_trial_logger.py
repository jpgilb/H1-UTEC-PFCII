#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from sensor_msgs.msg import JointState
from std_msgs.msg import String


class JointStateTrialLogger(Node):
    def __init__(self, output_dir, joints_filter):
        super().__init__("joint_state_trial_logger")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.joints_filter = set(joints_filter) if joints_filter else None

        self.active = False
        self.condition = ""
        self.trial = 0
        self.start_time = None

        self.file_path = os.path.join(self.output_dir, "actual_joint_states.csv")
        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "condition",
            "trial",
            "time_s",
            "joint_name",
            "position_rad",
            "velocity_rad_s",
            "effort"
        ])

        self.event_sub = self.create_subscription(
            String,
            "/experiment_event",
            self.event_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            50
        )

        self.get_logger().info(f"Registrando joint_states en: {self.file_path}")

    def event_callback(self, msg):
        parts = msg.data.strip().split(",")

        if len(parts) != 3:
            self.get_logger().warn(f"Evento inválido: {msg.data}")
            return

        event, condition, trial = parts

        if event == "START":
            self.active = True
            self.condition = condition
            self.trial = int(trial)
            self.start_time = self.get_clock().now().nanoseconds * 1e-9

            self.get_logger().info(
                f"Iniciando registro: {self.condition}, repetición {self.trial}"
            )

        elif event == "END":
            self.get_logger().info(
                f"Finalizando registro: {self.condition}, repetición {self.trial}"
            )
            self.active = False
            self.file.flush()

    def joint_state_callback(self, msg):
        if not self.active or self.start_time is None:
            return

        current_time = self.get_clock().now().nanoseconds * 1e-9
        t = current_time - self.start_time

        for i, joint_name in enumerate(msg.name):
            if self.joints_filter is not None and joint_name not in self.joints_filter:
                continue

            position = msg.position[i] if i < len(msg.position) else float("nan")
            velocity = msg.velocity[i] if i < len(msg.velocity) else float("nan")
            effort = msg.effort[i] if i < len(msg.effort) else float("nan")

            self.writer.writerow([
                self.condition,
                self.trial,
                f"{t:.6f}",
                joint_name,
                f"{position:.8f}",
                f"{velocity:.8f}",
                f"{effort:.8f}"
            ])

        self.file.flush()

    def destroy_node(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    ros_args = remove_ros_args(args=sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/tmp/h1_2_results",
        help="Carpeta donde se guardará actual_joint_states.csv"
    )
    parser.add_argument(
        "--joints",
        default="left_shoulder_pitch_joint,left_shoulder_roll_joint,left_shoulder_yaw_joint,left_elbow_joint,left_wrist_roll_joint,left_wrist_pitch_joint,left_wrist_yaw_joint",
        help="Lista de articulaciones separadas por coma. Usar vacío para registrar todas."
    )

    parsed_args = parser.parse_args(ros_args[1:])

    joints_filter = []
    if parsed_args.joints.strip():
        joints_filter = [j.strip() for j in parsed_args.joints.split(",") if j.strip()]

    rclpy.init(args=args)

    node = JointStateTrialLogger(
        output_dir=parsed_args.output_dir,
        joints_filter=joints_filter
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
