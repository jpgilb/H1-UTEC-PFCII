#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


class HandPoseTrialLogger(Node):
    def __init__(self, output_dir, base_frame, hand_frames, sample_rate_hz):
        super().__init__("hand_pose_trial_logger")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_frame = base_frame
        self.hand_frames = hand_frames
        self.sample_period = 1.0 / sample_rate_hz

        self.active = False
        self.condition = ""
        self.trial = 0
        self.start_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.file_path = os.path.join(self.output_dir, "actual_hand_poses.csv")
        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "condition",
            "trial",
            "time_s",
            "hand",
            "base_frame",
            "x_m",
            "y_m",
            "z_m",
            "qx",
            "qy",
            "qz",
            "qw"
        ])

        self.event_sub = self.create_subscription(
            String,
            "/experiment_event",
            self.event_callback,
            10
        )

        self.timer = self.create_timer(self.sample_period, self.timer_callback)

        self.get_logger().info(f"Registrando poses de mano en: {self.file_path}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info(f"Hand frames: {', '.join(self.hand_frames)}")

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
                f"Iniciando registro cartesiano: {self.condition}, repetición {self.trial}"
            )

        elif event == "END":
            self.get_logger().info(
                f"Finalizando registro cartesiano: {self.condition}, repetición {self.trial}"
            )
            self.active = False
            self.file.flush()

    def timer_callback(self):
        if not self.active or self.start_time is None:
            return

        current_time = self.get_clock().now().nanoseconds * 1e-9
        t = current_time - self.start_time

        for hand_frame in self.hand_frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    hand_frame,
                    rclpy.time.Time()
                )

                tr = transform.transform.translation
                rot = transform.transform.rotation

                self.writer.writerow([
                    self.condition,
                    self.trial,
                    f"{t:.6f}",
                    hand_frame,
                    self.base_frame,
                    f"{tr.x:.8f}",
                    f"{tr.y:.8f}",
                    f"{tr.z:.8f}",
                    f"{rot.x:.8f}",
                    f"{rot.y:.8f}",
                    f"{rot.z:.8f}",
                    f"{rot.w:.8f}"
                ])

            except Exception as exc:
                self.get_logger().debug(
                    f"No se pudo obtener TF {self.base_frame} -> {hand_frame}: {exc}"
                )

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
        help="Carpeta donde se guardará actual_hand_poses.csv"
    )
    parser.add_argument(
        "--base-frame",
        default="world",
        help="Frame base respecto al cual se mide la mano"
    )
    parser.add_argument(
        "--hand-frames",
        default="L_hand_base_link,R_hand_base_link",
        help="Frames de mano separados por coma"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=50.0,
        help="Frecuencia de muestreo en Hz"
    )

    parsed_args = parser.parse_args(ros_args[1:])

    hand_frames = [
        frame.strip()
        for frame in parsed_args.hand_frames.split(",")
        if frame.strip()
    ]

    rclpy.init(args=args)

    node = HandPoseTrialLogger(
        output_dir=parsed_args.output_dir,
        base_frame=parsed_args.base_frame,
        hand_frames=hand_frames,
        sample_rate_hz=parsed_args.sample_rate
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
