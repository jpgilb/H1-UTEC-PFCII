#!/usr/bin/env python3

import os
import time
import threading
import warnings
from copy import deepcopy

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


warnings.filterwarnings("ignore", message=".*Wayland.*")


class MuJoCoFollowJointTrajectoryBridge(Node):
    def __init__(self):
        super().__init__("mujoco_follow_joint_trajectory_bridge")

        self.declare_parameter(
            "model_path",
            "/home/jpgb/tesisws/src/h1_2_utec/h1_2_description/mjcf/scene_part_sequencing.xml",
        )
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("sim_rate_hz", 500.0)
        self.declare_parameter("joint_state_rate_hz", 50.0)

        # Parámetros críticos para evitar CONTROL_FAILED de MoveIt
        self.declare_parameter("min_trajectory_duration_sec", 3.0)
        self.declare_parameter("goal_settle_timeout_sec", 3.0)
        self.declare_parameter("goal_position_tolerance_rad", 0.10)

        self.model_path = self.get_parameter("model_path").value
        self.use_viewer = bool(self.get_parameter("use_viewer").value)
        self.sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        self.joint_state_rate_hz = float(self.get_parameter("joint_state_rate_hz").value)
        self.min_trajectory_duration_sec = float(self.get_parameter("min_trajectory_duration_sec").value)
        self.goal_settle_timeout_sec = float(self.get_parameter("goal_settle_timeout_sec").value)
        self.goal_position_tolerance_rad = float(self.get_parameter("goal_position_tolerance_rad").value)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MJCF no encontrado: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.get_logger().info(f"Modelo MuJoCo cargado: {self.model_path}")
        self.get_logger().info(f"Bodies: {self.model.nbody}")
        self.get_logger().info(f"Geoms: {self.model.ngeom}")
        self.get_logger().info(f"Joints: {self.model.njnt}")
        self.get_logger().info(f"Actuators: {self.model.nu}")
        self.get_logger().info(f"Cameras: {self.model.ncam}")

        if self.model.nu == 0:
            raise RuntimeError("El modelo no tiene actuadores. Revisa h1_2_stationary_controlled.xml")

        self.lock = threading.Lock()
        self.running = True

        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.actuator_by_joint = self._build_actuator_map()
        self.qpos_addr_by_joint = self._build_qpos_map()
        self.qvel_addr_by_joint = self._build_qvel_map()

        self.controlled_joints = [
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        missing = [j for j in self.controlled_joints if j not in self.actuator_by_joint]
        if missing:
            raise RuntimeError(f"Faltan actuadores para joints: {missing}")

        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)

        self.command = self._build_hold_command()

        self.active_trajectory = None
        self.active_goal_joints = []
        self.traj_start_time = None
        self.goal_done = threading.Event()

        self.action_servers = []
        for action_name in [
            "/both_arms_controller/follow_joint_trajectory",
            "/left_arm_controller/follow_joint_trajectory",
            "/right_arm_controller/follow_joint_trajectory",
        ]:
            server = ActionServer(
                self,
                FollowJointTrajectory,
                action_name,
                execute_callback=self.execute_callback,
                goal_callback=self.goal_callback,
                cancel_callback=self.cancel_callback,
            )
            self.action_servers.append(server)
            self.get_logger().info(f"Action server listo en {action_name}")

        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("Publicando estados en /joint_states")

    @staticmethod
    def _duration_to_sec(duration_msg):
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    @staticmethod
    def _write_sec_to_duration(duration_msg, seconds):
        seconds = max(0.0, float(seconds))
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        duration_msg.sec = sec
        duration_msg.nanosec = nanosec

    def _stretch_trajectory_timing(self, trajectory):
        if not trajectory.points:
            return trajectory

        final_time = self._duration_to_sec(trajectory.points[-1].time_from_start)

        if final_time >= self.min_trajectory_duration_sec:
            return trajectory

        if final_time <= 1e-6:
            n = len(trajectory.points)
            for i, point in enumerate(trajectory.points):
                t = self.min_trajectory_duration_sec * (i / max(1, n - 1))
                self._write_sec_to_duration(point.time_from_start, t)
        else:
            scale = self.min_trajectory_duration_sec / final_time
            for point in trajectory.points:
                t = self._duration_to_sec(point.time_from_start) * scale
                self._write_sec_to_duration(point.time_from_start, t)

        self.get_logger().warn(
            f"Trayectoria corta estirada: {final_time:.3f}s -> "
            f"{self.min_trajectory_duration_sec:.3f}s"
        )
        return trajectory

    def _build_actuator_map(self):
        actuator_by_joint = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue

            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name:
                actuator_by_joint[joint_name] = actuator_id

        return actuator_by_joint

    def _build_qpos_map(self):
        qpos_addr_by_joint = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name:
                qpos_addr_by_joint[joint_name] = int(self.model.jnt_qposadr[joint_id])

        return qpos_addr_by_joint

    def _build_qvel_map(self):
        qvel_addr_by_joint = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name:
                qvel_addr_by_joint[joint_name] = int(self.model.jnt_dofadr[joint_id])

        return qvel_addr_by_joint

    def _set_joint_qpos(self, joint_name, value):
        if joint_name not in self.qpos_addr_by_joint:
            self.get_logger().warn(f"Joint no encontrado en qpos: {joint_name}")
            return

        qpos_addr = self.qpos_addr_by_joint[joint_name]
        self.data.qpos[qpos_addr] = float(value)

    def _set_initial_pose(self):
        initial_pose = {
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": -0.30,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.55,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": -0.30,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.55,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

        mujoco.mj_resetData(self.model, self.data)

        for joint_name, value in initial_pose.items():
            self._set_joint_qpos(joint_name, value)

    def _build_hold_command(self):
        command = {}
        for joint_name in self.actuator_by_joint:
            if joint_name in self.qpos_addr_by_joint:
                qpos_addr = self.qpos_addr_by_joint[joint_name]
                command[joint_name] = float(self.data.qpos[qpos_addr])
        return command

    def _apply_command_to_mujoco(self):
        for joint_name, value in self.command.items():
            if joint_name not in self.actuator_by_joint:
                continue

            actuator_id = self.actuator_by_joint[joint_name]
            cmd = float(value)

            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                cmd = float(np.clip(cmd, low, high))

            self.data.ctrl[actuator_id] = cmd

    def _get_joint_position(self, joint_name):
        if joint_name not in self.qpos_addr_by_joint:
            return 0.0
        return float(self.data.qpos[self.qpos_addr_by_joint[joint_name]])

    def _get_joint_velocity(self, joint_name):
        if joint_name not in self.qvel_addr_by_joint:
            return 0.0
        return float(self.data.qvel[self.qvel_addr_by_joint[joint_name]])

    def _max_goal_error(self, joint_names):
        errors = []
        for joint_name in joint_names:
            desired = self.command.get(joint_name, None)
            if desired is None:
                continue
            actual = self._get_joint_position(joint_name)
            errors.append(abs(float(desired) - float(actual)))

        if not errors:
            return 0.0

        return max(errors)

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = list(self.command.keys())
        msg.position = [self._get_joint_position(j) for j in msg.name]
        msg.velocity = [self._get_joint_velocity(j) for j in msg.name]
        msg.effort = []

        self.joint_state_pub.publish(msg)

    def _insert_initial_point_if_needed(self, trajectory):
        if not trajectory.points:
            return trajectory

        first_time = self._duration_to_sec(trajectory.points[0].time_from_start)

        if first_time <= 1e-6:
            return trajectory

        initial_point = JointTrajectoryPoint()
        initial_point.positions = [
            self._get_joint_position(joint_name)
            for joint_name in trajectory.joint_names
        ]
        initial_point.time_from_start.sec = 0
        initial_point.time_from_start.nanosec = 0

        trajectory.points.insert(0, initial_point)
        return trajectory

    def _interpolate_active_trajectory(self, now):
        if self.active_trajectory is None:
            return False

        elapsed = now - self.traj_start_time
        trajectory = self.active_trajectory
        points = trajectory.points

        if not points:
            self.active_trajectory = None
            self.active_goal_joints = []
            self.goal_done.set()
            return False

        final_time = self._duration_to_sec(points[-1].time_from_start)

        if elapsed >= final_time:
            final_point = points[-1]

            for index, joint_name in enumerate(self.active_goal_joints):
                final_value = float(final_point.positions[index])
                self.command[joint_name] = final_value

            self.active_trajectory = None
            self.active_goal_joints = []
            self.goal_done.set()
            return True

        previous_point = points[0]
        next_point = points[-1]

        for index in range(1, len(points)):
            point_time = self._duration_to_sec(points[index].time_from_start)

            if elapsed <= point_time:
                previous_point = points[index - 1]
                next_point = points[index]
                break

        previous_time = self._duration_to_sec(previous_point.time_from_start)
        next_time = self._duration_to_sec(next_point.time_from_start)

        if abs(next_time - previous_time) < 1e-9:
            alpha = 1.0
        else:
            alpha = (elapsed - previous_time) / (next_time - previous_time)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        for index, joint_name in enumerate(self.active_goal_joints):
            q0 = float(previous_point.positions[index])
            q1 = float(next_point.positions[index])
            self.command[joint_name] = (1.0 - alpha) * q0 + alpha * q1

        return False

    def _simulation_step(self):
        now = time.monotonic()

        with self.lock:
            self._interpolate_active_trajectory(now)
            self._apply_command_to_mujoco()

        mujoco.mj_step(self.model, self.data)

    def _simulation_loop(self):
        sim_period = 1.0 / self.sim_rate_hz
        joint_state_period = 1.0 / self.joint_state_rate_hz
        last_joint_state_time = time.monotonic()

        if self.use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while self.running and rclpy.ok() and viewer.is_running():
                    start_time = time.monotonic()
                    self._simulation_step()

                    now = time.monotonic()
                    if now - last_joint_state_time >= joint_state_period:
                        self._publish_joint_states()
                        last_joint_state_time = now

                    viewer.sync()

                    elapsed = time.monotonic() - start_time
                    time.sleep(max(0.0, sim_period - elapsed))
        else:
            while self.running and rclpy.ok():
                start_time = time.monotonic()
                self._simulation_step()

                now = time.monotonic()
                if now - last_joint_state_time >= joint_state_period:
                    self._publish_joint_states()
                    last_joint_state_time = now

                elapsed = time.monotonic() - start_time
                time.sleep(max(0.0, sim_period - elapsed))

    def goal_callback(self, goal_request):
        trajectory = goal_request.trajectory

        if not trajectory.joint_names:
            self.get_logger().error("Trayectoria rechazada: no contiene joint_names")
            return GoalResponse.REJECT

        unsupported = [
            joint_name
            for joint_name in trajectory.joint_names
            if joint_name not in self.controlled_joints
        ]

        if unsupported:
            self.get_logger().error(
                f"Trayectoria rechazada. Joints no soportados: {unsupported}"
            )
            return GoalResponse.REJECT

        if not trajectory.points:
            self.get_logger().error("Trayectoria rechazada: no contiene puntos")
            return GoalResponse.REJECT

        for point in trajectory.points:
            if len(point.positions) != len(trajectory.joint_names):
                self.get_logger().error(
                    "Trayectoria rechazada: tamaño de positions no coincide con joint_names"
                )
                return GoalResponse.REJECT

        self.get_logger().info(f"Trayectoria aceptada para joints: {trajectory.joint_names}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Cancelación solicitada")

        with self.lock:
            self.active_trajectory = None
            self.active_goal_joints = []
            self.goal_done.set()

        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        trajectory = deepcopy(goal_handle.request.trajectory)
        trajectory = self._insert_initial_point_if_needed(trajectory)
        trajectory = self._stretch_trajectory_timing(trajectory)

        result = FollowJointTrajectory.Result()
        self.goal_done.clear()

        with self.lock:
            self.active_goal_joints = list(trajectory.joint_names)
            self.active_trajectory = trajectory
            self.traj_start_time = time.monotonic()

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(trajectory.joint_names)

        while rclpy.ok() and not self.goal_done.is_set():
            if goal_handle.is_cancel_requested:
                with self.lock:
                    self.active_trajectory = None
                    self.active_goal_joints = []
                    self.goal_done.set()

                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Goal canceled"
                return result

            with self.lock:
                actual_positions = [
                    self._get_joint_position(joint_name)
                    for joint_name in trajectory.joint_names
                ]
                desired_positions = [
                    self.command.get(joint_name, 0.0)
                    for joint_name in trajectory.joint_names
                ]
                error_positions = [
                    desired_positions[index] - actual_positions[index]
                    for index in range(len(trajectory.joint_names))
                ]

            feedback.actual.positions = actual_positions
            feedback.desired.positions = desired_positions
            feedback.error.positions = error_positions

            goal_handle.publish_feedback(feedback)
            time.sleep(0.05)

        settle_start = time.monotonic()
        final_error = float("inf")

        while rclpy.ok():
            with self.lock:
                final_error = self._max_goal_error(trajectory.joint_names)

            if final_error <= self.goal_position_tolerance_rad:
                break

            if time.monotonic() - settle_start > self.goal_settle_timeout_sec:
                self.get_logger().warn(
                    f"Timeout de asentamiento. Error final máximo: {final_error:.4f} rad"
                )
                break

            time.sleep(0.05)

        goal_handle.succeed()

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = ""

        self.get_logger().info(
            f"Trayectoria finalizada correctamente. Error final máximo: {final_error:.4f} rad"
        )

        return result

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    rclpy.init()

    node = MuJoCoFollowJointTrajectoryBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
