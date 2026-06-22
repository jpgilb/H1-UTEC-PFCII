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
from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import TransformBroadcaster


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
        self.declare_parameter("goal_settle_stable_sec", 0.5)
        self.declare_parameter("goal_position_tolerance_rad", 0.10)
        self.declare_parameter("hand_goal_position_tolerance_rad", 0.085)
        self.declare_parameter("upper_body_gravity_compensation", True)
        self.declare_parameter("arm_position_kp_scale", 20.0)
        self.declare_parameter("align_joint_and_actuator_force_limits", True)
        self.declare_parameter("mujoco_object_body", "seq_cube_red")
        self.declare_parameter("object_frame", "objeto_cubo")
        self.declare_parameter("base_body", "pelvis")
        self.declare_parameter("base_frame", "pelvis")
        self.declare_parameter("moveit_attached_object_id", "objeto_manipulado")

        self.model_path = self.get_parameter("model_path").value
        self.use_viewer = bool(self.get_parameter("use_viewer").value)
        self.sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        self.joint_state_rate_hz = float(self.get_parameter("joint_state_rate_hz").value)
        self.min_trajectory_duration_sec = float(self.get_parameter("min_trajectory_duration_sec").value)
        self.goal_settle_timeout_sec = float(self.get_parameter("goal_settle_timeout_sec").value)
        self.goal_settle_stable_sec = float(
            self.get_parameter("goal_settle_stable_sec").value
        )
        self.goal_position_tolerance_rad = float(self.get_parameter("goal_position_tolerance_rad").value)
        self.hand_goal_position_tolerance_rad = float(
            self.get_parameter("hand_goal_position_tolerance_rad").value
        )
        self.upper_body_gravity_compensation = bool(
            self.get_parameter("upper_body_gravity_compensation").value
        )
        self.arm_position_kp_scale = float(
            self.get_parameter("arm_position_kp_scale").value
        )
        self.align_joint_and_actuator_force_limits = bool(
            self.get_parameter("align_joint_and_actuator_force_limits").value
        )
        self.mujoco_object_body = str(self.get_parameter("mujoco_object_body").value)
        self.object_frame = str(self.get_parameter("object_frame").value)
        self.base_body = str(self.get_parameter("base_body").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.moveit_attached_object_id = str(
            self.get_parameter("moveit_attached_object_id").value
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MJCF no encontrado: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        if self.upper_body_gravity_compensation:
            self._enable_upper_body_gravity_compensation()
        self._scale_arm_position_gains()
        if self.align_joint_and_actuator_force_limits:
            self._align_joint_and_actuator_force_limits()

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
        self.tf_broadcaster = TransformBroadcaster(self)

        self.actuator_by_joint = self._build_actuator_map()
        self.qpos_addr_by_joint = self._build_qpos_map()
        self.qvel_addr_by_joint = self._build_qvel_map()
        self._initialize_manipulated_object()
        self.create_subscription(
            AttachedCollisionObject,
            "/attached_collision_object",
            self._attached_object_callback,
            10,
        )

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
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
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
	    "/left_hand_controller/follow_joint_trajectory",
	    "/right_hand_controller/follow_joint_trajectory",
	    "/torso_controller/follow_joint_trajectory",
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

    def _enable_upper_body_gravity_compensation(self):
        torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if torso_id < 0:
            raise RuntimeError("No se encontro el body torso_link para gravcomp")

        compensated_bodies = []
        for body_id in range(1, self.model.nbody):
            ancestor_id = body_id
            while ancestor_id > 0 and ancestor_id != torso_id:
                ancestor_id = int(self.model.body_parentid[ancestor_id])
            if ancestor_id == torso_id:
                self.model.body_gravcomp[body_id] = 1.0
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name:
                    compensated_bodies.append(body_name)

        self.get_logger().info(
            "Compensacion gravitatoria habilitada para torso y extremidades "
            f"superiores ({len(compensated_bodies)} bodies)."
        )

    def _scale_arm_position_gains(self):
        if self.arm_position_kp_scale <= 0.0:
            raise ValueError("arm_position_kp_scale debe ser mayor que cero")

        scaled = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name or not joint_name.startswith(("left_", "right_")):
                continue
            if not any(part in joint_name for part in ("shoulder", "elbow", "wrist")):
                continue

            self.model.actuator_gainprm[actuator_id, 0] *= self.arm_position_kp_scale
            self.model.actuator_biasprm[actuator_id, 1] *= self.arm_position_kp_scale
            scaled.append(joint_name)

        self.get_logger().info(
            f"Ganancias de posicion de brazos escaladas x{self.arm_position_kp_scale:.2f} "
            f"({len(scaled)} actuadores); los limites de fuerza no cambian."
        )

    def _align_joint_and_actuator_force_limits(self):
        aligned = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or not self.model.actuator_forcelimited[actuator_id]:
                continue
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name or not joint_name.startswith(("left_", "right_")):
                continue
            if not any(part in joint_name for part in ("shoulder", "elbow", "wrist")):
                continue

            actuator_range = self.model.actuator_forcerange[actuator_id]
            self.model.jnt_actfrclimited[joint_id] = 1
            self.model.jnt_actfrcrange[joint_id] = actuator_range
            aligned.append(
                f"{joint_name}=[{actuator_range[0]:.1f},{actuator_range[1]:.1f}]"
            )

        self.get_logger().info(
            "Limites de fuerza de joints alineados con sus actuadores: "
            + ", ".join(aligned)
        )

    def _initialize_manipulated_object(self):
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.mujoco_object_body
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body
        )
        if self.object_body_id < 0 or self.base_body_id < 0:
            raise RuntimeError(
                f"No se encontraron bodies de objeto/base: "
                f"{self.mujoco_object_body}, {self.base_body}"
            )

        self.object_joint_id = int(self.model.body_jntadr[self.object_body_id])
        if self.object_joint_id < 0 or int(self.model.jnt_type[self.object_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(
                f"El body {self.mujoco_object_body} no tiene un free joint"
            )
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_addr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.object_attached = False
        self.object_relative_pos = np.zeros(3, dtype=float)
        self.object_relative_rot = np.eye(3, dtype=float)

    def _attachment_body_name(self, link_name):
        if link_name.startswith("L_") or link_name.startswith("left_"):
            return "left_wrist_yaw_link"
        if link_name.startswith("R_") or link_name.startswith("right_"):
            return "right_wrist_yaw_link"
        return None

    def _attached_object_callback(self, message):
        if message.object.id != self.moveit_attached_object_id:
            return

        with self.lock:
            if message.object.operation == CollisionObject.ADD:
                attachment_body_name = self._attachment_body_name(message.link_name)
                if attachment_body_name is None:
                    self.get_logger().error(
                        f"No se pudo mapear link de attachment: {message.link_name}"
                    )
                    return
                attachment_body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, attachment_body_name
                )
                attachment_pos = np.array(self.data.xpos[attachment_body_id], dtype=float)
                attachment_rot = np.array(
                    self.data.xmat[attachment_body_id], dtype=float
                ).reshape(3, 3)
                object_pos = np.array(self.data.xpos[self.object_body_id], dtype=float)
                object_rot = np.array(
                    self.data.xmat[self.object_body_id], dtype=float
                ).reshape(3, 3)
                self.object_relative_pos = attachment_rot.T @ (
                    object_pos - attachment_pos
                )
                self.object_relative_rot = attachment_rot.T @ object_rot
                self.object_attachment_body_id = attachment_body_id
                self.object_attached = True
                self.get_logger().info(
                    f"Objeto MuJoCo adjuntado a {attachment_body_name}."
                )
            elif message.object.operation == CollisionObject.REMOVE:
                if self.object_attached:
                    self.object_attached = False
                    self.data.qvel[self.object_qvel_addr:self.object_qvel_addr + 6] = 0.0
                    self.get_logger().info("Objeto MuJoCo liberado del efector.")

    def _update_attached_object_pose(self):
        if not self.object_attached:
            return
        attachment_pos = np.array(
            self.data.xpos[self.object_attachment_body_id], dtype=float
        )
        attachment_rot = np.array(
            self.data.xmat[self.object_attachment_body_id], dtype=float
        ).reshape(3, 3)
        object_pos = attachment_pos + attachment_rot @ self.object_relative_pos
        object_rot = attachment_rot @ self.object_relative_rot
        object_quat_wxyz = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(object_quat_wxyz, object_rot.reshape(9))
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = object_pos
        self.data.qpos[self.object_qpos_addr + 3:self.object_qpos_addr + 7] = object_quat_wxyz
        self.data.qvel[self.object_qvel_addr:self.object_qvel_addr + 6] = 0.0

    def _publish_object_tf(self):
        base_pos = np.array(self.data.xpos[self.base_body_id], dtype=float)
        base_rot = np.array(self.data.xmat[self.base_body_id], dtype=float).reshape(3, 3)
        object_pos = np.array(self.data.xpos[self.object_body_id], dtype=float)
        object_rot = np.array(self.data.xmat[self.object_body_id], dtype=float).reshape(3, 3)
        relative_pos = base_rot.T @ (object_pos - base_pos)
        relative_rot = base_rot.T @ object_rot
        quat_wxyz = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat_wxyz, relative_rot.reshape(9))

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = self.object_frame
        transform.transform.translation.x = float(relative_pos[0])
        transform.transform.translation.y = float(relative_pos[1])
        transform.transform.translation.z = float(relative_pos[2])
        transform.transform.rotation.w = float(quat_wxyz[0])
        transform.transform.rotation.x = float(quat_wxyz[1])
        transform.transform.rotation.y = float(quat_wxyz[2])
        transform.transform.rotation.z = float(quat_wxyz[3])
        self.tf_broadcaster.sendTransform(transform)

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
            "left_shoulder_roll_joint": 0.20,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.50,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": -0.30,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.50,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

        mujoco.mj_resetData(self.model, self.data)

        # El cubo es simetrico, pero el yaw de 90 grados define la cara de
        # aproximacion usada por la tarea y debe coincidir con su TF dinamica.
        half_yaw = 0.25 * np.pi
        self.data.qpos[self.object_qpos_addr + 3:self.object_qpos_addr + 7] = [
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)
        ]

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
        errors = self._goal_errors(joint_names)
        if not errors:
            return 0.0
        return max(errors.values())

    def _goal_errors(self, joint_names):
        errors = {}
        for joint_name in joint_names:
            desired = self.command.get(joint_name, None)
            if desired is None:
                continue
            actual = self._get_joint_position(joint_name)
            errors[joint_name] = abs(float(desired) - float(actual))
        return errors

    def _goal_tolerance(self, joint_names):
        if joint_names and all(name.startswith(("L_", "R_")) for name in joint_names):
            return self.hand_goal_position_tolerance_rad
        return self.goal_position_tolerance_rad

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = list(self.command.keys())
        msg.position = [self._get_joint_position(j) for j in msg.name]
        msg.velocity = [self._get_joint_velocity(j) for j in msg.name]
        msg.effort = []

        self.joint_state_pub.publish(msg)
        self._publish_object_tf()

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
            self._update_attached_object_pose()

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
        within_tolerance_since = None
        final_error = float("inf")
        goal_tolerance = self._goal_tolerance(trajectory.joint_names)

        while rclpy.ok():
            with self.lock:
                final_error = self._max_goal_error(trajectory.joint_names)

            if final_error <= goal_tolerance:
                if within_tolerance_since is None:
                    within_tolerance_since = time.monotonic()
                if time.monotonic() - within_tolerance_since >= self.goal_settle_stable_sec:
                    break
            else:
                within_tolerance_since = None

            if time.monotonic() - settle_start > self.goal_settle_timeout_sec:
                with self.lock:
                    details = []
                    for joint_name, error in sorted(
                        self._goal_errors(trajectory.joint_names).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    ):
                        details.append(
                            f"{joint_name}: objetivo={self.command[joint_name]:.4f}, "
                            f"real={self._get_joint_position(joint_name):.4f}, "
                            f"error={error:.4f}"
                        )
                self.get_logger().error(
                    f"Timeout de asentamiento. Error final máximo: {final_error:.4f} rad\n  "
                    + "\n  ".join(details)
                )
                goal_handle.abort()
                result.error_code = (
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                )
                result.error_string = (
                    "MuJoCo no alcanzo la tolerancia articular: "
                    f"error_max={final_error:.4f} rad, "
                    f"tolerancia={goal_tolerance:.4f} rad"
                )
                return result

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
