#!/usr/bin/env python3

import os
import json
import math
import time
import threading
from typing import Optional, Dict, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory

import numpy as np
import pinocchio as pin

class H12DynamicsHoldController(Node):
    def __init__(self):
        super().__init__('h1_2_dynamics_hold_controller')
        
        self.lock = threading.Lock()
        self.cb_group = ReentrantCallbackGroup()
        
        # ROS Parameters
        self.declare_parameter('pinocchio_urdf_path', 
            '/home/sebas/ros2_ws/src/h1_2_mujoco_bringup/description/generated/h1_2_pinocchio_dynamics.urdf')
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('gravity_scale', 1.0)
        self.declare_parameter('torque_sign', 1.0)
        self.declare_parameter('enable_pd', True)
        self.declare_parameter('hold_capture_delay_sec', 0.5)
        self.declare_parameter('publish_zero_on_shutdown', True)
        self.declare_parameter('log_rate_hz', 1.0)
        
        # Gain Parameters
        self.declare_parameter('kp_shoulder_pitch_roll', 80.0)
        self.declare_parameter('kd_shoulder_pitch_roll', 10.0)
        self.declare_parameter('kp_shoulder_yaw', 30.0)
        self.declare_parameter('kd_shoulder_yaw', 4.0)
        self.declare_parameter('kp_elbow', 120.0)
        self.declare_parameter('kd_elbow', 14.0)
        self.declare_parameter('kp_wrist', 40.0)
        self.declare_parameter('kd_wrist', 5.0)

        # Telemetry Parameters
        self.declare_parameter('telemetry_enabled', False)
        self.declare_parameter('telemetry_rate_hz', 50.0)
        self.declare_parameter('telemetry_topic', '/h1_2_dynamics_hold_controller/telemetry')
        self.declare_parameter('telemetry_config_id', 'C2_nominal')

        # Get values
        self.urdf_path = self.get_parameter('pinocchio_urdf_path').get_parameter_value().string_value
        self.control_rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self.gravity_scale = self.get_parameter('gravity_scale').get_parameter_value().double_value
        self.torque_sign = self.get_parameter('torque_sign').get_parameter_value().double_value
        self.enable_pd = self.get_parameter('enable_pd').get_parameter_value().bool_value
        self.hold_capture_delay_sec = self.get_parameter('hold_capture_delay_sec').get_parameter_value().double_value
        self.publish_zero_on_shutdown = self.get_parameter('publish_zero_on_shutdown').get_parameter_value().bool_value
        self.log_rate_hz = self.get_parameter('log_rate_hz').get_parameter_value().double_value

        self.kp_shoulder_pitch_roll = self.get_parameter('kp_shoulder_pitch_roll').get_parameter_value().double_value
        self.kd_shoulder_pitch_roll = self.get_parameter('kd_shoulder_pitch_roll').get_parameter_value().double_value
        self.kp_shoulder_yaw = self.get_parameter('kp_shoulder_yaw').get_parameter_value().double_value
        self.kd_shoulder_yaw = self.get_parameter('kd_shoulder_yaw').get_parameter_value().double_value
        self.kp_elbow = self.get_parameter('kp_elbow').get_parameter_value().double_value
        self.kd_elbow = self.get_parameter('kd_elbow').get_parameter_value().double_value
        self.kp_wrist = self.get_parameter('kp_wrist').get_parameter_value().double_value
        self.kd_wrist = self.get_parameter('kd_wrist').get_parameter_value().double_value

        self.telemetry_enabled = self.get_parameter('telemetry_enabled').get_parameter_value().bool_value
        self.telemetry_rate_hz = self.get_parameter('telemetry_rate_hz').get_parameter_value().double_value
        self.telemetry_topic = self.get_parameter('telemetry_topic').get_parameter_value().string_value
        self.telemetry_config_id = self.get_parameter('telemetry_config_id').get_parameter_value().string_value

        # Pinocchio Initialization
        self.get_logger().info(f"[HOLD CONTROL] Loading URDF from: {self.urdf_path}")
        if not os.path.exists(self.urdf_path):
            self.get_logger().error(f"[HOLD CONTROL] URDF file does not exist: {self.urdf_path}")
            raise FileNotFoundError(f"URDF path not found: {self.urdf_path}")
            
        try:
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            self.data = self.model.createData()
            self.get_logger().info(f"[HOLD CONTROL] Model loaded. Joints: {self.model.njoints}, nq={self.model.nq}, nv={self.model.nv}")
        except Exception as e:
            self.get_logger().error(f"[HOLD CONTROL] Failed to build Pinocchio model: {e}")
            raise e

        # Arm Joints Definition
        self.left_joints = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'
        ]
        self.right_joints = [
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        self.all_arm_joints = self.left_joints + self.right_joints

        # Verify joint existence in Pinocchio model
        for joint_name in self.all_arm_joints:
            if not self.model.existJointName(joint_name):
                self.get_logger().error(f"[HOLD CONTROL] Joint '{joint_name}' does not exist in Pinocchio model!")
                raise ValueError(f"Missing joint in Pinocchio: {joint_name}")

        # Nominal Joint limits definition with safety padding in ROS
        self.joint_limits = {
            'left_shoulder_pitch_joint': [-3.14, 1.57],
            'left_shoulder_roll_joint': [-0.38, 3.4],
            'left_shoulder_yaw_joint': [-2.66, 3.01],
            'left_elbow_joint': [-0.95, 3.18],
            'left_wrist_roll_joint': [-3.01, 2.75],
            'left_wrist_pitch_joint': [-0.4625, 0.4625],
            'left_wrist_yaw_joint': [-1.27, 1.27],
            
            'right_shoulder_pitch_joint': [-3.14, 1.57],
            'right_shoulder_roll_joint': [-3.4, 0.38],
            'right_shoulder_yaw_joint': [-3.01, 2.66],
            'right_elbow_joint': [-0.95, 3.18],
            'right_wrist_roll_joint': [-2.75, 3.01],
            'right_wrist_pitch_joint': [-0.4625, 0.4625],
            'right_wrist_yaw_joint': [-1.27, 1.27],
        }

        # Desired state definition (defaults to 0.0 home zero for all joints)
        self.q_desired: Dict[str, float] = {}
        self.dq_desired: Dict[str, float] = {}
        for joint in self.all_arm_joints:
            self.q_desired[joint] = 0.0
            self.dq_desired[joint] = 0.0
            
        # Deprecated alias for backwards compatibility
        self.q_hold = self.q_desired

        # State Machine Variables
        self.latest_joint_state: Optional[JointState] = None
        self.joint_state_received_time: float = 0.0
        self.hold_active: bool = False
        self.last_log_time: float = 0.0
        self.hold_ready_state = False
        self.recaptured_flag = False

        # Active Trajectories Tracking
        self.active_trajectories = {
            'left': None,
            'right': None
        }

        # Debug helper
        self.last_fjt_debug_time = {
            'left': 0.0,
            'right': 0.0
        }

        # Publishers & Subscribers
        self.left_pub = self.create_publisher(Float64MultiArray, '/left_arm_effort_controller/commands', 10)
        self.right_pub = self.create_publisher(Float64MultiArray, '/right_arm_effort_controller/commands', 10)
        self.ready_pub = self.create_publisher(Bool, '/h1_2_dynamics_hold_controller/hold_ready', 10)
        
        # Telemetry Publisher
        from std_msgs.msg import String
        self.telemetry_pub = self.create_publisher(String, self.telemetry_topic, 10)
        self.last_telemetry_pub_time = 0.0
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10, callback_group=self.cb_group)
        self.desired_joint_sub = self.create_subscription(
            JointState, 
            '/h1_2_dynamics_hold_controller/desired_joint_states', 
            self.desired_joint_state_callback, 
            10,
            callback_group=self.cb_group
        )

        # Action Servers
        self.left_fjt_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/left_arm_controller/follow_joint_trajectory',
            execute_callback=lambda gh: self.execute_trajectory(gh, 'left'),
            goal_callback=lambda gr: self.goal_callback(gr, 'left'),
            handle_accepted_callback=lambda gh: self.handle_accepted_trajectory(gh, 'left'),
            cancel_callback=lambda gh: self.cancel_callback(gh, 'left'),
            callback_group=self.cb_group
        )
        self.right_fjt_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/right_arm_controller/follow_joint_trajectory',
            execute_callback=lambda gh: self.execute_trajectory(gh, 'right'),
            goal_callback=lambda gr: self.goal_callback(gr, 'right'),
            handle_accepted_callback=lambda gh: self.handle_accepted_trajectory(gh, 'right'),
            cancel_callback=lambda gh: self.cancel_callback(gh, 'right'),
            callback_group=self.cb_group
        )

        # Services
        self.recapture_srv = self.create_service(Trigger, '/h1_2_dynamics_hold_controller/recapture_hold', self.recapture_hold_callback, callback_group=self.cb_group)

        # Control Loop Timer
        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop, callback_group=self.cb_group)
        self.get_logger().info("[HOLD CONTROL] Initialization complete.")

    def joint_state_callback(self, msg: JointState):
        with self.lock:
            self.latest_joint_state = msg
            self.joint_state_received_time = self.get_clock().now().nanoseconds / 1e9

    def desired_joint_state_callback(self, msg: JointState):
        updated_joints = []
        with self.lock:
            updated_left = False
            updated_right = False
            for i, name in enumerate(msg.name):
                if name in self.all_arm_joints:
                    if len(msg.position) > i:
                        pos = msg.position[i]
                        # 1. Finite check
                        if not math.isfinite(pos):
                            self.get_logger().warn(f"[HOLD CONTROL] Ignored manual setpoint for {name}: position is not finite.")
                            continue
                        # 2. Limits check (with 0.02 rad safety margin)
                        lim = self.joint_limits[name]
                        if pos < lim[0] - 0.02 or pos > lim[1] + 0.02:
                            self.get_logger().warn(f"[HOLD CONTROL] Ignored manual setpoint for {name}: position {pos:.4f} is outside safety limits [{lim[0]-0.02:.4f}, {lim[1]+0.02:.4f}].")
                            continue
                            
                        # If finite and within limits, update q_desired
                        self.q_desired[name] = pos
                        
                        # Velocity check
                        if len(msg.velocity) > i:
                            vel = msg.velocity[i]
                            if math.isfinite(vel):
                                self.dq_desired[name] = vel
                            else:
                                self.dq_desired[name] = 0.0
                                self.get_logger().warn(f"[HOLD CONTROL] Non-finite velocity for {name} ignored, set to 0.0.")
                        else:
                            self.dq_desired[name] = 0.0
                            
                        updated_joints.append(name)
                        if name in self.left_joints:
                            updated_left = True
                        if name in self.right_joints:
                            updated_right = True

            # Cancel active trajectories if manual setpoint is received
            if updated_left and self.active_trajectories['left'] is not None:
                self.get_logger().info("[HOLD CONTROL] Manual setpoint received for left arm. Canceling active trajectory.")
                self.active_trajectories['left']['canceled'] = True
            if updated_right and self.active_trajectories['right'] is not None:
                self.get_logger().info("[HOLD CONTROL] Manual setpoint received for right arm. Canceling active trajectory.")
                self.active_trajectories['right']['canceled'] = True
        
        if updated_joints:
            self.get_logger().info(f"[HOLD CONTROL] Desired setpoint updated manually for joints: {updated_joints}")

    def goal_callback(self, goal_request, side: str) -> GoalResponse:
        self.get_logger().info(f"[FJT SERVER] Received FJT goal request for {side} arm.")
        success, msg = self.validate_goal(goal_request.trajectory, side)
        if success:
            self.get_logger().info(f"[FJT SERVER] {side} FJT goal request ACCEPTED.")
            return GoalResponse.ACCEPT
        else:
            self.get_logger().error(f"[FJT SERVER] {side} FJT goal request REJECTED: {msg}")
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle, side: str) -> CancelResponse:
        self.get_logger().info(f"[FJT SERVER] Received cancel request for {side} arm.")
        with self.lock:
            traj_info = self.active_trajectories[side]
            if traj_info is not None and traj_info['goal_handle'] == goal_handle:
                traj_info['canceled'] = True
                
                # Fix q_desired to current positions measured, if valid
                msg = self.latest_joint_state
                joints = self.left_joints if side == 'left' else self.right_joints
                if msg is not None:
                    for name in joints:
                        if name in msg.name:
                            idx = msg.name.index(name)
                            self.q_desired[name] = msg.position[idx]
                            self.dq_desired[name] = 0.0
        return CancelResponse.ACCEPT

    def validate_goal(self, trajectory, side: str) -> Tuple[bool, str]:
        expected_joints = self.left_joints if side == 'left' else self.right_joints
        
        if set(trajectory.joint_names) != set(expected_joints):
            return False, f"Joint names do not match expected joints for {side} arm. Expected: {expected_joints}, Got: {trajectory.joint_names}"
            
        if not trajectory.points:
            return False, "Trajectory has no points."
            
        n_joints = len(trajectory.joint_names)
        joint_indices = {name: idx for idx, name in enumerate(trajectory.joint_names)}
        
        prev_time = -1e-9
        max_joint_delta = 0.0
        max_est_vel = 0.0
        
        with self.lock:
            prev_positions = {name: self.q_desired[name] for name in expected_joints}
            
        for idx_pt, pt in enumerate(trajectory.points):
            if len(pt.positions) != n_joints:
                return False, f"Point {idx_pt} has positions length {len(pt.positions)} instead of {n_joints}."
                
            pt_time = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if pt_time < 0.0:
                return False, f"Point {idx_pt} has negative time_from_start: {pt_time}."
            if pt_time <= prev_time:
                return False, f"Point {idx_pt} time_from_start {pt_time} is not strictly increasing (prev: {prev_time})."
                
            # Validate positions and velocities
            for name in expected_joints:
                idx_j = joint_indices[name]
                pos = pt.positions[idx_j]
                
                if not math.isfinite(pos):
                    return False, f"Point {idx_pt}, joint {name} position is not finite: {pos}"
                    
                # Limits check
                lim = self.joint_limits[name]
                if pos < lim[0] - 0.02 or pos > lim[1] + 0.02:
                    return False, f"Point {idx_pt}, joint {name} position {pos:.4f} is outside safety limits [{lim[0] - 0.02:.4f}, {lim[1] + 0.02:.4f}]"
                    
                # Check velocity finiteness if provided
                if len(pt.velocities) == n_joints:
                    vel = pt.velocities[idx_j]
                    if not math.isfinite(vel):
                        return False, f"Point {idx_pt}, joint {name} velocity is not finite: {vel}"
                        
                # Calculate delta and velocity estimate
                pos_prev = prev_positions[name]
                delta = abs(pos - pos_prev)
                max_joint_delta = max(max_joint_delta, delta)
                
                dt = pt_time - (0.0 if idx_pt == 0 else prev_time)
                if dt > 1e-6:
                    est_vel = delta / dt
                    max_est_vel = max(max_est_vel, est_vel)
                    
                prev_positions[name] = pos
                
            prev_time = pt_time
            
        # Safety velocity threshold for manual testing
        if max_est_vel > 1.5:
            return False, f"Goal rejected: maximum estimated joint velocity of {max_est_vel:.4f} rad/s exceeds safety limit of 1.5 rad/s."
            
        return True, ""

    def handle_accepted_trajectory(self, goal_handle, side: str):
        trajectory = goal_handle.request.trajectory
        duration = trajectory.points[-1].time_from_start.sec + trajectory.points[-1].time_from_start.nanosec * 1e-9
        
        first_pt = [round(pt, 4) for pt in trajectory.points[0].positions]
        last_pt = [round(pt, 4) for pt in trajectory.points[-1].positions]
        
        self.get_logger().info(
            f"[FJT SERVER] HANDLE ACCEPTED for {side} arm:\n"
            f"  Duration: {duration:.2f} s\n"
            f"  Points: {len(trajectory.points)}\n"
            f"  Joints: {trajectory.joint_names}\n"
            f"  First Point Positions: {first_pt}\n"
            f"  Last Point Positions: {last_pt}"
        )
        
        start_time = self.get_clock().now().nanoseconds / 1e9
        
        with self.lock:
            start_positions = {}
            joints = self.left_joints if side == 'left' else self.right_joints
            for name in joints:
                start_positions[name] = self.q_desired[name]
                
            traj_info = {
                'goal_handle': goal_handle,
                'trajectory': trajectory,
                'start_time': start_time,
                'start_positions': start_positions,
                'finished': False,
                'canceled': False,
                'preempted': False,
                'aborted': False
            }
            
            # Preemption of existing trajectory on the same side
            old_traj = self.active_trajectories[side]
            if old_traj is not None:
                old_traj['preempted'] = True
            
            self.active_trajectories[side] = traj_info
            
        # Call FJT execution monitor
        goal_handle.execute()

    def execute_trajectory(self, goal_handle, side: str):
        self.get_logger().info(f"[FJT SERVER] Monitoring FJT trajectory execution for {side} arm...")
        
        with self.lock:
            traj_info = self.active_trajectories[side]
            
        if traj_info is None or traj_info['goal_handle'] != goal_handle:
            self.get_logger().error(f"[FJT SERVER] Active trajectory info not found for {side} arm!")
            goal_handle.abort()
            return FollowJointTrajectory.Result()
            
        # Monitor execution loop
        try:
            while rclpy.ok():
                with self.lock:
                    if traj_info['canceled']:
                        self.get_logger().warn(f"[FJT SERVER] Trajectory CANCELED for {side} arm.")
                        goal_handle.canceled()
                        return FollowJointTrajectory.Result()
                    if traj_info['preempted']:
                        self.get_logger().warn(f"[FJT SERVER] Trajectory PREEMPTED for {side} arm.")
                        result = FollowJointTrajectory.Result()
                        result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
                        goal_handle.abort()
                        return result
                    if traj_info['aborted']:
                        self.get_logger().error(f"[FJT SERVER] Trajectory ABORTED for {side} arm.")
                        result = FollowJointTrajectory.Result()
                        result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                        goal_handle.abort()
                        return result
                    if traj_info['finished']:
                        self.get_logger().info(f"[FJT SERVER] Trajectory SUCCEEDED for {side} arm.")
                        result = FollowJointTrajectory.Result()
                        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        goal_handle.succeed()
                        return result
                time.sleep(0.02)
        except Exception as e:
            self.get_logger().error(f"[FJT SERVER] Error in execution thread: {e}")
            
        return FollowJointTrajectory.Result()

    def recapture_hold_callback(self, request, response):
        self.get_logger().info("[HOLD CONTROL] Recapture hold requested (resetting desired state to zero home).")
        
        with self.lock:
            for name in self.all_arm_joints:
                self.q_desired[name] = 0.0
                self.dq_desired[name] = 0.0
                
            self.hold_active = True
            self.hold_ready_state = True
            self.recaptured_flag = False

            # Cancel active trajectories
            for side in ['left', 'right']:
                if self.active_trajectories[side] is not None:
                    self.active_trajectories[side]['canceled'] = True

        ready_msg = Bool()
        ready_msg.data = True
        self.ready_pub.publish(ready_msg)

        self.get_logger().info("[HOLD CONTROL] Desired state reset to zero home.")
        response.success = True
        response.message = "Desired state reset to zero home."
        return response

    def get_joint_gains(self, name: str) -> Tuple[float, float]:
        if 'shoulder_pitch' in name or 'shoulder_roll' in name:
            return self.kp_shoulder_pitch_roll, self.kd_shoulder_pitch_roll
        elif 'shoulder_yaw' in name:
            return self.kp_shoulder_yaw, self.kd_shoulder_yaw
        elif 'elbow' in name:
            return self.kp_elbow, self.kd_elbow
        elif 'wrist' in name:
            return self.kp_wrist, self.kd_wrist
        return 0.0, 0.0

    def saturate_torque(self, name: str, tau: float) -> Tuple[float, float]:
        if 'shoulder_pitch' in name or 'shoulder_roll' in name:
            lim = 40.0
        elif 'shoulder_yaw' in name:
            lim = 18.0
        elif 'elbow' in name:
            lim = 18.0
        elif 'wrist' in name:
            lim = 19.0
        else:
            lim = 100.0
        
        sat_val = np.clip(tau, -lim, lim)
        pct = (abs(sat_val) / lim) * 100.0 if lim > 0.0 else 0.0
        return float(sat_val), float(pct)

    def publish_zeros(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 7
        self.left_pub.publish(msg)
        self.right_pub.publish(msg)
        
        # Publish hold_ready as False
        ready_msg = Bool()
        ready_msg.data = False
        self.ready_pub.publish(ready_msg)

    def control_loop(self):
        t = self.get_clock().now().nanoseconds / 1e9
        
        # Check timeout
        with self.lock:
            latest_state = self.latest_joint_state
            received_time = self.joint_state_received_time
            
        if latest_state is None or (t - received_time) > 0.5:
            self.publish_zeros()
            if latest_state is not None:
                self.get_logger().warn("[HOLD CONTROL] Joint states timeout (>0.5s)! Sent zero torques.")
                with self.lock:
                    self.latest_joint_state = None
            return

        # Check wait for valid states and active subscribers before activating control
        if not self.hold_active:
            all_present = all(j in latest_state.name for j in self.all_arm_joints)
            left_subscribers = self.left_pub.get_subscription_count()
            right_subscribers = self.right_pub.get_subscription_count()
            subscribers_active = (left_subscribers >= 1) and (right_subscribers >= 1)
            
            conditions_met = all_present and subscribers_active
            
            if not conditions_met:
                self.publish_zeros()
                if t - self.last_log_time >= (1.0 / self.log_rate_hz):
                    self.last_log_time = t
                    if not all_present:
                        self.get_logger().info("[HOLD CONTROL] Waiting for joint states with all arm joints...")
                    elif not subscribers_active:
                        self.get_logger().info(
                            f"[HOLD CONTROL] Waiting for effort controller subscribers... "
                            f"(Left subs: {left_subscribers}, Right subs: {right_subscribers})"
                        )
                return
            else:
                with self.lock:
                    self.hold_active = True
                    self.hold_ready_state = True
                self.get_logger().info("[HOLD CONTROL] State and controllers ready. Starting hold control at zero home.")

        # Interpolate Desired Trajectories
        with self.lock:
            for side in ['left', 'right']:
                traj_info = self.active_trajectories[side]
                if traj_info is not None and not (traj_info['finished'] or traj_info['canceled'] or traj_info['preempted'] or traj_info['aborted']):
                    dt = t - traj_info['start_time']
                    trajectory = traj_info['trajectory']
                    joints = self.left_joints if side == 'left' else self.right_joints
                    
                    joint_indices = {name: idx for idx, name in enumerate(trajectory.joint_names)}
                    points = trajectory.points
                    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
                    
                    # 1. Before first point
                    if dt < times[0]:
                        t_start = 0.0
                        t_end = times[0]
                        alpha = dt / t_end if t_end > 0.0 else 1.0
                        alpha = max(0.0, min(1.0, alpha)) # Defensive clamping
                        
                        for name in joints:
                            q_start = traj_info['start_positions'][name]
                            q_end = points[0].positions[joint_indices[name]]
                            self.q_desired[name] = (1 - alpha) * q_start + alpha * q_end
                            
                            if len(points[0].velocities) == len(trajectory.joint_names):
                                dq_start = 0.0
                                dq_end = points[0].velocities[joint_indices[name]]
                                self.dq_desired[name] = (1 - alpha) * dq_start + alpha * dq_end
                            else:
                                self.dq_desired[name] = (q_end - q_start) / t_end if t_end > 0.0 else 0.0
                                
                    # 2. After last point
                    elif dt >= times[-1]:
                        for name in joints:
                            self.q_desired[name] = points[-1].positions[joint_indices[name]]
                            self.dq_desired[name] = 0.0
                            
                        # Verify final error tolerance
                        final_error_max = 0.0
                        for name in joints:
                            idx_msg = latest_state.name.index(name)
                            q_actual = latest_state.position[idx_msg]
                            final_error_max = max(final_error_max, abs(self.q_desired[name] - q_actual))
                            
                        if final_error_max <= 0.08:
                            traj_info['finished'] = True
                            self.active_trajectories[side] = None
                        elif dt >= times[-1] + 1.0:
                            # Abort on tolerance error
                            traj_info['aborted'] = True
                            self.active_trajectories[side] = None
                            self.get_logger().error(f"[FJT SERVER] Trajectory exceeded error tolerance limit: {final_error_max:.4f} rad.")
                            
                            # Safety action: Freeze at current measured q and clear desired velocity
                            if latest_state is not None:
                                for name in joints:
                                    if name in latest_state.name:
                                        idx = latest_state.name.index(name)
                                        self.q_desired[name] = latest_state.position[idx]
                                        self.dq_desired[name] = 0.0
                            
                    # 3. Between points
                    else:
                        for i in range(len(points) - 1):
                            if times[i] <= dt < times[i+1]:
                                t_start = times[i]
                                t_end = times[i+1]
                                denom = t_end - t_start
                                alpha = (dt - t_start) / denom if denom > 0.0 else 1.0
                                alpha = max(0.0, min(1.0, alpha)) # Defensive clamping
                                
                                for name in joints:
                                    q_start = points[i].positions[joint_indices[name]]
                                    q_end = points[i+1].positions[joint_indices[name]]
                                    self.q_desired[name] = (1 - alpha) * q_start + alpha * q_end
                                    
                                    has_vel_start = len(points[i].velocities) == len(trajectory.joint_names)
                                    has_vel_end = len(points[i+1].velocities) == len(trajectory.joint_names)
                                    if has_vel_start and has_vel_end:
                                        dq_start = points[i].velocities[joint_indices[name]]
                                        dq_end = points[i+1].velocities[joint_indices[name]]
                                        self.dq_desired[name] = (1 - alpha) * dq_start + alpha * dq_end
                                    else:
                                        self.dq_desired[name] = (q_end - q_start) / denom if denom > 0.0 else 0.0
                                break
                                
                    # Once per second FJT Debug Log
                    if t - self.last_fjt_debug_time[side] >= 1.0:
                        self.last_fjt_debug_time[side] = t
                        
                        elbow_joint = 'left_elbow_joint' if side == 'left' else 'right_elbow_joint'
                        idx_msg = latest_state.name.index(elbow_joint)
                        actual_elbow = latest_state.position[idx_msg]
                        desired_elbow = self.q_desired[elbow_joint]
                        
                        self.get_logger().info(
                            f"[FJT DEBUG] side={side} dt={dt:.2f}s duration={times[-1]:.2f}s "
                            f"desired_elbow={desired_elbow:.4f} actual_elbow={actual_elbow:.4f}"
                        )

        # Check finiteness of desired targets (NaN/Inf guard) before computing dynamics
        with self.lock:
            for side in ['left', 'right']:
                joints = self.left_joints if side == 'left' else self.right_joints
                finite_ok = True
                for name in joints:
                    if not math.isfinite(self.q_desired[name]) or not math.isfinite(self.dq_desired[name]):
                        finite_ok = False
                        break
                
                if not finite_ok:
                    self.get_logger().error(f"[FJT SAFETY] Invalid desired state detected (NaN/Inf) for {side} arm! Freezing desired state at current measured q.")
                    # Abort active trajectory if any
                    if self.active_trajectories[side] is not None:
                        self.active_trajectories[side]['aborted'] = True
                        self.active_trajectories[side] = None
                        
                    # Freeze at current position
                    if latest_state is not None:
                        for name in joints:
                            if name in latest_state.name:
                                idx = latest_state.name.index(name)
                                self.q_desired[name] = latest_state.position[idx]
                                self.dq_desired[name] = 0.0
                    else:
                        for name in joints:
                            self.q_desired[name] = 0.0
                            self.dq_desired[name] = 0.0

        # Pinocchio and command calculations
        q_full = pin.neutral(self.model)
        dq = np.zeros(self.model.nv)
        
        with self.lock:
            for i, name in enumerate(latest_state.name):
                if self.model.existJointName(name):
                    jid = self.model.getJointId(name)
                    joint = self.model.joints[jid]
                    if joint.nq == 1:
                        q_full[joint.idx_q] = latest_state.position[i]
                    if joint.nv == 1 and len(latest_state.velocity) > i:
                        dq[joint.idx_v] = latest_state.velocity[i]

        try:
            pin.computeGeneralizedGravity(self.model, self.data, q_full)
            tau_g = self.data.g
        except Exception as e:
            self.get_logger().error(f"[HOLD CONTROL] Pinocchio gravity computation error: {e}")
            self.publish_zeros()
            return

        with self.lock:
            # Compute Left Arm Commands
            tau_left = []
            max_sat_pct = 0.0
            log_data_l = []
            
            # Telemetry arrays for Left side
            telemetry_data_l = {
                'q_actual': [],
                'dq_actual': [],
                'q_desired': [],
                'dq_desired': [],
                'q_error': [],
                'tau_gravity': [],
                'tau_cmd_unsat': [],
                'tau_cmd_sat': [],
                'sat_pct': []
            }
            
            for name in self.left_joints:
                idx_msg = latest_state.name.index(name)
                q_actual = latest_state.position[idx_msg]
                dq_actual = latest_state.velocity[idx_msg] if len(latest_state.velocity) > idx_msg else 0.0
                q_target = self.q_desired[name]
                dq_target = self.dq_desired[name]
                
                jid = self.model.getJointId(name)
                idx_v = self.model.joints[jid].idx_v
                tau_g_j = tau_g[idx_v]
                
                kp, kd = self.get_joint_gains(name)
                if self.enable_pd:
                    tau_j = self.torque_sign * self.gravity_scale * tau_g_j + kp * (q_target - q_actual) + kd * (dq_target - dq_actual)
                else:
                    tau_j = self.torque_sign * self.gravity_scale * tau_g_j
                
                if math.isnan(tau_j) or math.isinf(tau_j):
                    self.get_logger().error(f"[HOLD CONTROL] Calculated NaN/Inf torque for joint {name}!")
                    self.publish_zeros()
                    return
                
                tau_sat, pct = self.saturate_torque(name, tau_j)
                max_sat_pct = max(max_sat_pct, pct)
                tau_left.append(tau_sat)
                log_data_l.append((q_actual, tau_g_j, tau_sat, q_target, q_target - q_actual))
                
                # Append left telemetry
                telemetry_data_l['q_actual'].append(q_actual)
                telemetry_data_l['dq_actual'].append(dq_actual)
                telemetry_data_l['q_desired'].append(q_target)
                telemetry_data_l['dq_desired'].append(dq_target)
                telemetry_data_l['q_error'].append(q_target - q_actual)
                telemetry_data_l['tau_gravity'].append(tau_g_j)
                telemetry_data_l['tau_cmd_unsat'].append(tau_j)
                telemetry_data_l['tau_cmd_sat'].append(tau_sat)
                telemetry_data_l['sat_pct'].append(pct)

            # Compute Right Arm Commands
            tau_right = []
            log_data_r = []
            
            # Telemetry arrays for Right side
            telemetry_data_r = {
                'q_actual': [],
                'dq_actual': [],
                'q_desired': [],
                'dq_desired': [],
                'q_error': [],
                'tau_gravity': [],
                'tau_cmd_unsat': [],
                'tau_cmd_sat': [],
                'sat_pct': []
            }
            
            for name in self.right_joints:
                idx_msg = latest_state.name.index(name)
                q_actual = latest_state.position[idx_msg]
                dq_actual = latest_state.velocity[idx_msg] if len(latest_state.velocity) > idx_msg else 0.0
                q_target = self.q_desired[name]
                dq_target = self.dq_desired[name]
                
                jid = self.model.getJointId(name)
                idx_v = self.model.joints[jid].idx_v
                tau_g_j = tau_g[idx_v]
                
                kp, kd = self.get_joint_gains(name)
                if self.enable_pd:
                    tau_j = self.torque_sign * self.gravity_scale * tau_g_j + kp * (q_target - q_actual) + kd * (dq_target - dq_actual)
                else:
                    tau_j = self.torque_sign * self.gravity_scale * tau_g_j
                
                if math.isnan(tau_j) or math.isinf(tau_j):
                    self.get_logger().error(f"[HOLD CONTROL] Calculated NaN/Inf torque for joint {name}!")
                    self.publish_zeros()
                    return
                
                tau_sat, pct = self.saturate_torque(name, tau_j)
                max_sat_pct = max(max_sat_pct, pct)
                tau_right.append(tau_sat)
                log_data_r.append((q_actual, tau_g_j, tau_sat, q_target, q_target - q_actual))
                
                # Append right telemetry
                telemetry_data_r['q_actual'].append(q_actual)
                telemetry_data_r['dq_actual'].append(dq_actual)
                telemetry_data_r['q_desired'].append(q_target)
                telemetry_data_r['dq_desired'].append(dq_target)
                telemetry_data_r['q_error'].append(q_target - q_actual)
                telemetry_data_r['tau_gravity'].append(tau_g_j)
                telemetry_data_r['tau_cmd_unsat'].append(tau_j)
                telemetry_data_r['tau_cmd_sat'].append(tau_sat)
                telemetry_data_r['sat_pct'].append(pct)

        # Publish MultiArrays
        msg_l = Float64MultiArray()
        msg_l.data = tau_left
        self.left_pub.publish(msg_l)

        msg_r = Float64MultiArray()
        msg_r.data = tau_right
        self.right_pub.publish(msg_r)

        # Publish hold_ready
        with self.lock:
            if self.recaptured_flag:
                self.recaptured_flag = False
                self.hold_ready_state = True
            ready_state = self.hold_ready_state

        ready_msg = Bool()
        ready_msg.data = ready_state
        self.ready_pub.publish(ready_msg)

        # Publish Telemetry if enabled and rate limit reached
        if self.telemetry_enabled:
            if (t - self.last_telemetry_pub_time) >= (1.0 / self.telemetry_rate_hz) - 1e-4:
                self.last_telemetry_pub_time = t
                with self.lock:
                    left_active_traj = self.active_trajectories['left'] is not None
                    right_active_traj = self.active_trajectories['right'] is not None
                
                # Left Side Telemetry
                json_data_l = {
                    'stamp': t,
                    'config_id': self.telemetry_config_id,
                    'side': 'left',
                    'active_trajectory': left_active_traj,
                    'joint_names': self.left_joints,
                    'q_desired': telemetry_data_l['q_desired'],
                    'q_actual': telemetry_data_l['q_actual'],
                    'dq_desired': telemetry_data_l['dq_desired'],
                    'dq_actual': telemetry_data_l['dq_actual'],
                    'q_error': telemetry_data_l['q_error'],
                    'tau_gravity': telemetry_data_l['tau_gravity'],
                    'tau_cmd_unsat': telemetry_data_l['tau_cmd_unsat'],
                    'tau_cmd_sat': telemetry_data_l['tau_cmd_sat'],
                    'sat_pct': telemetry_data_l['sat_pct']
                }
                # Right Side Telemetry
                json_data_r = {
                    'stamp': t,
                    'config_id': self.telemetry_config_id,
                    'side': 'right',
                    'active_trajectory': right_active_traj,
                    'joint_names': self.right_joints,
                    'q_desired': telemetry_data_r['q_desired'],
                    'q_actual': telemetry_data_r['q_actual'],
                    'dq_desired': telemetry_data_r['dq_desired'],
                    'dq_actual': telemetry_data_r['dq_actual'],
                    'q_error': telemetry_data_r['q_error'],
                    'tau_gravity': telemetry_data_r['tau_gravity'],
                    'tau_cmd_unsat': telemetry_data_r['tau_cmd_unsat'],
                    'tau_cmd_sat': telemetry_data_r['tau_cmd_sat'],
                    'sat_pct': telemetry_data_r['sat_pct']
                }
                
                # Publish std_msgs/String
                from std_msgs.msg import String
                msg_str_l = String()
                msg_str_l.data = json.dumps(json_data_l)
                self.telemetry_pub.publish(msg_str_l)
                
                msg_str_r = String()
                msg_str_r.data = json.dumps(json_data_r)
                self.telemetry_pub.publish(msg_str_r)

        # Telemetry Log at 1Hz
        if (t - self.last_log_time) >= (1.0 / self.log_rate_hz):
            self.last_log_time = t
            self.get_logger().info(
                f"[HOLD CONTROL STATE]\n"
                f"  Hold Active: {self.hold_active}\n"
                f"  Max Saturation: {max_sat_pct:.1f}%\n"
                f"  Left Desired Q: {[round(x[3], 3) for x in log_data_l]}\n"
                f"  Left Actual  Q: {[round(x[0], 3) for x in log_data_l]}\n"
                f"  Left Error   Q: {[round(x[4], 3) for x in log_data_l]}\n"
                f"  Left Arm G:     {[round(x[1], 3) for x in log_data_l]}\n"
                f"  Left Arm T:     {[round(x[2], 3) for x in log_data_l]}\n"
                f"  Right Desired Q: {[round(x[3], 3) for x in log_data_r]}\n"
                f"  Right Actual  Q: {[round(x[0], 3) for x in log_data_r]}\n"
                f"  Right Error   Q: {[round(x[4], 3) for x in log_data_r]}\n"
                f"  Right Arm G:     {[round(x[1], 3) for x in log_data_r]}\n"
                f"  Right Arm T:     {[round(x[2], 3) for x in log_data_r]}"
            )

    def shutdown(self):
        if self.publish_zero_on_shutdown:
            try:
                self.publish_zeros()
                self.get_logger().warn("[HOLD CONTROL] Published zero torques on shutdown.")
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = H12DynamicsHoldController()
    
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
