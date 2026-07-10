#!/usr/bin/env python3

import sys
import argparse
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from mujoco_ros2_control_msgs.srv import SetPause, ResetWorld

class MujocoPauseManager(Node):
    def __init__(self, mode: str):
        super().__init__('mujoco_pause_manager')
        self.mode = mode
        
        self.pause_client = self.create_client(SetPause, '/mujoco_ros2_control_node/set_pause')
        self.reset_client = self.create_client(ResetWorld, '/mujoco_ros2_control_node/reset_world')
        self.recapture_client = self.create_client(Trigger, '/h1_2_dynamics_hold_controller/recapture_hold')
        
        self.hold_ready_received = False
        self.state = "WAITING_INITIAL_HOLD"

        if self.mode == 'pause':
            self.get_logger().info("[PAUSE MANAGER] Mode: Pause simulation on start.")
            self.pause_simulation()
        elif self.mode == 'unpause':
            self.get_logger().info("[PAUSE MANAGER] Mode: Unpause when hold_ready is True.")
            self.hold_ready_sub = self.create_subscription(
                Bool,
                '/h1_2_dynamics_hold_controller/hold_ready',
                self.hold_ready_callback,
                10
            )
        elif self.mode == 'reset_after_hold_ready':
            self.get_logger().info("[PAUSE MANAGER] Mode: Reset after initial hold ready.")
            self.wait_for_services()
            self.hold_ready_sub = self.create_subscription(
                Bool,
                '/h1_2_dynamics_hold_controller/hold_ready',
                self.hold_ready_callback,
                10
            )

    def wait_for_services(self):
        self.get_logger().info("[PAUSE MANAGER] Waiting for required services to be online...")
        for name, client in [
            ('/mujoco_ros2_control_node/set_pause', self.pause_client),
            ('/mujoco_ros2_control_node/reset_world', self.reset_client),
            ('/h1_2_dynamics_hold_controller/recapture_hold', self.recapture_client)
        ]:
            if not client.wait_for_service(timeout_sec=15.0):
                self.get_logger().error(f"[PAUSE MANAGER] Service '{name}' not available! Exiting.")
                sys.exit(1)
        self.get_logger().info("[PAUSE MANAGER] All required services are online.")

    # Mode 1: --pause-first
    def pause_simulation(self):
        self.get_logger().info("[PAUSE MANAGER] Waiting for SetPause service...")
        if not self.pause_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("[PAUSE MANAGER] SetPause service not available! Exiting.")
            sys.exit(1)
            
        req = SetPause.Request()
        req.paused = True
        self.get_logger().info("[PAUSE MANAGER] Calling SetPause(paused=True)...")
        future = self.pause_client.call_async(req)
        future.add_done_callback(self.pause_done_callback)

    def pause_done_callback(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"[PAUSE MANAGER] Simulation PAUSED successfully: {res.message}")
                sys.exit(0)
            else:
                self.get_logger().error(f"[PAUSE MANAGER] Failed to pause simulation: {res.message}")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"[PAUSE MANAGER] Service call failed: {e}")
            sys.exit(1)

    # General callback for hold_ready subscription
    def hold_ready_callback(self, msg: Bool):
        if self.mode == 'unpause':
            if msg.data and not self.hold_ready_received:
                self.hold_ready_received = True
                self.get_logger().info("[PAUSE MANAGER] hold_ready is TRUE. Proceeding to unpause...")
                self.unpause_simulation()
                
        elif self.mode == 'reset_after_hold_ready':
            if self.state == "WAITING_INITIAL_HOLD" and msg.data:
                self.state = "EXECUTING_SEQUENCE"
                self.get_logger().info("[PAUSE MANAGER] Initial hold ready. Starting pause-reset-recapture sequence...")
                self.seq_step_1_pause()
            elif self.state == "WAITING_RECAPTURED_HOLD" and msg.data:
                self.state = "EXECUTING_UNPAUSE"
                self.get_logger().info("[PAUSE MANAGER] Hold recaptured successfully. Waiting 0.5 s before final unpause...")
                time.sleep(0.5)
                self.seq_step_4_unpause()

    # Mode 2: --unpause-when-hold-ready
    def unpause_simulation(self):
        self.get_logger().info("[PAUSE MANAGER] Waiting for SetPause service to unpause...")
        if not self.pause_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("[PAUSE MANAGER] SetPause service not available for unpausing!")
            sys.exit(1)
            
        req = SetPause.Request()
        req.paused = False
        self.get_logger().info("[PAUSE MANAGER] Calling SetPause(paused=False)...")
        future = self.pause_client.call_async(req)
        future.add_done_callback(self.unpause_done_callback)

    def unpause_done_callback(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"[PAUSE MANAGER] Simulation UNPAUSED successfully: {res.message}")
                sys.exit(0)
            else:
                self.get_logger().error(f"[PAUSE MANAGER] Failed to unpause simulation: {res.message}")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"[PAUSE MANAGER] Service call failed: {e}")
            sys.exit(1)

    # Mode 3: --reset-after-hold-ready sequence callbacks
    def seq_step_1_pause(self):
        req = SetPause.Request()
        req.paused = True
        future = self.pause_client.call_async(req)
        future.add_done_callback(self.seq_step_1_pause_done)

    def seq_step_1_pause_done(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info("[STARTUP RECOVERY] Paused after initial hold.")
                self.seq_step_2_reset()
            else:
                self.get_logger().error(f"[STARTUP RECOVERY] Failed to pause: {res.message}")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"[STARTUP RECOVERY] Pause service call failed: {e}")
            sys.exit(1)

    def seq_step_2_reset(self):
        req = ResetWorld.Request()
        req.keyframe = ''
        future = self.reset_client.call_async(req)
        future.add_done_callback(self.seq_step_2_reset_done)

    def seq_step_2_reset_done(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info("[STARTUP RECOVERY] World reset to initial state.")
                self.seq_step_3_recapture()
            else:
                self.get_logger().error(f"[STARTUP RECOVERY] Failed to reset world: {res.message}")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"[STARTUP RECOVERY] ResetWorld service call failed: {e}")
            sys.exit(1)

    def seq_step_3_recapture(self):
        req = Trigger.Request()
        future = self.recapture_client.call_async(req)
        future.add_done_callback(self.seq_step_3_recapture_done)

    def seq_step_3_recapture_done(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info("[STARTUP RECOVERY] Hold recaptured.")
                self.get_logger().info("[STARTUP RECOVERY] Waiting 0.2 s before final unpause...")
                time.sleep(0.2)
                self.seq_step_4_unpause()
            else:
                self.get_logger().error(f"[STARTUP RECOVERY] Failed to recapture hold: {res.message}")
                self.get_logger().warn("[STARTUP RECOVERY] Unpausing anyway to avoid leaving MuJoCo frozen.")
                self.seq_step_4_unpause()
        except Exception as e:
            self.get_logger().error(f"[STARTUP RECOVERY] Recapture service call failed: {e}")
            sys.exit(1)

    def seq_step_4_unpause(self):
        req = SetPause.Request()
        req.paused = False
        future = self.pause_client.call_async(req)
        future.add_done_callback(self.seq_step_4_unpause_done)

    def seq_step_4_unpause_done(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info("[STARTUP RECOVERY] Simulation unpaused after recaptured hold.")
                sys.exit(0)
            else:
                self.get_logger().error(f"[STARTUP RECOVERY] Failed to unpause simulation: {res.message}")
                sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"[STARTUP RECOVERY] Unpause service call failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Pause/Unpause/Reset Manager")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pause-first', action='store_true', help="Pause the simulation immediately")
    group.add_argument('--unpause-when-hold-ready', action='store_true', help="Unpause when hold controller is ready")
    group.add_argument('--reset-after-hold-ready', action='store_true', help="Reset and recapture hold after initial hold ready")

    # Important:
    # launch_ros adds ROS arguments such as:
    #   --ros-args -r __node:=...
    # argparse must ignore those and rclpy must receive them.
    args, ros_args = parser.parse_known_args(args=sys.argv[1:])

    if args.pause_first:
        mode = 'pause'
    elif args.unpause_when_hold_ready:
        mode = 'unpause'
    elif args.reset_after_hold_ready:
        mode = 'reset_after_hold_ready'
    else:
        raise RuntimeError("No valid mode selected.")

    rclpy.init(args=ros_args)
    node = MujocoPauseManager(mode)

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
