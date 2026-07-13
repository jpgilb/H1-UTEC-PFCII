#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def controller_spawner(controller_name, delay):
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    controller_name,
                    "--controller-manager",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "60",
                ],
                output="screen",
            )
        ],
    )


def generate_launch_description():
    pkg = FindPackageShare("h1_2_mujoco_bringup")

    robot_description_content = ParameterValue(
        Command([
            "xacro ",
            PathJoinSubstitution([
                pkg,
                "description",
                "h1_2_mujoco_wide_shelves_camera_dynamics.urdf.xacro",
            ]),
        ]),
        value_type=str,
    )

    robot_description = {
        "robot_description": robot_description_content,
        "use_sim_time": True,
    }

    controllers_yaml = PathJoinSubstitution([
        pkg,
        "config",
        "ros2_controllers_dynamics.yaml",
    ])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            robot_description,
            controllers_yaml,
            {"use_sim_time": True},
        ],
    )

    # Launch our hold controller script after the arm controllers are spawned
    hold_controller_node = TimerAction(
        period=4.5,
        actions=[
            Node(
                package="h1_2_mujoco_bringup",
                executable="h1_2_dynamics_hold_controller.py",
                name="h1_2_dynamics_hold_controller",
                output="screen",
                parameters=[{
                    "pinocchio_urdf_path": PathJoinSubstitution([
                        pkg,
                        "description",
                        "generated",
                        "h1_2_pinocchio_dynamics.urdf",
                    ]),
                    "control_rate_hz": 100.0,
                    "gravity_scale": 1.0,
                    "torque_sign": 1.0,
                    "enable_pd": True,
                    "hold_capture_delay_sec": 0.5,
                }],
            )
        ],
    )

    # 1. Early pause node: Calls set_pause(paused=True) as soon as the service is online
    pause_first_node = Node(
        package="h1_2_mujoco_bringup",
        executable="mujoco_pause_until_hold_ready.py",
        name="mujoco_pause_first",
        arguments=["--pause-first"],
        output="screen",
    )

    # 2. Unpause orchestrator node: Wait for hold_ready=True and calls set_pause(paused=False)
    unpause_node = Node(
        package="h1_2_mujoco_bringup",
        executable="mujoco_pause_until_hold_ready.py",
        name="mujoco_unpause_when_hold_ready",
        arguments=["--unpause-when-hold-ready"],
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        control_node,

        # Spawning sequence with optimized delay
        controller_spawner("joint_state_broadcaster", 2.0),
        controller_spawner("left_arm_effort_controller", 3.0),
        controller_spawner("right_arm_effort_controller", 3.5),
        
        hold_controller_node,
        
        controller_spawner("torso_controller", 5.5),
        controller_spawner("left_hand_controller", 6.0),
        controller_spawner("right_hand_controller", 6.5),
        
        # Pausing & Unpausing Orchestrators
        pause_first_node,
        unpause_node,
    ])
