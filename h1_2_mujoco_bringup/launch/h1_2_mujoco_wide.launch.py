#!/usr/bin/env python3

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
                "h1_2_mujoco_wide_shelves.urdf.xacro",
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
        "ros2_controllers_mujoco.yaml",
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

    return LaunchDescription([
        robot_state_publisher,
        control_node,

        # Spawn controllers sequentially after MuJoCo/controller_manager starts.
        controller_spawner("joint_state_broadcaster", 6.0),
        controller_spawner("left_arm_controller", 9.0),
        controller_spawner("right_arm_controller", 11.0),
        controller_spawner("torso_controller", 13.0),
        controller_spawner("left_hand_controller", 15.0),
        controller_spawner("right_hand_controller", 17.0),
    ])
