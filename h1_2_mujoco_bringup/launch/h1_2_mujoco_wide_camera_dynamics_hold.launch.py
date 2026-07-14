#!/usr/bin/env python3
"""
Launch de bringup del robot H1-2 en MuJoCo con controladores dinámicos.

Configura la simulación física, carga la descripción cinemática y dinámica,
inicia robot_state_publisher, ros2_control y spawnea los controladores de
esfuerzo, torso y manos. Además, arranca el controlador dinámico de retención.
"""

from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
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

    # ============================================================
    # Carga de la descripción del robot (URDF/Xacro)
    # ============================================================
    # ============================================================
    # Carga de la descripción del robot (URDF/Xacro)
    # ============================================================
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

    # ============================================================
    # Publicador del estado del robot (TF y descripción)
    # ============================================================
    # ============================================================
    # Publicador del estado del robot (TF y descripción)
    # ============================================================
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # ============================================================
    # Nodo puente ros2_control para MuJoCo
    # ============================================================
    # ============================================================
    # Nodo puente ros2_control para MuJoCo
    # ============================================================
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

    # Launch arguments
    # ============================================================
    # Parámetros y ganancias del controlador dinámico
    # ============================================================
    # ============================================================
    # Parámetros y ganancias del controlador dinámico
    # ============================================================
    control_rate_hz_arg = DeclareLaunchArgument('control_rate_hz', default_value='100.0')
    gravity_scale_arg = DeclareLaunchArgument('gravity_scale', default_value='1.0')
    torque_sign_arg = DeclareLaunchArgument('torque_sign', default_value='1.0')
    enable_pd_arg = DeclareLaunchArgument('enable_pd', default_value='true')
    hold_capture_delay_sec_arg = DeclareLaunchArgument('hold_capture_delay_sec', default_value='0.5')
    
    kp_shoulder_pitch_roll_arg = DeclareLaunchArgument('kp_shoulder_pitch_roll', default_value='80.0')
    kd_shoulder_pitch_roll_arg = DeclareLaunchArgument('kd_shoulder_pitch_roll', default_value='10.0')
    kp_shoulder_yaw_arg = DeclareLaunchArgument('kp_shoulder_yaw', default_value='30.0')
    kd_shoulder_yaw_arg = DeclareLaunchArgument('kd_shoulder_yaw', default_value='4.0')
    kp_elbow_arg = DeclareLaunchArgument('kp_elbow', default_value='120.0')
    kd_elbow_arg = DeclareLaunchArgument('kd_elbow', default_value='14.0')
    kp_wrist_arg = DeclareLaunchArgument('kp_wrist', default_value='40.0')
    kd_wrist_arg = DeclareLaunchArgument('kd_wrist', default_value='5.0')
    
    telemetry_enabled_arg = DeclareLaunchArgument('telemetry_enabled', default_value='false')
    telemetry_rate_hz_arg = DeclareLaunchArgument('telemetry_rate_hz', default_value='50.0')
    telemetry_config_id_arg = DeclareLaunchArgument('telemetry_config_id', default_value='C2_nominal')

    # ============================================================
    # Nodo controlador de retención dinámica por esfuerzo articular
    # ============================================================
    # ============================================================
    # Nodo controlador de retención dinámica por esfuerzo articular
    # ============================================================
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
                    "control_rate_hz": LaunchConfiguration("control_rate_hz"),
                    "gravity_scale": LaunchConfiguration("gravity_scale"),
                    "torque_sign": LaunchConfiguration("torque_sign"),
                    "enable_pd": LaunchConfiguration("enable_pd"),
                    "hold_capture_delay_sec": LaunchConfiguration("hold_capture_delay_sec"),
                    "kp_shoulder_pitch_roll": LaunchConfiguration("kp_shoulder_pitch_roll"),
                    "kd_shoulder_pitch_roll": LaunchConfiguration("kd_shoulder_pitch_roll"),
                    "kp_shoulder_yaw": LaunchConfiguration("kp_shoulder_yaw"),
                    "kd_shoulder_yaw": LaunchConfiguration("kd_shoulder_yaw"),
                    "kp_elbow": LaunchConfiguration("kp_elbow"),
                    "kd_elbow": LaunchConfiguration("kd_elbow"),
                    "kp_wrist": LaunchConfiguration("kp_wrist"),
                    "kd_wrist": LaunchConfiguration("kd_wrist"),
                    "telemetry_enabled": LaunchConfiguration("telemetry_enabled"),
                    "telemetry_rate_hz": LaunchConfiguration("telemetry_rate_hz"),
                    "telemetry_config_id": LaunchConfiguration("telemetry_config_id"),
                }],
            )
        ],
    )

    return LaunchDescription([
        control_rate_hz_arg,
        gravity_scale_arg,
        torque_sign_arg,
        enable_pd_arg,
        hold_capture_delay_sec_arg,
        kp_shoulder_pitch_roll_arg,
        kd_shoulder_pitch_roll_arg,
        kp_shoulder_yaw_arg,
        kd_shoulder_yaw_arg,
        kp_elbow_arg,
        kd_elbow_arg,
        kp_wrist_arg,
        kd_wrist_arg,
        telemetry_enabled_arg,
        telemetry_rate_hz_arg,
        telemetry_config_id_arg,

        robot_state_publisher,
        control_node,

        # ============================================================
        # Secuencia temporal de carga de controladores ros2_control.
        # ============================================================
        # Los retardos reducen condiciones de carrera durante el arranque,
        # pero no constituyen una comprobación determinista de disponibilidad.
        controller_spawner("joint_state_broadcaster", 2.0),
        controller_spawner("left_arm_effort_controller", 3.0),
        controller_spawner("right_arm_effort_controller", 3.5),

        hold_controller_node,

        controller_spawner("torso_controller", 5.5),
        controller_spawner("left_hand_controller", 6.0),
        controller_spawner("right_hand_controller", 6.5),
    ])
