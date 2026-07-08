#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # El nodo ejecutable 'mover_brazo_single.py' está instalado por el paquete 'h1_2_moveit_config'.
    package_name = 'h1_2_moveit_config'

    # WARNING: This demo uses a deterministic static objeto_esfera TF.
    # Do not run object_tracker simultaneously if it publishes the same frame, as publishing the same frame from two sources may create TF conflicts.

    declared_arguments = [
        # General parameters
        DeclareLaunchArgument(
            'arm_side',
            default_value='left',
            description='Lado del brazo a usar (left o right)'
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='pelvis',
            description='Frame base de referencia del robot'
        ),
        DeclareLaunchArgument(
            'object_frame',
            default_value='objeto_esfera',
            description='Frame TF2 del objeto a manipular'
        ),
        DeclareLaunchArgument(
            'object_type',
            default_value='sphere',
            description='Tipo de objeto (cube o sphere)'
        ),
        DeclareLaunchArgument(
            'object_dimension',
            default_value='0.025',
            description='Lado del cubo o diámetro de la esfera [m]'
        ),
        DeclareLaunchArgument(
            'enable_object_queue',
            default_value='false',
            description='Habilitar cola de objetos'
        ),

        # Task parameters
        DeclareLaunchArgument(
            'task_mode',
            default_value='bimanual_transfer',
            description='Modo de la tarea a ejecutar (e.g. bimanual_transfer)'
        ),
        DeclareLaunchArgument(
            'manipulation_geometry_mode',
            default_value='table',
            description='Geometría del pick (table o shelf)'
        ),
        DeclareLaunchArgument(
            'place_geometry_mode',
            default_value='table',
            description='Geometría del place (table o shelf)'
        ),
        DeclareLaunchArgument(
            'table_transfer_y',
            default_value='0.0',
            description='Coordenada Y para la transferencia intermedia en mesa'
        ),

        # Table & ArUco parameters
        DeclareLaunchArgument(
            'table_collision',
            default_value='true',
            description='Activar colisión con la mesa de trabajo'
        ),
        DeclareLaunchArgument(
            'use_aruco_table_target',
            default_value='false',
            description='Usar ArUco como destino de colocación intermedia en mesa'
        ),
        DeclareLaunchArgument(
            'use_aruco_table_height',
            default_value='false',
            description='Usar ArUco para ajustar dinámicamente la altura de colisión de la mesa'
        ),
        DeclareLaunchArgument(
            'object_pose_reference',
            default_value='center',
            description='Referencia de la pose del objeto (center o top_face_center)'
        ),

        # Motion & Cartesian parameters
        DeclareLaunchArgument(
            'use_cartesian_local_motions',
            default_value='false',
            description='Usar movimientos cartesianos locales para aproximación y retiro'
        ),
        DeclareLaunchArgument(
            'use_split_place_retreat',
            default_value='false',
            description='Usar retiro segmentado en fase de place'
        ),
        DeclareLaunchArgument(
            'phase4_motion_mode',
            default_value='auto',
            description='Modo de movimiento para la aproximación (cartesian, ompl o auto)'
        ),
        DeclareLaunchArgument(
            'min_cartesian_fraction',
            default_value='0.95',
            description='Fracción mínima para dar por válido un path cartesiano'
        ),

        # Debug & Settle parameters
        DeclareLaunchArgument(
            'debug_disable_object_collision',
            default_value='false',
            description='Desactivar colisiones del objeto con el mundo'
        ),
        DeclareLaunchArgument(
            'collision_test_assertions_enabled',
            default_value='false',
            description='Activar aserciones y detener ejecución en fallos de colisión'
        ),
        DeclareLaunchArgument(
            'post_motion_settle_time',
            default_value='2.0',
            description='Tiempo de espera tras movimiento para estabilización de física [s]'
        ),
        DeclareLaunchArgument(
            'debug_stop_after_phase',
            default_value='0',
            description='Detener el nodo después de una fase específica (0 = no detener)'
        ),
        DeclareLaunchArgument(
            'abort_on_invalid_state_before_motion',
            default_value='false',
            description='Abortar ejecución si el estado del robot es inválido antes de mover'
        ),
        DeclareLaunchArgument(
            'validate_tcp_before_detach',
            default_value='false',
            description='Validar alineación de TCP antes de soltar el objeto'
        ),
        DeclareLaunchArgument(
            'confirm_hand_open_before_detach',
            default_value='false',
            description='Confirmar apertura física de la mano antes de desasociar en MoveIt'
        ),

        # Phase 8 parameters
        DeclareLaunchArgument(
            'phase8_motion_mode',
            default_value='auto',
            description='Modo de movimiento para Phase 8 (cartesian, ompl o auto)'
        ),
        DeclareLaunchArgument(
            'phase8_transit_z',
            default_value='0.28',
            description='Altura Z de tránsito libre durante Phase 8 [m]'
        ),
        DeclareLaunchArgument(
            'phase8_min_cartesian_fraction',
            default_value='0.95',
            description='Fracción cartesiana mínima requerida en Phase 8'
        ),
        DeclareLaunchArgument(
            'phase8_staged_max_segment_distance',
            default_value='0.08',
            description='Distancia máxima de segmento para planificador OMPL segmentado en Phase 8 [m]'
        ),

        # Environment Scene parameters
        DeclareLaunchArgument(
            'environment_scene_enabled',
            default_value='true',
            description='Habilitar modelado de obstáculos en PlanningScene'
        ),
        DeclareLaunchArgument(
            'environment_profile',
            default_value='env_v1',
            description='Perfil del entorno a cargar (table_only o env_v1)'
        ),
        DeclareLaunchArgument(
            'environment_add_table_to_planning_scene',
            default_value='true',
            description='Agregar mesa de trabajo a la PlanningScene'
        ),
        DeclareLaunchArgument(
            'environment_add_shelves_to_planning_scene',
            default_value='true',
            description='Agregar estanterías a la PlanningScene'
        ),
    ]

    mover_brazo_single_node = Node(
        package=package_name,
        executable='mover_brazo_single.py',
        name='single_arm_face_approach_grasp_node',
        output='screen',
        parameters=[{
            'arm_side': LaunchConfiguration('arm_side'),
            'base_frame': LaunchConfiguration('base_frame'),
            'object_frame': LaunchConfiguration('object_frame'),
            'object_type': LaunchConfiguration('object_type'),
            'object_dimension': LaunchConfiguration('object_dimension'),
            'enable_object_queue': LaunchConfiguration('enable_object_queue'),
            'task_mode': LaunchConfiguration('task_mode'),
            'manipulation_geometry_mode': LaunchConfiguration('manipulation_geometry_mode'),
            'place_geometry_mode': LaunchConfiguration('place_geometry_mode'),
            'table_transfer_y': LaunchConfiguration('table_transfer_y'),
            'table_collision': LaunchConfiguration('table_collision'),
            'use_aruco_table_target': LaunchConfiguration('use_aruco_table_target'),
            'use_aruco_table_height': LaunchConfiguration('use_aruco_table_height'),
            'object_pose_reference': LaunchConfiguration('object_pose_reference'),
            'use_cartesian_local_motions': LaunchConfiguration('use_cartesian_local_motions'),
            'use_split_place_retreat': LaunchConfiguration('use_split_place_retreat'),
            'phase4_motion_mode': LaunchConfiguration('phase4_motion_mode'),
            'min_cartesian_fraction': LaunchConfiguration('min_cartesian_fraction'),
            'debug_disable_object_collision': LaunchConfiguration('debug_disable_object_collision'),
            'collision_test_assertions_enabled': LaunchConfiguration('collision_test_assertions_enabled'),
            'post_motion_settle_time': LaunchConfiguration('post_motion_settle_time'),
            'debug_stop_after_phase': LaunchConfiguration('debug_stop_after_phase'),
            'abort_on_invalid_state_before_motion': LaunchConfiguration('abort_on_invalid_state_before_motion'),
            'validate_tcp_before_detach': LaunchConfiguration('validate_tcp_before_detach'),
            'confirm_hand_open_before_detach': LaunchConfiguration('confirm_hand_open_before_detach'),
            'phase8_motion_mode': LaunchConfiguration('phase8_motion_mode'),
            'phase8_transit_z': LaunchConfiguration('phase8_transit_z'),
            'phase8_min_cartesian_fraction': LaunchConfiguration('phase8_min_cartesian_fraction'),
            'phase8_staged_max_segment_distance': LaunchConfiguration('phase8_staged_max_segment_distance'),
            'environment_scene_enabled': LaunchConfiguration('environment_scene_enabled'),
            'environment_profile': LaunchConfiguration('environment_profile'),
            'environment_add_table_to_planning_scene': LaunchConfiguration('environment_add_table_to_planning_scene'),
            'environment_add_shelves_to_planning_scene': LaunchConfiguration('environment_add_shelves_to_planning_scene'),
        }]
    )

    # Static transform publisher node to publish object_frame (objeto_esfera) deterministically
    static_tf_pub_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_objeto_esfera_publisher',
        arguments=['0.5', '0.30', '0.08', '0', '0', '0', '1', 'pelvis', 'objeto_esfera']
    )

    return LaunchDescription(declared_arguments + [
        mover_brazo_single_node,
        static_tf_pub_node
    ])
