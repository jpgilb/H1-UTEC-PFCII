#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # =========================================================================
    # CONFIGURACIÓN DEL PAQUETE ROS 2
    # =========================================================================
    # TODO: reemplazar por el nombre real del paquete del proyecto si es diferente.
    # Por defecto se utiliza 'h1_2_moveit_config', que es el paquete MoveIt de la simulación.
    package_name = 'h1_2_moveit_config'

    # Declaración de argumentos de lanzamiento para poder sobrescribirlos desde consola
    declared_arguments = [
        # ---------------------------------------------------------------------
        # Parámetros generales
        # ---------------------------------------------------------------------
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
            default_value='objeto_cubo',
            description='Frame TF2 del objeto a manipular'
        ),
        DeclareLaunchArgument(
            'object_type',
            default_value='cube',
            description='Tipo de objeto (cube o sphere)'
        ),
        DeclareLaunchArgument(
            'object_dimension',
            default_value='0.055',
            description='Lado del cubo o diámetro de la esfera [m]'
        ),

        # ---------------------------------------------------------------------
        # Modo de tarea
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'task_mode',
            default_value='bimanual_transfer',
            description='Modo de la tarea a ejecutar (e.g. bimanual_transfer)'
        ),
        DeclareLaunchArgument(
            'manipulation_geometry_mode',
            default_value='shelf',
            description='Geometría del pick (table o shelf)'
        ),
        DeclareLaunchArgument(
            'place_geometry_mode',
            default_value='shelf',
            description='Geometría del place (table o shelf)'
        ),
        DeclareLaunchArgument(
            'phase4_motion_mode',
            default_value='ompl',
            description='Modo de movimiento para la aproximación (cartesian u ompl)'
        ),

        # ---------------------------------------------------------------------
        # Parámetros ArUco
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'use_aruco_table_target',
            default_value='true',
            description='Usar ArUco como destino de colocación intermedia en mesa'
        ),
        DeclareLaunchArgument(
            'table_target_frame',
            default_value='aruco_mesa',
            description='Frame de referencia ArUco en mesa'
        ),
        DeclareLaunchArgument(
            'table_target_offset_x',
            default_value='0.0',
            description='Offset en X desde el ArUco para colocar el objeto'
        ),
        DeclareLaunchArgument(
            'table_target_offset_y',
            default_value='0.0',
            description='Offset en Y desde el ArUco para colocar el objeto'
        ),
        DeclareLaunchArgument(
            'table_target_offset_z',
            default_value='0.0',
            description='Offset en Z desde el ArUco para colocar el objeto'
        ),
        DeclareLaunchArgument(
            'table_target_offset_mode',
            default_value='base',
            description='Modo de offset (base o marker_yaw)'
        ),
        DeclareLaunchArgument(
            'aruco_z_is_table_surface',
            default_value='true',
            description='Si es verdadero, la coordenada Z representa la superficie de la mesa'
        ),

        # ---------------------------------------------------------------------
        # Referencia geométrica del objeto
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'object_pose_reference',
            default_value='top_face_center',
            description='Referencia de la pose del objeto (center o top_face_center)'
        ),

        # ---------------------------------------------------------------------
        # Altura dinámica de mesa
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'use_aruco_table_height',
            default_value='true',
            description='Usar ArUco para ajustar dinámicamente la altura de colisión de la mesa'
        ),
        DeclareLaunchArgument(
            'aruco_table_height_offset',
            default_value='-0.001',
            description='Offset de altura de mesa respecto al ArUco [m]'
        ),
        DeclareLaunchArgument(
            'allow_cached_aruco_table_height',
            default_value='false',
            description='Permitir el uso de la altura cacheada del ArUco si se pierde visión'
        ),

        # ---------------------------------------------------------------------
        # Parámetros de escena y diagnóstico
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'table_collision',
            default_value='true',
            description='Activar colisión con la mesa de trabajo'
        ),
        DeclareLaunchArgument(
            'publish_debug_markers',
            default_value='true',
            description='Publicar marcadores de depuración visual en RViz'
        ),
        DeclareLaunchArgument(
            'validate_aruco_table_target_ik',
            default_value='false',
            description='Validar cinemática (IK) para la pose calculada desde ArUco antes de moverse'
        ),

        # ---------------------------------------------------------------------
        # Parámetros de agarre y movimiento relevantes
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'shelf_pregrasp_distance',
            default_value='-1.0',
            description='Distancia frontal de aproximación/retirada para estantes (-1.0 para usar approach_distance)'
        ),
        DeclareLaunchArgument(
            'shelf_surface_clearance',
            default_value='-1.0',
            description='Claro de superficie mínimo para estantes (-1.0 para usar surface_clearance)'
        ),
        DeclareLaunchArgument(
            'shelf_skip_vertical_descent',
            default_value='true',
            description='Saltar el descenso vertical en modo estante (shelf)'
        ),
        DeclareLaunchArgument(
            'shelf_lift_before_retreat_enabled',
            default_value='true',
            description='Habilitar micro-lift vertical antes de retirada frontal en estantes'
        ),
        DeclareLaunchArgument(
            'shelf_lift_before_retreat_distance',
            default_value='0.010',
            description='Distancia vertical del micro-lift [m]'
        ),
        DeclareLaunchArgument(
            'shelf_lift_before_retreat_motion_mode',
            default_value='cartesian',
            description='Modo de movimiento para el micro-lift (cartesian u ompl)'
        ),
        DeclareLaunchArgument(
            'shelf_retreat_keep_lift_height',
            default_value='true',
            description='Mantener la altura elevada durante la retirada frontal'
        ),
        DeclareLaunchArgument(
            'use_cartesian_local_motions',
            default_value='true',
            description='Usar movimientos locales cartesianos lineales'
        ),
        DeclareLaunchArgument(
            'phase4_start_sync_enabled',
            default_value='true',
            description='Sincronización de inicio para Fase 4 activa'
        ),
        DeclareLaunchArgument(
            'use_split_place_retreat',
            default_value='false',
            description='Usar retirada dividida de colocación'
        ),

        # ---------------------------------------------------------------------
        # Parámetros de mano (flexiones de dedos)
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'index_finger_fraction',
            default_value='0.0',
            description='Multiplicador flexión índice [0.0 a 1.0]'
        ),
        DeclareLaunchArgument(
            'middle_finger_fraction',
            default_value='0.0',
            description='Multiplicador flexión medio [0.0 a 1.0]'
        ),
        DeclareLaunchArgument(
            'thumb_pitch_fraction',
            default_value='0.0',
            description='Multiplicador flexión pitch del pulgar [0.0 a 1.0]'
        ),
        DeclareLaunchArgument(
            'thumb_yaw_fraction',
            default_value='0.0',
            description='Multiplicador rotación yaw del pulgar [0.0 a 1.0]'
        ),
        DeclareLaunchArgument(
            'ring_finger_fraction',
            default_value='0.0',
            description='Multiplicador flexión anular [0.0 a 1.0]'
        ),
        DeclareLaunchArgument(
            'pinky_finger_fraction',
            default_value='0.0',
            description='Multiplicador flexión meñique [0.0 a 1.0]'
        ),

        # ---------------------------------------------------------------------
        # Confirmación de apertura de mano
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'confirm_hand_open_before_detach',
            default_value='true',
            description='Confirmar apertura lógica de mano antes de desadjuntar objeto'
        ),
        DeclareLaunchArgument(
            'release_settle_time',
            default_value='0.5',
            description='Tiempo de espera/estabilización tras apertura de mano [s]'
        ),
        DeclareLaunchArgument(
            'hand_open_position_tolerance',
            default_value='0.08',
            description='Tolerancia de error de posición en juntas para confirmar apertura [rad]'
        ),
        DeclareLaunchArgument(
            'allow_detach_without_hand_joint_feedback',
            default_value='true',
            description='Permitir detach si no hay feedback suficiente de joints en /joint_states'
        ),
        # ---------------------------------------------------------------------
        # Validación geométrica de TCP y colocación mesa
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'validate_tcp_before_detach',
            default_value='true',
            description='Validar pose del TCP antes de desadjuntar y registrar el objeto'
        ),
        DeclareLaunchArgument(
            'tcp_place_position_tolerance',
            default_value='0.012',
            description='Tolerancia posicional para validación de TCP en colocación [m]'
        ),
        DeclareLaunchArgument(
            'tcp_place_orientation_tolerance_deg',
            default_value='8.0',
            description='Tolerancia angular para validación de TCP en colocación [grados]'
        ),
        DeclareLaunchArgument(
            'abort_on_tcp_place_mismatch',
            default_value='true',
            description='Abortar ejecución si la validación de pose del TCP falla'
        ),
        DeclareLaunchArgument(
            'table_place_motion_mode',
            default_value='cartesian',
            description='Modo de movimiento en colocación sobre mesa (cartesian, ompl, auto)'
        ),
        DeclareLaunchArgument(
            'shelf_access_direction_mode',
            default_value='base_axis',
            description='Modo de dirección de acceso al estante (object_approach o base_axis)'
        ),
        DeclareLaunchArgument(
            'shelf_out_dir_x',
            default_value='-1.0',
            description='Componente X del vector de salida del estante'
        ),
        DeclareLaunchArgument(
            'shelf_out_dir_y',
            default_value='0.0',
            description='Componente Y del vector de salida del estante'
        ),
        DeclareLaunchArgument(
            'shelf_out_dir_z',
            default_value='0.0',
            description='Componente Z del vector de salida del estante'
        ),
        DeclareLaunchArgument(
            'shelf_retreat_motion_mode',
            default_value='auto',
            description='Modo de movimiento para la retirada de estante post-pick (cartesian, ompl, auto)'
        ),
        DeclareLaunchArgument(
            'shelf_insert_motion_mode',
            default_value='auto',
            description='Modo de movimiento para la inserción de estante pre-place (cartesian, ompl, auto)'
        ),
        DeclareLaunchArgument(
            'shelf_post_place_retreat_motion_mode',
            default_value='auto',
            description='Modo de movimiento para la retirada de estante post-place (cartesian, ompl, auto)'
        ),
        DeclareLaunchArgument(
            'enable_object_queue',
            default_value='false',
            description='Habilitar procesamiento en cola de múltiples objetos'
        ),
        DeclareLaunchArgument(
            'object_queue_frames',
            default_value='objeto_cubo,objeto_esfera',
            description='Lista de frames TF2 para los objetos en cola'
        ),
        DeclareLaunchArgument(
            'object_queue_types',
            default_value='cube,sphere',
            description='Lista de tipos para los objetos en cola (cube o sphere)'
        ),
        DeclareLaunchArgument(
            'object_queue_dimensions',
            default_value='0.055,0.055',
            description='Lista de dimensiones para los objetos en cola [m]'
        ),
        DeclareLaunchArgument(
            'object_queue_priorities',
            default_value='1,2',
            description='Lista de prioridades para los objetos en cola'
        ),
        DeclareLaunchArgument(
            'queue_order_mode',
            default_value='priority',
            description='Modo de ordenamiento de la cola (priority, distance_to_base, left_to_right, right_to_left)'
        ),
        DeclareLaunchArgument(
            'queue_register_inactive_objects_as_obstacles',
            default_value='true',
            description='Registrar objetos inactivos como obstáculos en la PlanningScene'
        ),
        DeclareLaunchArgument(
            'queue_keep_processed_objects_as_obstacles',
            default_value='true',
            description='Mantener objetos procesados como obstáculos en su pose final'
        ),
        DeclareLaunchArgument(
            'queue_continue_on_object_failure',
            default_value='false',
            description='Continuar con el siguiente objeto si uno de ellos falla'
        ),
        # ---------------------------------------------------------------------
        # Parámetros de planificación OMPL
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'ompl_num_planning_attempts',
            default_value='60',
            description='Número de intentos de planificación OMPL'
        ),
        DeclareLaunchArgument(
            'ompl_allowed_planning_time',
            default_value='15.0',
            description='Tiempo permitido de planificación OMPL [s]'
        ),
        DeclareLaunchArgument(
            'ompl_max_velocity_scaling_factor',
            default_value='0.10',
            description='Factor de escala de velocidad máxima para OMPL'
        ),
        DeclareLaunchArgument(
            'ompl_max_acceleration_scaling_factor',
            default_value='0.10',
            description='Factor de escala de aceleración máxima para OMPL'
        ),
        DeclareLaunchArgument(
            'ompl_joint_goal_tolerance',
            default_value='0.015',
            description='Tolerancia de articulación para el goal OMPL [rad]'
        ),
    ]

    # =========================================================================
    # NOTA DE DEPURACIÓN EN RVIZ (TFs SIMULADOS)
    # =========================================================================
    # Los TFs simulados del cubo y del ArUco se deben publicar en terminales 
    # separadas para pruebas en RViz. No se incluyen en este launch.
    #
    # Comando para publicar el cubo como cara superior:
    # ros2 run tf2_ros static_transform_publisher \
    #   --x 0.5 --y -0.35 --z 0.0974 \
    #   --yaw -0.4 \
    #   --frame-id pelvis \
    #   --child-frame-id objeto_cubo
    #
    # Comando para publicar el ArUco como superficie de mesa:
    # ros2 run tf2_ros static_transform_publisher \
    #   --x 0.50 --y 0.00 --z 0.0424 \
    #   --yaw 0.0 \
    #   --frame-id pelvis \
    #   --child-frame-id aruco_mesa
    # =========================================================================

    mover_brazo_single_node = Node(
        package=package_name,
        executable='mover_brazo_single.py',  # TODO: reemplazar por el nombre real del ejecutable si se registra como console_script sin .py
        name='single_arm_face_approach_grasp_node',
        output='screen',
        parameters=[{
            'arm_side': LaunchConfiguration('arm_side'),
            'base_frame': LaunchConfiguration('base_frame'),
            'object_frame': LaunchConfiguration('object_frame'),
            'object_type': LaunchConfiguration('object_type'),
            'object_dimension': LaunchConfiguration('object_dimension'),
            'task_mode': LaunchConfiguration('task_mode'),
            'manipulation_geometry_mode': LaunchConfiguration('manipulation_geometry_mode'),
            'place_geometry_mode': LaunchConfiguration('place_geometry_mode'),
            'phase4_motion_mode': LaunchConfiguration('phase4_motion_mode'),
            'use_aruco_table_target': LaunchConfiguration('use_aruco_table_target'),
            'table_target_frame': LaunchConfiguration('table_target_frame'),
            'table_target_offset_x': LaunchConfiguration('table_target_offset_x'),
            'table_target_offset_y': LaunchConfiguration('table_target_offset_y'),
            'table_target_offset_z': LaunchConfiguration('table_target_offset_z'),
            'table_target_offset_mode': LaunchConfiguration('table_target_offset_mode'),
            'aruco_z_is_table_surface': LaunchConfiguration('aruco_z_is_table_surface'),
            'object_pose_reference': LaunchConfiguration('object_pose_reference'),
            'use_aruco_table_height': LaunchConfiguration('use_aruco_table_height'),
            'aruco_table_height_offset': LaunchConfiguration('aruco_table_height_offset'),
            'allow_cached_aruco_table_height': LaunchConfiguration('allow_cached_aruco_table_height'),
            'table_collision': LaunchConfiguration('table_collision'),
            'publish_debug_markers': LaunchConfiguration('publish_debug_markers'),
            'validate_aruco_table_target_ik': LaunchConfiguration('validate_aruco_table_target_ik'),
            'shelf_pregrasp_distance': LaunchConfiguration('shelf_pregrasp_distance'),
            'shelf_surface_clearance': LaunchConfiguration('shelf_surface_clearance'),
            'shelf_skip_vertical_descent': LaunchConfiguration('shelf_skip_vertical_descent'),
            'shelf_lift_before_retreat_enabled': LaunchConfiguration('shelf_lift_before_retreat_enabled'),
            'shelf_lift_before_retreat_distance': LaunchConfiguration('shelf_lift_before_retreat_distance'),
            'shelf_lift_before_retreat_motion_mode': LaunchConfiguration('shelf_lift_before_retreat_motion_mode'),
            'shelf_retreat_keep_lift_height': LaunchConfiguration('shelf_retreat_keep_lift_height'),
            'use_cartesian_local_motions': LaunchConfiguration('use_cartesian_local_motions'),
            'phase4_start_sync_enabled': LaunchConfiguration('phase4_start_sync_enabled'),
            'use_split_place_retreat': LaunchConfiguration('use_split_place_retreat'),
            'index_finger_fraction': LaunchConfiguration('index_finger_fraction'),
            'middle_finger_fraction': LaunchConfiguration('middle_finger_fraction'),
            'thumb_pitch_fraction': LaunchConfiguration('thumb_pitch_fraction'),
            'thumb_yaw_fraction': LaunchConfiguration('thumb_yaw_fraction'),
            'ring_finger_fraction': LaunchConfiguration('ring_finger_fraction'),
            'pinky_finger_fraction': LaunchConfiguration('pinky_finger_fraction'),
            'confirm_hand_open_before_detach': LaunchConfiguration('confirm_hand_open_before_detach'),
            'release_settle_time': LaunchConfiguration('release_settle_time'),
            'hand_open_position_tolerance': LaunchConfiguration('hand_open_position_tolerance'),
            'allow_detach_without_hand_joint_feedback': LaunchConfiguration('allow_detach_without_hand_joint_feedback'),
            'validate_tcp_before_detach': LaunchConfiguration('validate_tcp_before_detach'),
            'tcp_place_position_tolerance': LaunchConfiguration('tcp_place_position_tolerance'),
            'tcp_place_orientation_tolerance_deg': LaunchConfiguration('tcp_place_orientation_tolerance_deg'),
            'abort_on_tcp_place_mismatch': LaunchConfiguration('abort_on_tcp_place_mismatch'),
            'table_place_motion_mode': LaunchConfiguration('table_place_motion_mode'),
            'shelf_access_direction_mode': LaunchConfiguration('shelf_access_direction_mode'),
            'shelf_out_dir_x': LaunchConfiguration('shelf_out_dir_x'),
            'shelf_out_dir_y': LaunchConfiguration('shelf_out_dir_y'),
            'shelf_out_dir_z': LaunchConfiguration('shelf_out_dir_z'),
            'shelf_retreat_motion_mode': LaunchConfiguration('shelf_retreat_motion_mode'),
            'shelf_insert_motion_mode': LaunchConfiguration('shelf_insert_motion_mode'),
            'shelf_post_place_retreat_motion_mode': LaunchConfiguration('shelf_post_place_retreat_motion_mode'),
            'enable_object_queue': LaunchConfiguration('enable_object_queue'),
            'object_queue_frames': LaunchConfiguration('object_queue_frames'),
            'object_queue_types': LaunchConfiguration('object_queue_types'),
            'object_queue_dimensions': LaunchConfiguration('object_queue_dimensions'),
            'object_queue_priorities': LaunchConfiguration('object_queue_priorities'),
            'queue_order_mode': LaunchConfiguration('queue_order_mode'),
            'queue_register_inactive_objects_as_obstacles': LaunchConfiguration('queue_register_inactive_objects_as_obstacles'),
            'queue_keep_processed_objects_as_obstacles': LaunchConfiguration('queue_keep_processed_objects_as_obstacles'),
            'queue_continue_on_object_failure': LaunchConfiguration('queue_continue_on_object_failure'),
            'ompl_num_planning_attempts': LaunchConfiguration('ompl_num_planning_attempts'),
            'ompl_allowed_planning_time': LaunchConfiguration('ompl_allowed_planning_time'),
            'ompl_max_velocity_scaling_factor': LaunchConfiguration('ompl_max_velocity_scaling_factor'),
            'ompl_max_acceleration_scaling_factor': LaunchConfiguration('ompl_max_acceleration_scaling_factor'),
            'ompl_joint_goal_tolerance': LaunchConfiguration('ompl_joint_goal_tolerance'),
        }]
    )

    return LaunchDescription(declared_arguments + [mover_brazo_single_node])
