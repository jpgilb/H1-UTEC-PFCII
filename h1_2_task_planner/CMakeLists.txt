cmake_minimum_required(VERSION 3.8)
project(h1_2_task_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

# Nodo para enviar una pose objetivo individual
add_executable(pose_goal_executor
  src/pose_goal_executor.cpp
)

ament_target_dependencies(pose_goal_executor
  rclcpp
  geometry_msgs
  visualization_msgs
  moveit_ros_planning_interface
)

# Nodo para pruebas de trayectoria de un brazo y recolección de métricas
add_executable(experiment_runner
  src/experiment_runner.cpp
)

ament_target_dependencies(experiment_runner
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  moveit_ros_planning_interface
)

# Nodo para movimiento bimanual coordinado
add_executable(bimanual_coordinated_executor
  src/bimanual_coordinated_executor.cpp
)

ament_target_dependencies(bimanual_coordinated_executor
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  pose_goal_executor
  experiment_runner
  bimanual_coordinated_executor
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/joint_state_trial_logger.py
  scripts/hand_pose_trial_logger.py
  scripts/plot_results.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/joint_state_trial_logger.py
  scripts/plot_results.py
  scripts/hand_pose_trial_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
