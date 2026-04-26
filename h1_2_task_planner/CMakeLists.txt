cmake_minimum_required(VERSION 3.8)
project(h1_2_task_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(pose_goal_executor src/pose_goal_executor.cpp)

ament_target_dependencies(pose_goal_executor
  rclcpp
  geometry_msgs
  visualization_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  pose_goal_executor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
