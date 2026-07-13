cmake_minimum_required(VERSION 3.8)
project(h1_2_mujoco_bringup)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(mujoco_ros2_control REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(pluginlib REQUIRED)

# Compile with C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define our plugin library
add_library(h1_2_mujoco_plugins SHARED
  src/h1_2_mujoco_plugins.cpp
)

ament_target_dependencies(h1_2_mujoco_plugins
  rclcpp
  std_msgs
  mujoco_ros2_control
  mujoco_ros2_control_plugins
  pluginlib
)

# Export the pluginlib plugin xml
pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins h1_2_mujoco_plugins.xml)

# Install targets
install(TARGETS h1_2_mujoco_plugins
  EXPORT export_h1_2_mujoco_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install description, config, launch, scripts directories
install(DIRECTORY description config launch scripts
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/h1_2_dynamics_hold_controller.py
  scripts/mujoco_pause_until_hold_ready.py
  DESTINATION lib/${PROJECT_NAME}
)


# Export dependencies
ament_export_libraries(h1_2_mujoco_plugins)
ament_export_dependencies(
  rclcpp
  std_msgs
  mujoco_ros2_control
  mujoco_ros2_control_plugins
  pluginlib
)

# Install resources/package.xml
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
