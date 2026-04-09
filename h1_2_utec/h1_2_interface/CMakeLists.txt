cmake_minimum_required(VERSION 3.8)
project(h1_2_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_H1_2_INTERFACE "Build H1-2 hardware interface nodes" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

if(BUILD_H1_2_INTERFACE)

  find_package(unitree_hg REQUIRED)

  add_executable(lowstate_to_jointstates  src/lowstate_to_jointstates.cpp)
  ament_target_dependencies(
    lowstate_to_jointstates 
    rclcpp 
    sensor_msgs
    unitree_hg)

  install(TARGETS lowstate_to_jointstates
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()
