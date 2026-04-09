cmake_minimum_required(VERSION 3.8)
project(h1_2_description)

# find dependencies
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    urdf
    meshes
    launch
    rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
