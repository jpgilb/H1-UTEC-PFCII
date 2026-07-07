cmake_minimum_required(VERSION 3.22)
project(h1_2_moveit_config)

find_package(ament_cmake REQUIRED)


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  ../scripts/mover_brazo_single.py
  ../scripts/mover_brazo.py
  ../scripts/dynamic_mover_brazo.py
  ../scripts/medir_mesh_palma.py
  ../scripts/reset_robot.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
