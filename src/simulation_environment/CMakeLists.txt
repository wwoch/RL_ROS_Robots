cmake_minimum_required(VERSION 3.8)
project(simulation_environment)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

install(
  DIRECTORY launch rviz worlds simulation_environment
  DESTINATION share/${PROJECT_NAME}/
)

# install(
#   DIRECTORY launch rviz worlds simulation_environment
#   DESTINATION lib/${PROJECT_NAME}/
# )

# install(
#   FILES
#   simulation_environment/lidar_node.py
#   DESTINATION lib/${PROJECT_NAME}
# )

install(
  PROGRAMS
  simulation_environment/lidar_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  simulation_environment/train_auto_drive_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  simulation_environment/pytorch_dqn.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  simulation_environment/test_auto_drive_node.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_python_install_package(${PROJECT_NAME})

ament_package()
