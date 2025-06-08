cmake_minimum_required(VERSION 3.0.2)
project(groundingdino)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  private/Ground/ros_ground.py
  private/Ollama/ollama_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
