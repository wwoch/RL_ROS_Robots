#!/bin/bash

colcon build

source install/setup.bash

ros2 run simulation_environment pytorch_dqn.py --ros-args -p robot_type:=A
