#!/bin/bash

colcon build

source install/setup.bash

ros2 launch simulation_environment learn_env.launch.py robot_type:=B
