#!/bin/bash

colcon build

source install/setup.bash

ros2 launch simulation_environment test_env.launch.py robot_type:=A
