#!/bin/bash

colcon build

source install/setup.bash

ros2 launch simulation_environment env_gazebo.launch.py
