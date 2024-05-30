#!/bin/bash

colcon build

source install/setup.bash

ros2 launch simulation_environment train_env.launch.py
