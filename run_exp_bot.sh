#!/bin/bash

colcon build

source install/setup.bash

ros2 launch explorer_bot display.launch.py
