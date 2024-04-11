import os
from launch import LaunchDescription
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_path
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node

def generate_launch_description():

    urdf_path = os.path.join(get_package_share_path('explorer_bot'),
                             'urdf', 'explorer_robot.urdf.xacro')

    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str) #pip install xacro

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui"
    )

    rvizz2_node = Node(
        package="rviz2",
        executable="rviz2"
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rvizz2_node
    ])
