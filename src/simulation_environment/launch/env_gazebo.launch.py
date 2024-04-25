import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
 
    urdf_path = os.path.join(get_package_share_path('explorer_bot'),
                             'urdf', 'my_robot.urdf.xacro')

    gazebo_world_path = os.path.join(get_package_share_path('simulation_environment'),
                             'worlds', 'sq_wall.world')

    rviz_config_path = os.path.join(get_package_share_path('simulation_environment'),
                             'rviz', 'urdf_config.rviz')


    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
    )


    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', rviz_config_path]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )


    return LaunchDescription([
        robot_state_publisher,        
        gazebo,
        spawn_entity,
        rviz2_node
    ])

