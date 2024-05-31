import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import Shutdown, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessIO


def generate_launch_description():
 
    urdf_path = os.path.join(get_package_share_path('explorer_bot'),
                             'urdf', 'my_robot.urdf.xacro')

    gazebo_world_path = os.path.join(get_package_share_path('simulation_environment'),
                             'worlds', 'train_world.world')

    rviz_config_path = os.path.join(get_package_share_path('simulation_environment'),
                             'rviz', 'urdf_config.rviz')


    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': gazebo_world_path}.items(),
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
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )

    lidar_node = Node(
        package='simulation_environment',
        executable='lidar_node.py',
        name='lidar_node',
        output='screen',
    )

    train_auto_drive_node = Node(
        package='simulation_environment',
        executable='train_auto_drive_node.py',
        name='train_auto_drive_node',
        output='screen',
    )

    pytorch_dqn = Node(
        package='simulation_environment',
        executable='pytorch_dqn.py',
        name='pytorch_dqn',
        output='screen',
    )

    return LaunchDescription([
        robot_state_publisher,        
        gazebo,
        spawn_entity,
        rviz2_node,
        lidar_node,
        train_auto_drive_node,
        pytorch_dqn
    ])
