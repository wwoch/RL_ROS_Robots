#!/usr/bin/env python3
from setuptools import setup, find_packages

package_name = 'simulation_environment'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(where='src'),
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROS 2 Developer',
    maintainer_email='ros2@ros.com',
    description='Pakiet do symulacji środowiska w ROS2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points22={
        'console_scripts': [
            'lidar_node = simulation_environment.lidar_node:main',
            'auto_drive_node = simulation_environment.auto_drive:main',
        ],
    },
)
