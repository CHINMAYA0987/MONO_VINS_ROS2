import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('mono_vio')
    default_cfg  = os.path.join(pkg_dir, 'config', 'params.yaml')
    default_rviz = os.path.join(pkg_dir, 'rviz',   'mono_vio.rviz')

    cfg_arg = DeclareLaunchArgument(
        'config', default_value=default_cfg,
        description='Path to params.yaml')

    vio_node = Node(
        package='mono_vio',
        executable='mono_vio_node',
        name='mono_vio_node',
        output='screen',
        parameters=[LaunchConfiguration('config')],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz],
        output='screen',
    )

    return LaunchDescription([cfg_arg, vio_node, rviz_node])