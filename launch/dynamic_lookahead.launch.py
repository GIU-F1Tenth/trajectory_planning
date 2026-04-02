"""
Dynamic Lookahead Publisher Launch File

This launch file starts the Dynamic lookahead publisher node that generates
lookahead points for path following based on CSV waypoints.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the Dynamic lookahead publisher.

    Returns:
        LaunchDescription: The launch description containing the lookahead publisher node
    """
    ld = LaunchDescription()

    param_path = os.path.join(get_package_share_directory(
        "trajectory_planning"), "config", "dynamic_lookahead_pub_config.yaml")
    dynamic_lookahead_path_pub = Node(
        package='trajectory_planning',
        executable='dynamic_lookahead_pub_exe',
        name='dynamic_lookahead_pub_node',
        parameters=[param_path],
        output='screen'
    )

    ld.add_action(dynamic_lookahead_path_pub)

    return ld
