"""
A* Path Planner Launch File

This launch file starts the A* path planning node with its configuration parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the A* path planner.

    Returns:
        LaunchDescription: The launch description containing the A* planner node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    config_path = os.path.join(
        get_package_share_directory("trajectory_planning"),
        "config",
        "a_star_planner_config.yaml"
    )

    # Create the A* planner node
    a_star_node = Node(
        package='trajectory_planning',
        executable='a_star_exe',
        name='a_star_planner',
        parameters=[config_path],
        output='screen',
        emulate_tty=True
    )

    ld.add_action(a_star_node)

    return ld
