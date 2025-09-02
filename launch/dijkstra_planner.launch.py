"""
Dijkstra Path Planner Launch File

This launch file starts the Dijkstra path planning node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the Dijkstra path planner.

    Returns:
        LaunchDescription: The launch description containing the Dijkstra planner node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    config_path = os.path.join(
        get_package_share_directory("trajectory_planning"),
        "config",
        "dijkstra_planner_config.yaml"
    )

    # Create the Dijkstra planner node
    dijkstra_node = Node(
        package='trajectory_planning',
        executable='dijkstra_exe',
        name='dijkstra_node',
        output='screen',
        parameters=[config_path],
        emulate_tty=True
    )

    ld.add_action(dijkstra_node)

    return ld
