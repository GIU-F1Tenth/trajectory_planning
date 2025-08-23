"""
Hybrid A* Path Planner Launch File

This launch file starts the Hybrid A* path planning node with its configuration parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the Hybrid A* path planner.

    Returns:
        LaunchDescription: The launch description containing the Hybrid A* planner node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    config_path = os.path.join(
        get_package_share_directory("trajectory_planning"),
        "config",
        "hybrid_Astar_config.yaml"
    )

    # Create the Hybrid A* planner node
    hybrid_Astar_node = Node(
        package='trajectory_planning',
        executable='hybrid_Astar_exe',
        name='hybrid_Astar_node',
        parameters=[config_path],
        output='screen',
        emulate_tty=True
    )

    ld.add_action(hybrid_Astar_node)

    return ld
