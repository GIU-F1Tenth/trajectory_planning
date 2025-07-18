"""
Lookahead to Planner Launch File

This launch file starts the lookahead to planner bridge node that converts
lookahead markers into path planning goals using Nav2.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the lookahead to planner bridge.

    Returns:
        LaunchDescription: The launch description containing the bridge node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    param_path = os.path.join(
        get_package_share_directory("trajectory_planning"),
        "config",
        "lookahead_to_planner_config.yaml"
    )

    # Create the lookahead to planner bridge node
    lookahead_planner_node = Node(
        package='trajectory_planning',
        executable='lookahead_to_planner_exe',
        name='lookahead_to_path_planner',
        parameters=[param_path],
        output='screen',
        emulate_tty=True
    )

    ld.add_action(lookahead_planner_node)

    return ld
