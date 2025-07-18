"""
Complete Trajectory Planning System Launch File

This launch file starts all components of the trajectory planning system:
- CSV path publisher
- A* lookahead publisher  
- A* path planner
- Lookahead to planner bridge
- Pure pursuit controller
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the complete launch description for the trajectory planning system.

    Returns:
        LaunchDescription: The launch description containing all trajectory planning nodes
    """
    ld = LaunchDescription()

    # Get configuration file paths
    config_dir = os.path.join(get_package_share_directory(
        "trajectory_planning"), "config")

    csv_config = os.path.join(config_dir, "csv_pub_config.yaml")
    astar_lookahead_config = os.path.join(
        config_dir, "astar_lookahead_pub_config.yaml")
    astar_planner_config = os.path.join(
        config_dir, "a_star_planner_config.yaml")
    lookahead_planner_config = os.path.join(
        config_dir, "lookahead_to_planner_config.yaml")
    pure_pursuit_config = os.path.join(
        config_dir, "pure_pursuit_v2_params.yaml")

    # CSV Path Publisher
    csv_pub_node = Node(
        package='trajectory_planning',
        executable='csv_pub_exe',
        name='csv_path_publisher',
        parameters=[csv_config],
        output='screen',
        emulate_tty=True
    )

    # A* Lookahead Publisher
    astar_lookahead_node = Node(
        package='trajectory_planning',
        executable='astar_lookahead_pub_exe',
        name='astar_lookahead_publisher',
        parameters=[astar_lookahead_config],
        output='screen',
        emulate_tty=True
    )

    # A* Path Planner
    astar_planner_node = Node(
        package='trajectory_planning',
        executable='a_star_exe',
        name='a_star_planner',
        parameters=[astar_planner_config],
        output='screen',
        emulate_tty=True
    )

    # Lookahead to Planner Bridge
    lookahead_planner_node = Node(
        package='trajectory_planning',
        executable='lookahead_to_planner_exe',
        name='lookahead_to_path_planner',
        parameters=[lookahead_planner_config],
        output='screen',
        emulate_tty=True
    )

    # Pure Pursuit Controller
    pure_pursuit_node = Node(
        package='trajectory_planning',
        executable='pure_pursuit_node_v2',
        name='pure_pursuit_v2',
        parameters=[pure_pursuit_config],
        output='screen',
        emulate_tty=True
    )

    # Add all nodes to launch description
    ld.add_action(csv_pub_node)
    ld.add_action(astar_lookahead_node)
    ld.add_action(astar_planner_node)
    ld.add_action(lookahead_planner_node)
    ld.add_action(pure_pursuit_node)

    return ld
