from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()
    
    param_path = os.path.join(get_package_share_directory("trajectory_planning"), "config", "lookahead_to_planner_config.yaml")
    csv_path_pub = Node(
            package='trajectory_planning',
            executable='lookahead_to_planner_exe',
            name='lookahead_to_path_planner',
            parameters=[param_path],
            output='screen'
        )
    
    ld.add_action(csv_path_pub)

    return ld