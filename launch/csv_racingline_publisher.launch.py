from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()
    
    param_path = os.path.join(get_package_share_directory("trajectory_planning"), "config", "csv_pub_config.yaml")
    csv_path_pub = Node(
            package='trajectory_planning',
            executable='csv_pub_exe',
            name='csv_path_publisher_node',
            parameters=[param_path],
            output='screen'
        )
    
    ld.add_action(csv_path_pub)

    return ld