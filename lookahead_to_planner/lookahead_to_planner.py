"""
Lookahead to Planner Module

This module implements a ROS2 node that bridges lookahead markers with path planning.
It subscribes to lookahead markers and uses them as goal points for the Nav2 path planner,
then publishes the resulting paths.
"""

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from rclpy.action import ActionClient
from nav_msgs.msg import Path


class LookaheadToPathPlanner(Node):
    """
    A ROS2 node that converts lookahead markers to path planning goals.

    This node subscribes to visualization markers (typically lookahead points)
    and uses them as goal points for the Nav2 ComputePathToPose action server.
    The resulting paths are published for use by path following controllers.
    """

    def __init__(self):
        """
        Initialize the LookaheadToPathPlanner node.

        Sets up parameters, subscribers, publishers, and action clients.
        """
        super().__init__('lookahead_to_path_planner')

        # Declare parameters with default values
        self.declare_parameter("astar_lookahead_marker_topic", "")
        self.declare_parameter("compute_path_to_pose_topic", "")
        self.declare_parameter("output_path_topic", "/astar_pp_path")

        # Get parameter values
        self.astar_lookahead_marker_topic = self.get_parameter(
            "astar_lookahead_marker_topic").get_parameter_value().string_value
        self.compute_path_to_pose_topic = self.get_parameter(
            "compute_path_to_pose_topic").get_parameter_value().string_value
        self.output_path_topic = self.get_parameter(
            "output_path_topic").get_parameter_value().string_value

        # Subscriber to the marker
        self.subscription = self.create_subscription(
            Marker,
            self.astar_lookahead_marker_topic,
            self.marker_callback,
            10
        )

        # Publisher for the computed path
        self.path_publisher = self.create_publisher(
            Path, self.output_path_topic, 10)

        # Action client for ComputePathToPose
        self._action_client = ActionClient(
            self, ComputePathToPose, self.compute_path_to_pose_topic)
        self.goal_sent = False

    def marker_callback(self, msg: Marker):
        """
        Callback function for processing incoming marker messages.

        This function receives lookahead markers and converts them to path planning goals.

        Args:
            msg (Marker): The received marker message containing the goal position
        """
        # Wait for the action server to become available
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('ComputePathToPose action server not available.')
            return

        # Construct the goal message
        goal_msg = ComputePathToPose.Goal()

        # Set goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = msg.header.frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position = msg.pose.position
        goal_pose.pose.orientation = msg.pose.orientation

        goal_msg.goal = goal_pose
        goal_msg.use_start = False  # Using robot's current pose

        # Send the goal
        self.get_logger().info(
            f"Sending goal to x={goal_pose.pose.position.x}, y={goal_pose.pose.position.y}")
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Callback function for handling goal response from the action server.

        Args:
            future: The future object containing the goal response
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by Nav2.')
            self.goal_sent = False
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Callback function for handling the final result from the action server.

        Args:
            future: The future object containing the action result
        """
        result = future.result().result

        if result.path:
            for pose_stamped in result.path.poses:
                pose_stamped.pose.orientation.w = 0.0
            # result.path.poses.reverse()
            self.path_publisher.publish(result.path)
        else:
            self.get_logger().warn("No path returned in result")
        self.goal_sent = False


def main(args=None):
    """
    Main function to initialize and run the LookaheadToPathPlanner node.

    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = LookaheadToPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
