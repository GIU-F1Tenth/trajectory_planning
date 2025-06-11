import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped

from rclpy.action import ActionClient


class LookaheadToPathPlanner(Node):
    def __init__(self):
        super().__init__('lookahead_to_path_planner')

        self.declare_parameter("astar_lookahead_marker_topic","")
        self.declare_parameter("compute_path_to_pose_topic", "")

        self.astar_lookahead_marker_topic = self.get_parameter("astar_lookahead_marker_topic").get_parameter_value().string_value
        self.compute_path_to_pose_topic = self.get_parameter("compute_path_to_pose_topic").get_parameter_value().string_value

        # Subscriber to the marker
        self.subscription = self.create_subscription(
            Marker,
            self.astar_lookahead_marker_topic,
            self.marker_callback,
            10
        )

        # Action client for ComputePathToPose
        self._action_client = ActionClient(self, ComputePathToPose, self.compute_path_to_pose_topic)

    def marker_callback(self, msg: Marker):
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
        self.get_logger().info(f"Sending goal to x={goal_pose.pose.position.x}, y={goal_pose.pose.position.y}")
        self._action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LookaheadToPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
