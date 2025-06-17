#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import math
import csv
import os
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class AstarLookahead(Node):
    def __init__(self):
        super().__init__("astar_lookahead_pub_node")

        self.declare_parameter("is_antiClockwise", False)
        self.declare_parameter("path_topic", "")
        self.declare_parameter("lookahead_distance", 0.0)
        self.declare_parameter("lookahead_marker_topic", "")
        self.declare_parameter("object_detected_topic", "/tmp/obj_detected")

        self.is_antiClockwise = self.get_parameter("is_antiClockwise").get_parameter_value().bool_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.lookahead_distance = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        self.marker_pub_topic = self.get_parameter("lookahead_marker_topic").get_parameter_value().string_value
        self.obj_detected_topic = self.get_parameter("object_detected_topic").get_parameter_value().string_value
        self.path_sub = self.create_subscription(Path, self.path_topic, self.path_update_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.005, self.get_pose)  # 50 Hz
        self.path = [] # a tuple of (x, y, v) 

        self.lookahead_marker_pub = self.create_publisher(Marker, self.marker_pub_topic, 10)
        self.lookahead_circle_pub = self.create_publisher(Marker, "/astar_lookahead_circle", 10)
        self.path_publisher = self.create_publisher(Path, "/pp_path", 10)
        self.obj_detected_sub = self.create_subscription(Bool, self.obj_detected_topic, self.create_path, 10)
        self._action_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.goal_sent = False

    def create_path(self, msg: Bool): 
        if not msg.data:
            return 
        marker = self.marker
        goal_msg = ComputePathToPose.Goal()

        # Create a PoseStamped message for the goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = marker.header.frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose = marker.pose

        # Assign the goal pose
        goal_msg.goal = goal_pose

        # Wait for action server and send goal
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.goal_sent = True

    def path_update_cb(self, msg:Path):
        self.path.clear() # to clear the path
        for i in range(len(msg.poses)):
            self.path.append((msg.poses[i].pose.position.x, msg.poses[i].pose.position.y, msg.poses[i].pose.orientation.w))
        if self.is_antiClockwise:
            self.path.reverse()
        self.get_logger().info(f"path has been updated...")

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                'map',      # target_frame
                'laser',    # source_frame (your base_frame)
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Convert quaternion to yaw
            orientation_list = [rot.x, rot.y, rot.z, rot.w]
            _, _, yaw = euler_from_quaternion(orientation_list)

            # self.get_logger().info(f"Robot Pose - x: {trans.x:.2f}, y: {trans.y:.2f}, yaw: {yaw:.2f}")
            x, y = trans.x, trans.y
            
            self.publish_lookahead_circle(x, y)
            lookahead_point, closest_point, lookahead_index = self.find_lookahead_point(x, y)
            if lookahead_point is None:
                self.get_logger().warn("No lookahead point found go ")
            else:
                self.publish_lookahead_marker(lookahead_point)

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")



    def find_lookahead_point(self, x, y):
        """
        This function returns the lookahead distance and the closest point to the car.
        Returns the index of the lookahead point as a third element.
        """

        closest_idx = 0
        min_dist = float('inf')
        # First find the closest path point to the car
        for i, point in enumerate(self.path):
            dx = point[0] - x
            dy = point[1] - y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Now search only forward from that point
        for i in range(closest_idx, len(self.path)):
            dx = self.path[i][0] - x
            dy = self.path[i][1] - y
            distance = math.sqrt(dx**2 + dy**2)
            if distance >= self.lookahead_distance:
                return self.path[i], self.path[closest_idx], i

        # If no point was found, assume starting and reset closest idx
        closest_idx = 0
        for i in range(closest_idx, len(self.path)):
            dx = self.path[i][0] - x
            dy = self.path[i][1] - y
            distance = math.sqrt(dx**2 + dy**2)
            if distance >= self.lookahead_distance:
                return self.path[i], self.path[closest_idx], i

        return None, None, None


    def publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.frame_id = 'map'  # or 'map' depending on your frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "astar_lookahead"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.1  # Slightly above ground
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.lookahead_marker_pub.publish(marker)
        if self.goal_sent:
            return
        
        self.marker = marker

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by Nav2.')
            self.goal_sent = False
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
        
    def get_result_callback(self, future):
        result = future.result().result
            
        if result.path: 
            for pose_stamped in result.path.poses:
                pose_stamped.pose.orientation.w = 0.0
            result.path.poses.reverse()
            self.path_publisher.publish(result.path)
        else:
            self.get_logger().warn("No path returned in result")
        self.goal_sent = False

    def publish_lookahead_circle(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'  # or 'map' if you're using that
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "astar_lookahead"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03  # line thickness
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Create circle points
        resolution = 60  # more = smoother circle
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            px = x + self.lookahead_distance * math.cos(angle)
            py = y + self.lookahead_distance * math.sin(angle)
            p = Point()  # Dummy initialization to get geometry_msgs/Point
            p.x = px
            p.y = py
            p.z = 0.05
            marker.points.append(p)

        self.lookahead_circle_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = AstarLookahead()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()