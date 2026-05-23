#!/usr/bin/env python3
"""
Dynamic Lookahead Publisher Module (Dynamic Lookahead)

This module implements a ROS2 node that publishes lookahead points.
The lookahead distance is dynamically adjusted based on the robot's current velocity
from odometry.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import math
import csv
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener


class DynamicLookahead(Node):
    def __init__(self):
        super().__init__("dynamic_lookahead_pub_node")

        # Parameters
        self.declare_parameter("lookahead_distance_min",
                               0.5)   # minimum lookahead
        self.declare_parameter("lookahead_distance_max",
                               3.0)   # maximum lookahead
        self.declare_parameter("lookahead_marker_topic", "/lookahead_goal")
        self.declare_parameter("csv_path_topic", "/csv_path")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("source_frame", "ego_racecar/base_link")
        self.declare_parameter("timer_frequency", 30.0)
        self.declare_parameter("odom_topic", "ego_racecar/odom")
        self.declare_parameter("velocity_scale", 1.0)
        self.declare_parameter("use_lateral_deviation_based_lookahead", False)
        self.declare_parameter("lookahead_velocity_weight", 0.5)
        self.declare_parameter("lookahead_lateral_deviation_weight", 0.5)
        self.declare_parameter("max_lateral_deviation", 1.0)
        self.declare_parameter("use_lateral_error_speed_reducer", True)
        self.declare_parameter("lateral_error_speed_reducer_gain", 0.1)
        self.declare_parameter("reverse", False)

        # Get parameters
        self.lookahead_distance_min = self.get_parameter(
            "lookahead_distance_min").get_parameter_value().double_value
        self.lookahead_distance_max = self.get_parameter(
            "lookahead_distance_max").get_parameter_value().double_value
        self.marker_pub_topic = self.get_parameter(
            "lookahead_marker_topic").get_parameter_value().string_value
        self.csv_path_topic = self.get_parameter(
            "csv_path_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value
        self.source_frame = self.get_parameter(
            "source_frame").get_parameter_value().string_value
        self.timer_frequency = self.get_parameter(
            "timer_frequency").get_parameter_value().double_value
        self.velocity_scale = self.get_parameter(
            "velocity_scale").get_parameter_value().double_value
        self.odom_topic = self.get_parameter(
            "odom_topic").get_parameter_value().string_value
        self.use_lateral_deviation_based_lookahead = self.get_parameter(
            "use_lateral_deviation_based_lookahead"
        ).get_parameter_value().bool_value
        self.lookahead_velocity_weight = self.get_parameter(
            "lookahead_velocity_weight").get_parameter_value().double_value
        self.lookahead_lateral_deviation_weight = self.get_parameter(
            "lookahead_lateral_deviation_weight").get_parameter_value().double_value
        self.max_lateral_deviation = self.get_parameter(
            "max_lateral_deviation").get_parameter_value().double_value
        self.use_lateral_error_speed_reducer = self.get_parameter(
            "use_lateral_error_speed_reducer").get_parameter_value().bool_value
        self.lateral_error_speed_reducer_gain = self.get_parameter(
            "lateral_error_speed_reducer_gain").get_parameter_value().double_value
        self.reverse = self.get_parameter(
            "reverse").get_parameter_value().bool_value

        # Publishers
        self.lookahead_marker_pub = self.create_publisher(
            Marker, self.marker_pub_topic, 10)
        self.lookahead_circle_pub = self.create_publisher(
            Marker, "/dynamic_lookahead_circle", 10)

        # Subscribers
        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(
            Path, self.csv_path_topic, self.csv_path_callback, 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer
        timer_period = 1.0 / self.timer_frequency
        self.timer = self.create_timer(timer_period, self.get_pose)

        # Variables
        self.current_velocity = 0.0
        self.marker = None
        self.path = []
        self.min_velocity = 0.0

    def csv_path_callback(self, msg: Path):
        """Convert Path message to internal path format."""
        self.path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            v = pose.pose.orientation.w
            self.path.append((x, y, v))

        if self.reverse:
            self.path.reverse()  # reverse for easier pop from end

        self.min_velocity = self.get_path_min_velocity()

    def get_path_min_velocity(self):
        velocities = [point[2] for point in self.path if point[2] > 0.0]
        if not velocities:
            return 0.0

        return min(velocities)

    def odom_callback(self, msg: Odometry):
        """Extract velocity magnitude from odometry."""
        self.current_velocity = msg.twist.twist.linear.x

    def compute_dynamic_lookahead(self):
        """Scale lookahead distance based on velocity."""
        dynamic_lookahead = self.current_velocity * self.velocity_scale
        # Clamp between min and max
        return max(self.lookahead_distance_min,
                   min(dynamic_lookahead, self.lookahead_distance_max))

    def compute_lateral_deviation_lookahead(self, lateral_error):
        """Scale lookahead distance based on lateral deviation from the path."""
        if self.max_lateral_deviation <= 0.0:
            return self.lookahead_distance_max

        lookahead = (
            self.lookahead_distance_min
            + abs(lateral_error)
            * (self.lookahead_distance_max - self.lookahead_distance_min)
            / self.max_lateral_deviation
        )
        return lookahead

    def compute_lateral_deviation_velocity_lookahead(self, lateral_error):
        """Blend velocity-based and lateral-deviation-based lookahead."""
        velocity_lookahead = self.compute_dynamic_lookahead()
        lateral_lookahead = self.compute_lateral_deviation_lookahead(lateral_error)
        lookahead = (
            self.lookahead_velocity_weight * velocity_lookahead
            + self.lookahead_lateral_deviation_weight * lateral_lookahead
        )
        return max(
            self.lookahead_distance_min,
            min(lookahead, self.lookahead_distance_max),
        )

    def reduce_velocity_for_lateral_error(self, velocity, lateral_error):
        if not self.use_lateral_error_speed_reducer or velocity <= 0.0:
            return velocity

        speed_reduction = abs(lateral_error) * self.lateral_error_speed_reducer_gain
        return max(self.min_velocity, velocity - speed_reduction)

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            trans = transform.transform.translation
            rot = transform.transform.rotation
            x, y = trans.x, trans.y
            yaw = self.get_yaw_from_quaternion(rot)

            # Compute dynamic lookahead
            self.lookahead_distance = self.compute_dynamic_lookahead()
            closest_path_point = self.find_closest_point_on_path(x, y)
            lateral_error = 0.0
            if closest_path_point is not None:
                _, lateral_error = self.transform_to_vehicle_frame(
                    closest_path_point, x, y, yaw
                )

            if self.use_lateral_deviation_based_lookahead:
                if closest_path_point is not None:
                    self.lookahead_distance = (
                        self.compute_lateral_deviation_velocity_lookahead(
                            lateral_error
                        )
                    )

            # Publish visuals
            self.publish_lookahead_circle(x, y)
            lookahead_point, closest_point, _ = self.find_lookahead_point(x, y)
            if lookahead_point:
                target_velocity = self.reduce_velocity_for_lateral_error(
                    closest_point[2], lateral_error
                )
                lookahead_point = (
                    lookahead_point[0], lookahead_point[1], target_velocity)
                self.publish_lookahead_marker(lookahead_point)

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def find_closest_point_on_path(self, x, y):
        closest_point = None
        min_dist = float("inf")
        for point in self.path:
            dx = point[0] - x
            dy = point[1] - y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        return closest_point

    def find_lookahead_point(self, x, y):
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

    def transform_to_vehicle_frame(self, point, x, y, yaw):
        dx = point[0] - x
        dy = point[1] - y
        transformed_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        transformed_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return transformed_x, transformed_y

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dynamic_lookahead"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = point[2]  # velocity stored here
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.lookahead_marker_pub.publish(marker)
        self.marker = marker

    def publish_lookahead_circle(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dynamic_lookahead"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        resolution = 60
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            px = x + self.lookahead_distance * math.cos(angle)
            py = y + self.lookahead_distance * math.sin(angle)
            p = Point()
            p.x = px
            p.y = py
            p.z = 0.05
            marker.points.append(p)

        self.lookahead_circle_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicLookahead()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
