#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import math
import csv
import os
# from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from tf2_ros import Buffer, TransformListener

class AstarLookahead(Node):
    def __init__(self):
        super().__init__("astar_lookahead_pub_node")

        self.declare_parameter("lookahead_distance", 0.0)
        self.declare_parameter("lookahead_marker_topic", "")
        self.declare_parameter("csv_path", "")   
        self.declare_parameter("astar_pp_path", "")

        self.lookahead_distance = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        self.marker_pub_topic = self.get_parameter("lookahead_marker_topic").get_parameter_value().string_value
        self.csv_path = self.get_parameter("csv_path").get_parameter_value().string_value
        self.astar_pp_path = self.get_parameter("astar_pp_path").get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.001, self.get_pose)  # 500 Hz
        self.path = [] # a tuple of (x, y, v) 
        self.path = self.load_path_from_csv(self.csv_path)

        self.lookahead_marker_pub = self.create_publisher(Marker, self.marker_pub_topic, 10)
        self.lookahead_circle_pub = self.create_publisher(Marker, "/astar_lookahead_circle", 10)
        self.path_publisher = self.create_publisher(Path, self.astar_pp_path, 10)
        self.marker = None

    def load_path_from_csv(self, csv_path):
        path = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x, y, v = float(row[0]), float(row[1]), float(row[2])
                path.append((x, y, v))
        return path

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
        self.marker = marker

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