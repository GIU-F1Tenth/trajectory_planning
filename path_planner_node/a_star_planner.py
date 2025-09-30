#!/usr/bin/env python3
"""
A* Path Planner Module

This module implements an A* path planning algorithm for autonomous navigation.
It subscribes to occupancy grids and goal markers, then computes optimal paths
using the A* search algorithm.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue
import time
import numpy as np
from scipy.interpolate import CubicSpline
import math
import dubins
import reeds_shepp

class GraphNode:
    """A node in the search graph for A* pathfinding."""

    def __init__(self, x, y, cost=0, heuristic=0, prev=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.prev = prev

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1])
    
    def __str__(self):   # like toString()
        return f"({self.x}, {self.y})"

    def __repr__(self):  # debug version
        return f"({self.x}, {self.y})"
    
class AStarPlanner(Node):
    """A ROS2 node implementing the A* path planning algorithm."""

    def __init__(self):
        super().__init__("a_star_node")

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS for map subscription
        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # -------------------------------
        # Parameter Declarations
        # -------------------------------
        # General parameters
        self.declare_parameter("is_antiClockwise", False)
        self.declare_parameter("costmap_topic", "/costmap/costmap")
        self.declare_parameter("marker_topic", "/astar_lookahead_marker")
        self.declare_parameter("path_topic", "/pp_path")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "laser")
        self.declare_parameter("log_level", "INFO")
        self.declare_parameter("point_topic", "/clicked_point")
        self.declare_parameter("visited_map_topic", "/visited_map")
        self.declare_parameter("heuristic", "reeds_shepp")  # heuristic type

        # Planning namespace
        self.declare_parameter("planning.use_8_connected", False)
        self.declare_parameter("planning.base_movement_cost", 1)
        self.declare_parameter("planning.diagonal_movement_cost", 14)
        self.declare_parameter("planning.reject_behind_robot", True)
        self.declare_parameter("planning.interpolation_resolution", 0.1)  # meters
        self.declare_parameter("planning.coordinates_tolerance", 1)  # cells

        # Occupancy namespace
        self.declare_parameter("occupancy.occupied_threshold", 99)
        self.declare_parameter("occupancy.free_threshold", 0)

        # -------------------------------
        # Parameter Retrieval
        # -------------------------------
        self.is_antiClockwise = self.get_parameter("is_antiClockwise").get_parameter_value().bool_value
        self.costmap_topic = self.get_parameter("costmap_topic").get_parameter_value().string_value
        self.marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.log_level = self.get_parameter("log_level").get_parameter_value().string_value
        self.point_topic = self.get_parameter("point_topic").get_parameter_value().string_value
        self.visited_map_topic = self.get_parameter("visited_map_topic").get_parameter_value().string_value
        self.heuristic = self.get_parameter("heuristic").get_parameter_value().string_value
        
        # Planning
        self.use_8_connected = self.get_parameter("planning.use_8_connected").get_parameter_value().bool_value
        self.base_movement_cost = self.get_parameter("planning.base_movement_cost").get_parameter_value().integer_value
        self.diagonal_movement_cost = self.get_parameter("planning.diagonal_movement_cost").get_parameter_value().integer_value
        self.reject_behind_robot = self.get_parameter("planning.reject_behind_robot").get_parameter_value().bool_value
        self.interpolation_resolution = self.get_parameter("planning.interpolation_resolution").get_parameter_value().double_value
        self.coordinates_tolerance = self.get_parameter("planning.coordinates_tolerance").get_parameter_value().integer_value

        # Occupancy
        self.occupied_threshold = self.get_parameter("occupancy.occupied_threshold").get_parameter_value().integer_value
        self.free_threshold = self.get_parameter("occupancy.free_threshold").get_parameter_value().integer_value

        # -------------------------------
        # Subscribers & Publishers
        # -------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.map_callback, map_qos
        )
        self.point_sub = self.create_subscription(
            Marker, self.marker_topic, self.point_callback, 10
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.goal_sub = self.create_subscription(
            PointStamped, self.point_topic, self.goal_callback, 10
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, self.visited_map_topic, 10
        )

        # Initialize maps
        self.map_ = None
        self.visited_map_ = OccupancyGrid()

    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def point_callback(self, marker: Marker):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        pose = PoseStamped()
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y
        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, self.base_frame, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().error("Could not transform from map to base_link")
            return

        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        path = self.plan(map_to_base_pose, pose.pose)
        if path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def goal_callback(self, point: PointStamped):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, self.base_frame, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().error("Could not transform from map to base_link")
            return

        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        pose = PoseStamped()
        pose.pose.position.x = point.point.x
        pose.pose.position.y = point.point.y

        self.get_logger().info(f"from: ({map_to_base_pose.position.x:.2f}, {map_to_base_pose.position.y:.2f}) to: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})")

        # measure planning time
        start = time.perf_counter()
        # send to planner
        path = self.plan(map_to_base_pose, pose.pose)
        end = time.perf_counter()
        self.get_logger().info(f"Planning time: {end - start:.4f} seconds")

        if path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def compute_path_length(self, path: Path) -> float:
        total_length = 0.0
        for i in range(1, len(path.poses)):
            p1 = path.poses[i-1].pose.position
            p2 = path.poses[i].pose.position
            segment_length = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            total_length += segment_length
        return total_length

    def goal_reached(self, node: GraphNode, goal: GraphNode,
                    pos_tol_cells: int):
        # position tolerance in grid cells
        if abs(node.x - goal.x) > pos_tol_cells or abs(node.y - goal.y) > pos_tol_cells:
            return False
        return True

    def construct_path(self, active_node: GraphNode, goal_node: GraphNode) -> Path:
        self.get_logger().info(f"active node: {active_node} goal node: {goal_node}")
        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        while active_node and rclpy.ok():
            last_pose: Pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        if self.is_antiClockwise == False:
            path.poses.reverse()
        interpolated_path: Path = self.interpolate_path_with_spline(path)
        path_length = self.compute_path_length(interpolated_path)
        self.get_logger().info(f"Path constructed with {self.heuristic} heuristic and {len(path.poses)} poses ({len(interpolated_path.poses)} after interpolation), length: {path_length:.2f} meters")
        return interpolated_path
    
    def interpolate_path_with_spline(self, path: Path) -> Path:
        """
        Takes a ROS 2 Path and applies natural cubic spline interpolation.
        Number of interpolation points is computed from path length and resolution.
        
        Args:
            path (Path): Input ROS2 path containing poses.
            resolution (float): Desired spacing between interpolated points (meters).

        Returns:
            Path: A new Path with spline-interpolated poses.
        """
        if len(path.poses) < 2:
            raise ValueError("Path must contain at least 2 poses for interpolation")
        resolution = self.interpolation_resolution
        if resolution <= 0:
            raise ValueError("Resolution must be a positive value")
        
        # Extract x, y coordinates
        x = np.array([pose.pose.position.x for pose in path.poses])
        y = np.array([pose.pose.position.y for pose in path.poses])

        # Use cumulative distance as parameter (arc-length-like)
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        t = np.concatenate(([0], np.cumsum(distances)))

        # Total path length
        path_length = t[-1]

        # Compute number of points automatically
        num_points = max(2, int(path_length / resolution))
        self.get_logger().info(f"Interpolating path with {num_points} points at {resolution} m resolution")
        # Build natural cubic splines
        spline_x = CubicSpline(t, x, bc_type="natural")
        spline_y = CubicSpline(t, y, bc_type="natural")

        # Resample at evenly spaced points
        t_new = np.linspace(0, t[-1], num_points)
        x_new = spline_x(t_new)
        y_new = spline_y(t_new)

        # Build new Path message
        new_path = Path()
        new_path.header = path.header

        for xi, yi in zip(x_new, y_new):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(xi)
            pose.pose.position.y = float(yi)
            pose.pose.position.z = 0.0  # keep 2D path
            new_path.poses.append(pose)

        return new_path

    def plan(self, start: Pose, goal: Pose):
        # Build movement directions based on parameters
        if self.use_8_connected:
            explore_directions = [
                (-1, 0, self.base_movement_cost),
                (1, 0, self.base_movement_cost),
                (0, -1, self.base_movement_cost),
                (0, 1, self.base_movement_cost),
                (-1, 1, self.diagonal_movement_cost),
                (1, 1, self.diagonal_movement_cost),
                (1, -1, self.diagonal_movement_cost),
                (-1, -1, self.diagonal_movement_cost)
            ]
        else:
            explore_directions = [
                (-1, 0, self.base_movement_cost),
                (1, 0, self.base_movement_cost),
                (0, -1, self.base_movement_cost),
                (0, 1, self.base_movement_cost)
            ]

        # Priority queue
        pending_nodes = PriorityQueue()
        visited_nodes = {}
        closed_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)
        start_node.heuristic = self.compute_heuristic(start_node, goal_node)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node: GraphNode = pending_nodes.get()

            if self.goal_reached(active_node, goal_node, self.coordinates_tolerance):
                return self.construct_path(active_node, goal_node)

            for dir_x, dir_y, cost in explore_directions:
                new_node: GraphNode = active_node + (dir_x, dir_y)

                if (new_node not in closed_nodes and self.pose_on_map(new_node) and
                        self.free_threshold <= self.map_.data[self.pose_to_cell(new_node)] < self.occupied_threshold):
                    if (new_node.x, new_node.y) in visited_nodes:
                        new_node = visited_nodes[(new_node.x, new_node.y)]
                        new_cost = active_node.cost + cost + self.map_.data[self.pose_to_cell(new_node)]
                        if new_node.cost > new_cost:
                            new_node.cost = new_cost
                            new_node.prev = active_node
                    else:
                        new_node.cost = active_node.cost + cost + self.map_.data[self.pose_to_cell(new_node)]
                        new_node.prev = active_node
                        new_node.heuristic = self.compute_heuristic(new_node, goal_node)
                        pending_nodes.put(new_node)
                        visited_nodes[(new_node.x, new_node.y)] = new_node

            closed_nodes.add(active_node)
            self.visited_map_.data[self.pose_to_cell(active_node)] = -106
            self.map_pub.publish(self.visited_map_)

        return None

    def manhattan_distance(self, node: GraphNode, goal_node: GraphNode):
        return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)

    def euclidean_distance(self, node: GraphNode, goal_node: GraphNode):
        return ((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)**0.5

    def pose_on_map(self, node: GraphNode):
        return 0 <= node.x < self.map_.info.width and 0 <= node.y < self.map_.info.height

    def compute_heuristic(self, node: GraphNode, goal_node: GraphNode):
        """
        Compute heuristic value for a node based on its distance to the goal.
        "reeds_shepp", "dubins", "euclidean", "manhattan"

        Args:
            node (GraphNode): Current node in the search graph
            goal_node (GraphNode): Goal node in the search graph

        Returns:
            float: Heuristic value (distance) from current node to goal node
        """

        if self.heuristic == "euclidean":
            return self.euclidean_distance(node, goal_node)
        elif self.heuristic == "manhattan":
            return self.manhattan_distance(node, goal_node)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")


    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int((pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int((pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        return GraphNode(grid_x, grid_y)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        pose.position.x = node.x * self.map_.info.resolution + self.map_.info.origin.position.x
        pose.position.y = node.y * self.map_.info.resolution + self.map_.info.origin.position.y
        return pose

    def pose_to_cell(self, node: GraphNode):
        return node.y * self.map_.info.width + node.x


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
