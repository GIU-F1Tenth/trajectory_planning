#!/usr/bin/env python3
"""
Hybrid A* Path Planner Module

This module implements an Hybrid A* path planning algorithm for autonomous navigation.
It subscribes to occupancy grids and goal markers, then computes optimal paths
using the Hybrid A* search algorithm.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue
from std_msgs.msg import Header
import numpy as np
from scipy.ndimage import grey_dilation
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
import math
from scipy.interpolate import CubicSpline
import dubins
import reeds_shepp
from geometry_msgs.msg import Quaternion
import tf_transformations
import time

class GraphNode:
    """
    A node in the search graph for Hybrid A* pathfinding.

    Represents a single cell in the occupancy grid with associated costs
    and heuristic values for the Hybrid A* algorithm.
    """

    bins = 72  # default number of bins for discretization

    def __init__(self, x, y, theta, cost=0, heuristic=0, prev=None):
        """
        Initialize a graph node.

        Args:
            x (int): Grid x-coordinate
            y (int): Grid y-coordinate  
            theta (float): Yaw orientation in radians
            cost (float): Cost to reach this node from start
            heuristic (float): Heuristic estimate to goal
            prev (GraphNode): Previous node in path
        """
        self.x = x
        self.y = y
        self.theta = self.normalize_and_discretize_angle(theta)
        self.cost = cost
        self.heuristic = heuristic
        self.prev = prev

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __hash__(self):
        return hash((self.x, self.y, self.theta))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1], self.theta + other[2])
    
    def __str__(self):   # like toString()
        return f"({self.x}, {self.y}, {math.degrees(self.theta)})"

    def __repr__(self):  # debug version
        return f"({self.x}, {self.y}, {math.degrees(self.theta)})"

    def normalize_and_discretize_angle(self, theta):
        """
        Normalizes an angle to [-π, π] range and discretizes it to 5-degree increments.

        Args:
            theta (float): Input angle in radians
            
        Returns:
            float: Normalized and discretized angle in radians
        """
        # Step 1: Normalize angle to [-π, π] range
        # Use atan2(sin(theta), cos(theta)) for robust normalization
        normalized_theta = math.atan2(math.sin(theta), math.cos(theta))

        # Step 2: Convert 10 degrees to radians
        discretization_step = math.radians(360 / GraphNode.bins)

        # Step 3: Discretize to nearest 10-degree increment
        # Round to nearest multiple of discretization_step
        discretized_theta = round(normalized_theta / discretization_step) * discretization_step
        
        return discretized_theta


class AStarPlanner(Node):
    """
    A ROS2 node implementing the Hybrid A* path planning algorithm.

    This node subscribes to occupancy grids and goal markers, then computes
    optimal paths using the Hybrid A* search algorithm. The computed paths are
    published for use by path following controllers.
    """

    def __init__(self):
        """
        Initialize the AStarPlanner node.

        Sets up subscribers, publishers, parameters, and TF listeners.
        """
        super().__init__("hybrid_Astar_node")

        # Set up TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Configure QoS for map subscription
        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Declare parameters
        self.declare_parameter("is_antiClockwise", True)
        self.declare_parameter("costmap_topic", "/inflated_costmap")
        self.declare_parameter("marker_topic", "/astar_lookahead_marker")
        self.declare_parameter("path_topic", "/pp_path")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "ego_racecar/base_link")
        self.declare_parameter("visited_map_topic", "/visited_map")
        self.declare_parameter("point_topic", "/clicked_point")
        self.declare_parameter("search_angle", 30)  # degrees
        self.declare_parameter("search_step", 5)  # degrees    
        self.declare_parameter("vehicle_length", 0.33)  # meters
        self.declare_parameter("velocity", 0.3)  # m/s
        self.declare_parameter("coordinates_tolerance", 1)  # cells
        self.declare_parameter("yaw_tolerance", 10)  # degrees
        self.declare_parameter("min_forward_cost", 1)  # minimum cost for forward movement
        self.declare_parameter("max_forward_cost", 10)  # maximum cost for forward movement
        self.declare_parameter("min_reverse_cost", 12)  # minimum cost for reverse movement
        self.declare_parameter("max_reverse_cost", 20)  # maximum cost for reverse
        self.declare_parameter("heuristic", "reeds_shepp")  # heuristic type
        self.declare_parameter("interpolation_resolution", 0.1)  # meters
        self.declare_parameter("goal_pose_topic", "/goal_pose")  # topic for goal pose
        self.declare_parameter("expansion_vectors_topic", "/expansion_vectors")  # topic for expansion vectors
        self.declare_parameter("discretization_bins", 36)  # number of bins for discretization

        # Get parameter values
        self.is_antiClockwise = self.get_parameter(
            "is_antiClockwise").get_parameter_value().bool_value
        self.costmap_topic = self.get_parameter(
            "costmap_topic").get_parameter_value().string_value
        self.marker_topic = self.get_parameter(
            "marker_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter(
            "path_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter(
            "map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter(
            "base_frame").get_parameter_value().string_value
        self.visited_map_topic = self.get_parameter(
            "visited_map_topic").get_parameter_value().string_value
        self.point_topic = self.get_parameter(
            "point_topic").get_parameter_value().string_value
        self.search_angle = self.get_parameter(
            "search_angle").get_parameter_value().integer_value
        self.search_step = self.get_parameter(
            "search_step").get_parameter_value().integer_value
        self.vehicle_length = self.get_parameter(
            "vehicle_length").get_parameter_value().double_value
        self.velocity = self.get_parameter(
            "velocity").get_parameter_value().double_value
        self.coordinates_tolerance = self.get_parameter(
            "coordinates_tolerance").get_parameter_value().integer_value
        self.yaw_tolerance = self.get_parameter(
            "yaw_tolerance").get_parameter_value().integer_value
        self.min_forward_cost = self.get_parameter(
            "min_forward_cost").get_parameter_value().integer_value
        self.max_forward_cost = self.get_parameter(
            "max_forward_cost").get_parameter_value().integer_value
        self.min_reverse_cost = self.get_parameter(
            "min_reverse_cost").get_parameter_value().integer_value
        self.max_reverse_cost = self.get_parameter(
            "max_reverse_cost").get_parameter_value().integer_value
        self.heuristic = self.get_parameter(
            "heuristic").get_parameter_value().string_value
        self.interpolation_resolution = self.get_parameter(
            "interpolation_resolution").get_parameter_value().double_value
        self.goal_pose_topic = self.get_parameter(
            "goal_pose_topic").get_parameter_value().string_value
        self.expansion_vectors_topic = self.get_parameter(
            "expansion_vectors_topic").get_parameter_value().string_value
        self.discretization_bins = self.get_parameter(
            "discretization_bins").get_parameter_value().integer_value

        GraphNode.bins = self.discretization_bins  # set bins for GraphNode class

        # Set up subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.map_callback, map_qos
        )
        self.point_sub = self.create_subscription(
            Marker, self.marker_topic, self.point_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PointStamped, self.point_topic, self.goal_point_callback, 10
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, self.goal_pose_topic, self.goal_pose_callback, 10
        )
        self.path_pub = self.create_publisher(
            Path, self.path_topic, 10
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, self.visited_map_topic, 10
        )
        # New publisher for path expansion vectors
        self.vector_pub = self.create_publisher(
            MarkerArray, self.expansion_vectors_topic, 10
        )

        # Initialize map storage
        self.map_ = OccupancyGrid()
        self.visited_map_ = OccupancyGrid()

    def publish_path_arrows(self, path: Path):
        marker_array = MarkerArray()
        marker_id = 0

        for pose_stamped in path.poses:
            marker = Marker()
            marker.header.frame_id = path.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "astar_path"
            marker.id = marker_id
            marker_id += 1

            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose = pose_stamped.pose  # use the pose from the path

            # extract yaw from quaternion to reapply orientation
            q = (
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
            )
            _, _, yaw = tf_transformations.euler_from_quaternion(q)
            q_new = tf_transformations.quaternion_from_euler(0, 0, yaw)
            marker.pose.orientation = Quaternion(x=q_new[0], y=q_new[1], z=q_new[2], w=q_new[3])

            # size and color
            marker.scale.x = 0.2
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.vector_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} arrows for path")

    def map_callback(self, map_msg: OccupancyGrid):
        """
        Callback function for receiving occupancy grid maps.

        Args:
            map_msg (OccupancyGrid): The received occupancy grid message
        """
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def point_callback(self, marker: Marker):
        """
        Callback function for processing goal markers.

        Args:
            marker (Marker): The received marker message containing the goal position
        """
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return
        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

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

    def goal_pose_callback(self, goal: PoseStamped):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        # reset visited map
        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, self.base_frame, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().error("Could not transform from map to base_link")
            return

        # start pose (robot current pose in map frame)
        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        # goal pose (from RViz 2D Goal Pose)
        pose = PoseStamped()
        pose.header = goal.header
        pose.pose.position.x = goal.pose.position.x
        pose.pose.position.y = goal.pose.position.y
        pose.pose.orientation = goal.pose.orientation  # keep goal orientation

        base_theta = tf_transformations.euler_from_quaternion((
            map_to_base_pose.orientation.x,
            map_to_base_pose.orientation.y,
            map_to_base_pose.orientation.z,
            map_to_base_pose.orientation.w,
        ))[2]

        goal_theta = tf_transformations.euler_from_quaternion((
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ))[2]

        self.get_logger().info(
            f"from: ({map_to_base_pose.position.x:.2f}, {map_to_base_pose.position.y:.2f}, {base_theta:.2f}) "
            f"to: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}, {goal_theta:.2f})"
        )

        # measure planning time
        start = time.perf_counter()
        # send to planner
        path = self.plan(map_to_base_pose, pose.pose)
        end = time.perf_counter() 
        self.get_logger().info(f"Planning time: {end - start:.4f} seconds")

        if path and path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def goal_point_callback(self, point: PointStamped):
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
        pose.pose.orientation = map_to_base_pose.orientation # to make the goal orientation the same as the start orientation

        # sending goal pose to the planner
        path = self.plan(map_to_base_pose, pose.pose)
        if path and path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def goal_reached(self, node: GraphNode, goal: GraphNode,
                    pos_tol_cells: int, yaw_tol_rad: float):
        # position tolerance in grid cells
        if abs(node.x - goal.x) > pos_tol_cells or abs(node.y - goal.y) > pos_tol_cells:
            return False
        # orientation tolerance (optional; relax if you only care about position)
        dtheta = math.atan2(math.sin(node.theta - goal.theta), math.cos(node.theta - goal.theta))
        return abs(dtheta) <= yaw_tol_rad

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
        self.publish_path_arrows(path)
        self.get_logger().info(f"Path constructed with {self.heuristic} heuristic and {len(path.poses)} poses")
        return self.interpolate_path_with_spline(path)

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
        # Define possible movement directions
        # (v, delta, cost) +ve is left
        explore_directions = []
        v_forward = self.velocity
        v_reverse = -self.velocity

        for delta in range(-self.search_angle, self.search_angle+1, self.search_step):  
            # cost increases with steering angle magnitude
            cost = self.min_forward_cost + int(abs(delta) / self.search_angle * (self.max_forward_cost - self.min_forward_cost))
            explore_directions.append((v_forward, delta, cost))

        for delta in range(-self.search_angle, self.search_angle+1, self.search_step):
            # reverse is more expensive overall
            cost = self.min_reverse_cost + int(abs(delta) / self.search_angle * (self.max_reverse_cost - self.min_reverse_cost))
            explore_directions.append((v_reverse, delta, cost))

        length = self.vehicle_length
        # Priority queue with custom comparison for Hybrid A* based on cost + heuristic
        pending_nodes = PriorityQueue()
        visited_nodes = {}
        closed_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)
        self.get_logger().info(f"Planning path using Hybrid A*... from {start_node} to {goal_node}, {GraphNode.bins} bins")

        start_node.heuristic = self.compute_heuristic(start_node, goal_node)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node: GraphNode = pending_nodes.get()
    
            # Goal found!
            if self.goal_reached(active_node, goal_node, self.coordinates_tolerance, math.radians(self.yaw_tolerance)):
                return self.construct_path(active_node, goal_node)
            
            for v, delta, cost in explore_directions:
                
                delta = math.radians(delta)
                
                # vehicle kinematics model calculate x, y coordinates based on the old theta not from the newly calculated theta 
                # what we are doing is that we are integrating the velocity x_dot and y_dot over the time step to get the new position assume dt = 1
                # x_dot = v * cos(theta)
                # y_dot = v * sin(theta)
                # theta_dot = (v / length) * tan(delta)
                # integrate you will get the new position

                x = round(v*np.cos(active_node.theta)/self.map_.info.resolution)
                y = round(v*np.sin(active_node.theta)/self.map_.info.resolution)
                theta = (v/length)*np.tan(delta)
                new_node: GraphNode = active_node + (x, y, theta)
                
                if (new_node not in closed_nodes and self.pose_on_map(new_node) and
                        0 <= self.map_.data[self.pose_to_cell(new_node)] < 99):
                    if (new_node.x, new_node.y, new_node.theta) in visited_nodes:
                        old_node: GraphNode = visited_nodes[(new_node.x, new_node.y, new_node.theta)]
                        new_cost = active_node.cost + cost + self.map_.data[self.pose_to_cell(new_node)]
                        if old_node.cost > new_cost:
                            old_node.cost = new_cost 
                            old_node.prev = active_node
                    else:    
                        new_node.cost = active_node.cost + cost + \
                            self.map_.data[self.pose_to_cell(new_node)]
                        new_node.prev = active_node
                        new_node.heuristic = self.compute_heuristic(
                            new_node, goal_node)
                        pending_nodes.put(new_node)
                        visited_nodes[(new_node.x, new_node.y, new_node.theta)] = new_node

            closed_nodes.add(active_node)
            self.visited_map_.data[self.pose_to_cell(active_node)] = np.clip(active_node.cost, -127, 127)
            self.map_pub.publish(self.visited_map_)

        return None

    def manhattan_distance(self, node: GraphNode, goal_node: GraphNode):
        return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)

    def euclidean_distance(self, node: GraphNode, goal_node: GraphNode):
        return ((node.x - goal_node.x)**2 + (node.y - goal_node.y)**2)**0.5

    def dubins_distance(self, node: GraphNode, goal_node: GraphNode):
        start = (node.x, node.y, node.theta)
        end = (goal_node.x, goal_node.y, goal_node.theta)
        path = dubins.shortest_path(start, end, self.vehicle_length)
        return path.path_length()

    def reeds_shepp_distance(self, node: GraphNode, goal_node: GraphNode):
        start = (node.x, node.y, node.theta)
        end = (goal_node.x, goal_node.y, goal_node.theta)
        return reeds_shepp.path_length(start, end, self.vehicle_length)

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

        if self.heuristic == "reeds_shepp":
            return self.reeds_shepp_distance(node, goal_node)
        elif self.heuristic == "dubins":
            return self.dubins_distance(node, goal_node)
        elif self.heuristic == "euclidean":
            return self.euclidean_distance(node, goal_node)
        elif self.heuristic == "manhattan":
            return self.manhattan_distance(node, goal_node)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")

    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int(
            (pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int(
            (pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        return GraphNode(grid_x, grid_y, yaw)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        pose.position.x = node.x * self.map_.info.resolution + \
            self.map_.info.origin.position.x
        pose.position.y = node.y * self.map_.info.resolution + \
            self.map_.info.origin.position.y
        pose.position.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, node.theta)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
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
