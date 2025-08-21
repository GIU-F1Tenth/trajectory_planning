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
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue
from std_msgs.msg import Header
import numpy as np
from scipy.ndimage import grey_dilation
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
import math

def euler_from_quaternion(quaternion):
    """
    Convert quaternion to Euler angles.

    Converts quaternion (w in last place) to euler roll, pitch, yaw.
    This should be replaced when porting for ROS 2 Python tf_conversions is done.

    Args:
        quaternion (list): Quaternion as [x, y, z, w]

    Returns:
        tuple: (roll, pitch, yaw) in radians
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

class GraphNode:
    """
    A node in the search graph for Hybrid A* pathfinding.

    Represents a single cell in the occupancy grid with associated costs
    and heuristic values for the Hybrid A* algorithm.
    """

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

    def normalize_and_discretize_angle(self, theta, bins=72):
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
        discretization_step = math.radians(360 / bins)

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
        super().__init__("a_star_node")

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
        self.declare_parameter("search_angle", 60)  # degrees
        self.declare_parameter("search_step", 10)  # degrees    
        self.declare_parameter("vehicle_length", 0.8)  # meters
        self.declare_parameter("velocity", 0.4)  # m/s
        self.declare_parameter("coordinates_tolerance", 1)  # cells
        self.declare_parameter("yaw_tolerance", 5)  # degrees
        self.declare_parameter("min_forward_cost", 2)  # minimum cost for forward movement
        self.declare_parameter("max_forward_cost", 10)  # maximum cost for forward movement
        self.declare_parameter("min_reverse_cost", 40)  # minimum cost for reverse movement
        self.declare_parameter("max_reverse_cost", 50)  # maximum cost for reverse
        
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

        # Set up subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.map_callback, map_qos
        )
        self.point_sub = self.create_subscription(
            Marker, self.marker_topic, self.point_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PointStamped, self.point_topic, self.goal_callback, 10
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, self.visited_map_topic, 10)

        # Initialize map storage
        self.map_ = OccupancyGrid()
        self.visited_map_ = OccupancyGrid()

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
        start_node.heuristic = self.euclidean_distance(start_node, goal_node)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node: GraphNode = pending_nodes.get()
    
            # Goal found!
            if self.goal_reached(active_node, goal_node, self.coordinates_tolerance, math.radians(self.yaw_tolerance)):
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

                return path
            
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
                            pending_nodes.put(old_node) # to update the cost in the queue
                    else:    
                        new_node.cost = active_node.cost + cost + \
                            self.map_.data[self.pose_to_cell(new_node)]
                        new_node.prev = active_node
                        new_node.heuristic = self.euclidean_distance(
                            new_node, goal_node)
                        pending_nodes.put(new_node)
                        self.get_logger().info(f"adding {new_node} size {pending_nodes.qsize()}")
                        visited_nodes[(new_node.x, new_node.y, new_node.theta)] = new_node

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

    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int(
            (pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int(
            (pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        return GraphNode(grid_x, grid_y, yaw)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        pose.position.x = node.x * self.map_.info.resolution + \
            self.map_.info.origin.position.x
        pose.position.y = node.y * self.map_.info.resolution + \
            self.map_.info.origin.position.y
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
