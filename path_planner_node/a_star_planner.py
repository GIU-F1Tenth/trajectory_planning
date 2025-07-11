#!/usr/bin/env python3

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

class GraphNode:
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

class AStarPlanner(Node):
    def __init__(self):
        super().__init__("a_star_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.declare_parameter("is_antiClockwise", True)

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/costmap/costmap", self.map_callback, map_qos
        )
        # self.pose_sub = self.create_subscription(
        #     PoseStamped, "/goal_pose", self.goal_callback, 10
        # )
        self.point_sub = self.create_subscription(
            Marker, "/astar_lookahead_marker", self.point_callback, 10
        )
        self.path_pub = self.create_publisher(Path, "/pp_path", 10)
        # self.map_pub = self.create_publisher(OccupancyGrid, "/a_star/visited_map", 10)

        self.is_antiClockwise = self.get_parameter("is_antiClockwise").get_parameter_value().string_value
        self.map_ = None
        # self.visited_map_ = OccupancyGrid()
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return
        # self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)
        
        pose = PoseStamped()
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y
        
        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, "laser", rclpy.time.Time()
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
            self.get_logger().warn("No path")
                                   
    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg
        # self.map_ = self.create_cspace(map_msg)
        # self.visited_map_.header.frame_id = map_msg.header.frame_id
        # self.visited_map_.info = map_msg.info
        # self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def point_callback(self, marker: Marker):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return
        # self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)
        
        pose = PoseStamped()
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y
        
        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, "laser", rclpy.time.Time()
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

    def goal_callback(self, pose: PoseStamped):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        # self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, "base_link", rclpy.time.Time()
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

    def plan(self, start: Pose, goal: Pose):
        # Define possible movement directions
        # explore_directions = [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 1), (1, -1), (-1, -1)] # my addition for the perfect path
        explore_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        # Priority queue with custom comparison for A* based on cost + heuristic
        pending_nodes = PriorityQueue()
        visited_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)
        start_node.heuristic = self.manhattan_distance(start_node, goal_node)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node = pending_nodes.get()

            # Goal found!
            if active_node == goal_node:
                break
            
            # Explore neighbors
            # for dir_x, dir_y in explore_directions:
            #     new_node: GraphNode = active_node + (dir_x, dir_y)
            for dir_x, dir_y in explore_directions:
                new_node: GraphNode = active_node + (dir_x, dir_y)

                # Transform the grid cell to world coordinates
                world_pose = self.grid_to_world(new_node)

                try:
                    # Transform point into base_link frame
                    transform = self.tf_buffer.lookup_transform("base_link", self.map_.header.frame_id, rclpy.time.Time())
                    transformed = do_transform_pose(world_pose, transform)
                    x_in_base = transformed.position.x
                except (LookupException, Exception) as e:
                    self.get_logger().warn(f"TF failed: {e}")
                    continue

                # Reject points behind the car
                if x_in_base < 0:
                    visited_nodes.add(new_node)
                    continue

                # if new_node.cost == 0: # if the node is newely created so we set its value with its value in the map "my addition"
                #     new_node.cost = self.map_.data[self.pose_to_cell(new_node)] # my addition

                if (new_node not in visited_nodes and self.pose_on_map(new_node) and 
                    0 <= self.map_.data[self.pose_to_cell(new_node)] < 99):
                    
                    new_node.cost = active_node.cost + 1 + self.map_.data[self.pose_to_cell(new_node)]
                    # if abs(dir_y) == 1 and abs(dir_x) == 1: # my addition for the perfect path, if on the edges
                        # new_node.cost = min(active_node.cost + 14 + self.map_.data[self.pose_to_cell(new_node)], new_node.cost) # my addition for the perfect path    
                    # else: # my addition for the perfect path
                        # new_node.cost = min(active_node.cost + 10 + self.map_.data[self.pose_to_cell(new_node)], new_node.cost) # my addition for the perfect path
                    new_node.heuristic = self.manhattan_distance(new_node, goal_node)
                    new_node.prev = active_node

                    pending_nodes.put(new_node)
                    visited_nodes.add(new_node)
                    # visited_nodes.add(active_node) # my addition for the perfect path

            # self.visited_map_.data[self.pose_to_cell(active_node)] = -106
            # self.map_pub.publish(self.visited_map_)

        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        while active_node and active_node.prev and rclpy.ok():
            last_pose: Pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        if self.is_antiClockwise == False:
            path.poses.reverse()
        return path

    def manhattan_distance(self, node: GraphNode, goal_node: GraphNode):
        return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)

    def pose_on_map(self, node: GraphNode):
        return 0 <= node.x < self.map_.info.width and 0 <= node.y < self.map_.info.height

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


