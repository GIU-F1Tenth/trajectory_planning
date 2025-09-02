#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue
import time


class GraphNode:
    def __init__(self, x, y, cost=0, prev=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.prev = prev

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1])


class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__("dijkstra_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # -----------------------------
        # Declare parameters
        # -----------------------------
        self.declare_parameter("costmap_topic", "/inflated_costmap")
        self.declare_parameter("goal_topic", "/clicked_point")
        self.declare_parameter("path_topic", "/pp_path")
        self.declare_parameter("visited_map_topic", "/visited_map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "ego_racecar/base_link")

        # Planning parameters
        self.declare_parameter("planning.use_8_connected", True)
        self.declare_parameter("planning.base_movement_cost", 10)
        self.declare_parameter("planning.diagonal_movement_cost", 14)

        # Occupancy thresholds
        self.declare_parameter("occupancy.occupied_threshold", 99)
        self.declare_parameter("occupancy.free_threshold", 0)

        # Logging
        self.declare_parameter("log_level", "INFO")

        # -----------------------------
        # Get parameters
        # -----------------------------
        self.costmap_topic = self.get_parameter("costmap_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.visited_map_topic = self.get_parameter("visited_map_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        self.use_8_connected = self.get_parameter("planning.use_8_connected").get_parameter_value().bool_value
        self.base_movement_cost = self.get_parameter("planning.base_movement_cost").get_parameter_value().integer_value
        self.diagonal_movement_cost = self.get_parameter("planning.diagonal_movement_cost").get_parameter_value().integer_value

        self.occupied_threshold = self.get_parameter("occupancy.occupied_threshold").get_parameter_value().integer_value
        self.free_threshold = self.get_parameter("occupancy.free_threshold").get_parameter_value().integer_value

        # -----------------------------
        # Subscribers & Publishers
        # -----------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.map_callback, map_qos
        )
        self.pose_sub = self.create_subscription(
            PointStamped, self.goal_topic, self.goal_callback, 10
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, self.visited_map_topic, 10)

        self.map_ = None
        self.visited_map_ = OccupancyGrid()

    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

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
            self.get_logger().error(f"Could not transform from map to {self.base_frame}")
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

    def plan(self, start: Pose, goal: Pose):
        # Movement directions
        if self.use_8_connected:
            explore_directions = [
                (-1, 0, self.base_movement_cost), (1, 0, self.base_movement_cost),
                (0, -1, self.base_movement_cost), (0, 1, self.base_movement_cost),
                (-1, -1, self.diagonal_movement_cost), (-1, 1, self.diagonal_movement_cost),
                (1, -1, self.diagonal_movement_cost), (1, 1, self.diagonal_movement_cost)
            ]
        else:
            explore_directions = [
                (-1, 0, self.base_movement_cost), (1, 0, self.base_movement_cost),
                (0, -1, self.base_movement_cost), (0, 1, self.base_movement_cost)
            ]

        pending_nodes = PriorityQueue()
        visited_nodes = {}
        closed_nodes = set()

        start_node = self.world_to_grid(start)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node: GraphNode = pending_nodes.get()

            # Goal found!
            if active_node == self.world_to_grid(goal):
                break

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
                        pending_nodes.put(new_node)
                        visited_nodes[(new_node.x, new_node.y)] = new_node

            self.visited_map_.data[self.pose_to_cell(active_node)] = 10
            self.map_pub.publish(self.visited_map_)

        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        while active_node and active_node.prev and rclpy.ok():
            last_pose: Pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        path.poses.reverse()
        return path

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
    node = DijkstraPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
