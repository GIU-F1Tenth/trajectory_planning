# !/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Parameters
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('look_ahead_distance', 1.0)
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('max_steering_angle_deg', 35.0)

        self.wheelbase = self.get_parameter('wheelbase').value
        self.look_ahead_distance = self.get_parameter('look_ahead_distance').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering_angle = np.deg2rad(self.get_parameter('max_steering_angle_deg').value)

        # Waypoints from uploaded file
        self.waypoints = [
            (-0.6561624, 6.1064005),
            (-0.6159769, 5.9089623),
            (-0.5513409, 5.7204395),
            (-0.4678902, 5.5421112),
            (-0.3688091, 5.3697715),
            (-0.2584882, 5.2042210),
            (-0.1395495, 5.0456501),
            (-0.0123666, 4.8933928),
            (0.1227745, 4.7473836),
            (0.2653068, 4.6074942),
            (0.4128141, 4.4750002),
            (0.5657695, 4.3486218),
            (0.7242536, 4.2276939),
            (0.8875704, 4.1121535),
            (1.0527333, 4.0033015),
            (1.2223268, 3.8987539),
            (1.3954097, 3.7986574),
            (1.5693229, 3.7040291),
            (1.7469697, 3.6128878),
            (1.9259663, 3.5261704),
            (2.1067359, 3.4433713),
            (2.2901726, 3.3638920),
            (2.4735587, 3.2887039),
            (2.6598602, 3.2164321),
            (2.8461674, 3.1480839),
            (3.0345784, 3.0827674),
            (3.2236193, 3.0209131),
            (3.4139069, 2.9622299),
            (3.6051798, 2.9067312),
            (3.7971477, 2.8544291),
            (3.9902397, 2.8051407),
            (4.1837008, 2.7589892),
            (4.3782603, 2.7157268),
            (4.5730275, 2.6754760),
            (4.7687488, 2.6379994),
            (4.9646287, 2.6033682),
            (5.1612522, 2.5713917),
            (5.3580705, 2.5420753),
            (5.5553666, 2.5152813),
            (5.7529389, 2.4909540),
            (5.9507477, 2.4690395),
            (6.1488218, 2.4494651),
            (6.3470909, 2.4320995),
            (6.5454973, 2.4167773),
            (6.7441193, 2.4032830),
            (6.9427779, 2.3914247),
            (7.1415841, 2.3810222),
            (7.3404073, 2.3719167),
            (7.5393116, 2.3639441),
            (7.7382385, 2.3569525),
            (7.9372032, 2.3507981),
            (8.1361928, 2.3453488),
            (8.3351908, 2.3404865),
            (8.5342126, 2.3360928),
            (8.7332374, 2.3320480),
            (8.9322640, 2.3282426),
            (9.1313223, 2.3245790),
            (9.3303249, 2.3209855),
            (9.5294386, 2.3174099),
            (9.7283786, 2.3138324),
            (9.9275839, 2.3102414),
            (10.1264028, 2.3066698),
            (10.3257661, 2.3031389),
            (10.5243703, 2.2997151),
            (10.7239461, 2.2964162),
            (10.9222610, 2.2933289),
            (11.1221391, 2.2904843),
            (11.3207032, 2.2880840),
            (11.5203242, 2.2862875),
            (11.7193101, 2.2853081),
            (11.9182173, 2.2853723),
            (12.1178450, 2.2867740),
            (12.3154658, 2.2898259),
            (12.5158102, 2.2950325),
            (12.7142764, 2.3027946),
            (12.9125335, 2.3137099),
            (13.1111253, 2.3284478),
            (13.3082276, 2.3475960),
            (13.5056788, 2.3718223),
            (13.6986106, 2.4006139),
            (13.8957036, 2.4370505),
            (14.0823887, 2.4790349),
            (14.2779523, 2.5300552),
            (14.4490518, 2.5854930),
            (14.6392017, 2.6744367),
            (14.8144026, 2.7894604),
            (14.9631691, 2.9245777),
            (15.0774103, 3.0894055),
            (15.1278604, 3.2803705),
            (15.1049987, 3.4600523),
            (15.0278347, 3.6420608),
            (14.9075276, 3.8113427),
            (14.7552367, 3.9528448),
            (14.5662634, 4.0608714),
            (14.3649530, 4.1267647),
            (14.1715363, 4.1619679),
            (14.0005473, 4.1805921),
            (13.7656063, 4.2061542),
            (13.6064147, 4.2348433),
            (13.4176667, 4.2923560),
            (13.2314885, 4.3613910),
            (13.0458377, 4.4289903),
            (12.8572955, 4.4908253),
            (12.6656061, 4.5454658),
            (12.4705460, 4.5928339),
            (12.2769207, 4.6323801),
            (12.0808477, 4.6657252),
            (11.8821157, 4.6934717),
            (11.6855721, 4.7156244),
            (11.4869337, 4.7332726),
            (11.2878580, 4.7466880),
            (11.0895126, 4.7562231),
            (10.8895626, 4.7622044),
            (10.6912993, 4.7647354),
            (10.4918434, 4.7642644),
            (10.2930560, 4.7612395),
            (10.0940036, 4.7561554),
            (9.8949333, 4.7494502),
            (9.6961856, 4.7415052),
            (9.4969933, 4.7326483),
            (9.2984270, 4.7232811),
            (9.0992010, 4.7137008),
            (8.9006566, 4.7043295),
            (8.7014526, 4.6954852),
            (8.5027765, 4.6876062),
            (8.3036476, 4.6810462),
            (8.1047201, 4.6762197),
            (7.9056147, 4.6735003),
            (7.7065407, 4.6732678),
            (7.5075035, 4.6758761),
            (7.3084851, 4.6816535),
            (7.1096730, 4.6908691),
            (6.9110063, 4.7038023),
            (6.7127313, 4.7207223),
            (6.5148362, 4.7419372),
            (6.3174514, 4.7677640),
            (6.1208204, 4.7985270),
            (5.9248960, 4.8346099),
            (5.7303175, 4.8758721),
            (5.5364853, 4.9221676),
            (5.3443989, 4.9730852),
            (5.1529322, 5.0288130),
            (4.9636085, 5.0887945),
            (4.7747720, 5.1534630),
            (4.5882738, 5.2220972),
            (4.4026958, 5.2951311),
            (4.2191796, 5.3720567),
            (4.0373418, 5.4529734),
            (3.8569594, 5.5379729),
            (3.6794664, 5.6263647),
            (3.5024510, 5.7194269),
            (3.3289571, 5.8157099),
            (3.1569323, 5.9165525),
            (2.9876933, 6.0215343),
            (2.8223931, 6.1304909),
            (2.6586138, 6.2451832),
            (2.4975031, 6.3619022),
            (2.3369938, 6.4784174),
            (2.1740081, 6.5936788),
            (2.0093822, 6.7052276),
            (1.8422114, 6.8123675),
            (1.6712298, 6.9145490),
            (1.4969780, 7.0098415),
            (1.3182833, 7.0969848),
            (1.1349307, 7.1738087),
            (0.9477682, 7.2373866),
            (0.7536350, 7.2848546),
            (0.5566912, 7.3115561),
            (0.3569183, 7.3142466),
            (0.1592140, 7.2893845),
            (-0.0321574, 7.2329913),
            (-0.2089902, 7.1431121),
            (-0.3647084, 7.0198978),
            (-0.4921635, 6.8676130),
            (-0.5861365, 6.6926971),
            (-0.6439747, 6.5025776),
            (-0.6663915, 6.3040660),
            (-0.6561624, 6.1064005)
        ]


        self.current_waypoint_idx = 0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Subscribers and Publishers
        self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        self.path_publisher = self.create_publisher(Path, '/path', 10)

        self.get_logger().info("Pure Pursuit Node Initialized.")

    def pose_callback(self, pose_msg):
        # Extract position and orientation
        self.current_x = pose_msg.pose.pose.position.x
        self.current_y = pose_msg.pose.pose.position.y
        orientation = pose_msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(orientation)

        # Publish path for visualization
        self.publish_path()

        # Get lookahead point
        lookahead_point = self.get_lookahead_point()
        if lookahead_point:
            steering_angle = self.compute_steering_angle(lookahead_point)
            self.publish_drive_command(steering_angle, self.max_speed)
        else:
            self.get_logger().warn("No valid lookahead point found!")

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for (x, y) in self.waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        self.path_publisher.publish(path_msg)

    def get_lookahead_point(self):
        for i in range(len(self.waypoints)):
            idx = (self.current_waypoint_idx + i) % len(self.waypoints)
            wpx, wpy = self.waypoints[idx]
            dist = math.sqrt((wpx - self.current_x)**2 + (wpy - self.current_y)**2)
            if dist >= self.look_ahead_distance:
                self.current_waypoint_idx = idx
                return (wpx, wpy)
        return None

    def compute_steering_angle(self, target_point):
        dx = target_point[0] - self.current_x
        dy = target_point[1] - self.current_y

        # Transform to vehicle coordinates
        local_x = dx * math.cos(-self.current_yaw) - dy * math.sin(-self.current_yaw)
        local_y = dx * math.sin(-self.current_yaw) + dy * math.cos(-self.current_yaw)

        if local_x <= 0.0:
            # If point is behind, return max steering
            return math.copysign(self.max_steering_angle, local_y)

        alpha = math.atan2(local_y, local_x)
        L = self.wheelbase
        Ld = math.sqrt(local_x**2 + local_y**2)
        steering_angle = math.atan2(2.0 * L * math.sin(alpha), Ld)
        return max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))

    def publish_drive_command(self, steering_angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_publisher.publish(drive_msg)

    @staticmethod
    def quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
