#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32

class MinimalDistanceNode(Node):

    def __init__(self):
        super().__init__('minimal_distance_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.publisher_distance = self.create_publisher(Float32, '/distance_to_obstacle', 10)
        self.publisher_collision = self.create_publisher(Bool, '/collision_detected', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def listener_callback(self, msg):
        min_distance = min(msg.ranges)
        self.publisher_distance.publish(Float32(data=min_distance))
        self.get_logger().info(f'Distance to Obstacle: {min_distance:.2f} metra')
        if min_distance <= 0.4:
            self.get_logger().info('Detected collision, publishing collision message...')
            self.publisher_collision.publish(Bool(data=True))

    def timer_callback(self):
        self.get_logger().debug('Timer callback')

def main(args=None):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    rclpy.init(args=args)
    minimal_distance_node = MinimalDistanceNode()
    rclpy.spin(minimal_distance_node)
    minimal_distance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
