#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class MinimalDistanceNode(Node):

    def __init__(self):
        super().__init__('minimal_distance_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        min_distance = min(msg.ranges)
        self.get_logger().info(f'Najbliższa przeszkoda jest w odległości: {min_distance:.2f} metra')

def main(args=None):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    rclpy.init(args=args)
    minimal_distance_node = MinimalDistanceNode()
    rclpy.spin(minimal_distance_node)
    minimal_distance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
