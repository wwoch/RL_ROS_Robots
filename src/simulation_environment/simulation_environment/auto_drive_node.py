#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import random
from std_srvs.srv import Empty
import math
import json

class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Bool, '/obstacle_detected', self.obstacle_callback, 10)
        self.timer = self.create_timer(1, self.timer_callback)
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        self.current_direction = random.uniform(-(math.pi/2), math.pi/2)
        self.current_velocity = 0.5
        self.distance_to_obstacle = None  # Zmienna do przechowywania odległości od przeszkody
        self.publish_velocity()

    def obstacle_callback(self, msg):
        if msg.data:
            self.distance_to_obstacle = random.uniform(0.5, 5.0)  # Symulacja odczytu odległości
            self.get_logger().info('Obstacle detected, resetting simulation...')
            self.reset_simulation()
            self.change_direction()
            self.save_data_to_json()

    def timer_callback(self):
        self.change_direction()
        self.save_data_to_json()  # Zapisz dane przy każdej zmianie kierunku

    def reset_simulation(self):
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset simulation service not available, waiting again...')
        reset_request = Empty.Request()
        future = self.reset_simulation_client.call_async(reset_request)
        future.add_done_callback(self.future_callback)

    def future_callback(self, future):
        try:
            future.result()
            self.get_logger().info('Simulation reset successfully')
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def change_direction(self):
        self.current_direction = random.uniform(-(math.pi/4), math.pi/4)
        self.publish_velocity()

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = self.current_velocity
        msg.angular.z = self.current_direction
        self.publisher_.publish(msg)
        self.get_logger().info(f'Changed driving direction to: {self.current_direction:.2f}, moving forward')

    def save_data_to_json(self):
        data = {
            "distance_to_obstacle": self.distance_to_obstacle,
            "current_direction": self.current_direction
        }
        try:
            with open('driving_data.json', 'r') as json_file:
                data_history = json.load(json_file)
        except FileNotFoundError:
            data_history = []

        data_history.append(data)

        with open('driving_data.json', 'w') as json_file:
            json.dump(data_history, json_file, indent=4)

def main(args=None):
    rclpy.init(args=args)
    auto_drive_node = AutoDriveNode()
    rclpy.spin(auto_drive_node)
    auto_drive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
