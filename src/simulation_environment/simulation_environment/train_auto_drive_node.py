#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import random
from std_srvs.srv import Empty
import math
import json
from rclpy.time import Time

class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')
        self.robot_type = self.declare_parameter('robot_type', 'A').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_distance = self.create_subscription(Float32, '/distance_to_obstacle', self.distance_callback, 10)
        self.subscription_collision = self.create_subscription(Bool, '/collision_detected', self.obstacle_callback, 10)
        self.publisher_stop_lidar = self.create_publisher(Bool, '/stop_lidar', 10)

        self.data_timer = self.create_timer(0.2, self.data_timer_callback)
        self.direction_timer = self.create_timer(1.0, self.direction_timer_callback)
        self.time_check_timer = self.create_timer(0.1, self.time_check_callback)
        self.current_direction = random.uniform(-(math.pi/2), math.pi/2)
        self.current_velocity = 0.5
        self.distance_to_obstacle = None
        self.previous_distance = None
        self.previous_distance_status = None
 
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        self.simulation_count = 0
        self.simulation_limit = 50
        self.simulation_start_time = self.get_clock().now()  
        self.simulation_duration = None  
        self.simulation_time_limit = 5.0
        
        self.publish_velocity()
        self.clear_json_file()

    def distance_callback(self, msg):
        self.distance_to_obstacle = msg.data
        self.get_logger().info(f'Distance to obstacle: {self.distance_to_string(self.distance_to_obstacle)}')

    def obstacle_callback(self, msg):
        if msg.data:
            self.get_logger().info('Obstacle detected, resetting simulation...')
            self.reset_simulation_due_to_obstacle()

    def direction_timer_callback(self):
        self.change_direction()

    def data_timer_callback(self):
        self.save_data_to_json()

    def time_check_callback(self):
        elapsed_time = (self.get_clock().now() - self.simulation_start_time).nanoseconds / 1e9
        if elapsed_time > self.simulation_time_limit:
            self.get_logger().info('Simulation time limit reached, resetting simulation...')
            self.reset_simulation_due_to_time()

    def reset_simulation(self):
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset simulation service not available, waiting again...')
        reset_request = Empty.Request()
        future = self.reset_simulation_client.call_async(reset_request)
        future.add_done_callback(self.future_callback)

    def reset_simulation_due_to_obstacle(self):
        self.simulation_end_time = self.get_clock().now()
        self.simulation_duration = self.simulation_end_time - self.simulation_start_time
        self.save_data_to_json(final_reward=True)
        self.simulation_count += 1

        if self.simulation_count >= self.simulation_limit:
            self.get_logger().info('Simulation limit reached, shutting down...')
            self.publisher_stop_lidar.publish(Bool(data=True))
            rclpy.shutdown()
            return

        self.reset_simulation()

    def reset_simulation_due_to_time(self):
        self.simulation_end_time = self.get_clock().now()
        self.simulation_duration = self.simulation_end_time - self.simulation_start_time
        self.save_data_to_json(final_reward=True)
        self.simulation_count += 1

        if self.simulation_count >= self.simulation_limit:
            self.get_logger().info('Simulation limit reached, shutting down...')
            self.publisher_stop_lidar.publish(Bool(data=True))
            rclpy.shutdown()
            return

        self.reset_simulation()

    def future_callback(self, future):
        try:
            future.result()
            self.get_logger().info('Simulation reset successfully')
            self.distance_to_obstacle = None
            self.simulation_start_time = self.get_clock().now()  
            self.simulation_duration = None  
        except Exception as e:
            self.get_logger().error(f'Service call failed {e!r}')

    def change_direction(self):
        self.current_direction = random.uniform(-(math.pi/2), math.pi/2)
        self.publish_velocity()

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = self.current_velocity
        msg.angular.z = self.current_direction
        self.publisher_.publish(msg)
        self.get_logger().info(f'Changed driving direction to: {self.current_direction:.2f}, moving forward')

    def clear_json_file(self):
        filename = f'driving_data_{self.robot_type.lower()}.json'
        with open(filename, 'w') as json_file:
            json.dump([], json_file)

    def direction_to_string(self, direction):
        if direction > math.pi/4:
            return 'hard right'
        elif direction > math.pi/12:
            return 'right'
        elif direction < -math.pi/4:
            return 'hard left'
        elif direction < -math.pi/12:
            return 'left'
        else:
            return 'forward'
    
    def distance_to_string(self, distance):
        if distance < 0.4:
            return "hit"
        elif distance < 0.6:
            return "very close"
        elif distance < 1.0:
            return "close"
        elif distance < 1.5:
            return "safe"
        elif distance > 2.0:
            return "far"
        else:
            return "safe"
    
    def RL_rewards(self, final_reward=False):
        reward = 0
        if self.distance_to_obstacle is not None:
            current_distance_status = self.distance_to_string(self.distance_to_obstacle)
            sym_time = self.simulation_duration.nanoseconds / 1e9 if self.simulation_duration else (self.get_clock().now() - self.simulation_start_time).nanoseconds / 1e9

            if current_distance_status != "hit":
                if final_reward:
                    reward = sym_time * 10
                if self.previous_distance_status in ["close", "very close"] and current_distance_status in ["safe", "far"]:
                    reward += 35
                elif self.previous_distance_status in ["very close"] and current_distance_status == "close":
                    reward += 25
                elif self.previous_distance_status in ["safe"] and current_distance_status == "far":
                     reward += 10
                elif self.previous_distance_status == "close" and current_distance_status == "very close":
                    reward -= 10
            else:
                reward = (-50 + sym_time * (-10))
            self.previous_distance_status = current_distance_status
        else:
            self.get_logger().info('No data available for reward calculation.')

        return reward


    def save_data_to_json(self, final_reward=False):
        if self.simulation_duration is not None:
            duration_in_seconds = self.simulation_duration.nanoseconds / 1e9
        else:
            duration_in_seconds = (self.get_clock().now() - self.simulation_start_time).nanoseconds / 1e9

        if self.distance_to_obstacle is not None:
            distance_str = self.distance_to_string(self.distance_to_obstacle)
        else:
            distance_str = "far"

        if self.current_direction is not None:
            direction_str = self.direction_to_string(self.current_direction)
        else:
            direction_str = "unknown"

        if self.previous_distance != distance_str or final_reward:
            self.previous_distance = distance_str

            reward = self.RL_rewards(final_reward=final_reward)
            reward = round(reward, 2)

            data = {
                "simulation_count": self.simulation_count,
                "distance_to_obstacle": distance_str,
                "current_direction": direction_str,
                "simulation_duration": duration_in_seconds,
                "reward": reward
            }

            filename = f'driving_data_{self.robot_type.lower()}.json'
            try:
                with open(filename, 'r') as json_file:
                    data_history = json.load(json_file)
            except FileNotFoundError:
                data_history = []

            data_history.append(data)

            with open(filename, 'w') as json_file:
                json.dump(data_history, json_file, indent=4)

def main(args=None):
    rclpy.init(args=args)
    auto_drive_node = AutoDriveNode()
    rclpy.spin(auto_drive_node)
    auto_drive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
