#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math

class DQN(nn.Module):
    def __init__(self, in_position, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_position, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNController(Node):
    def __init__(self):
        super().__init__('dqn_controller')
        self.robot_type = self.declare_parameter('robot_type', 'C').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_distance = self.create_subscription(Float32, '/distance_to_obstacle', self.distance_callback, 10)
        self.current_distance = None
        self.model = DQN(in_position=2, out_actions=5)
        
        model_path = f'dqn_model_{self.robot_type.lower()}.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.get_logger().error(f'Model file {model_path} does not exist.')
            rclpy.shutdown()
            return
        
        self.start_time = time.time()
        self.collision_detected = False

    def distance_callback(self, msg):
        if self.collision_detected:
            return
        self.current_distance = msg.data
        if self.current_distance < 0.31:
            self.collision_detected = True
            self.save_simulation_duration()
            rclpy.shutdown()
        else:
            self.make_decision()

    def make_decision(self):
        if self.current_distance is not None:
            state = torch.tensor([[self.current_distance, 0]], dtype=torch.float32)
            with torch.no_grad():
                action_values = self.model(state)
                action = action_values.max(1)[1].item()
                self.get_logger().info(f'Action values: {action_values}, Selected action: {action}')
            self.publish_velocity(action)

    def publish_velocity(self, action):
        msg = Twist()
        msg.linear.x = 0.5
        if action == 0:
            msg.angular.z = -math.pi/12
        elif action == 1:
            msg.angular.z = -math.pi/4
        elif action == 2:
            msg.angular.z = 0.0
        elif action == 3:
            msg.angular.z = math.pi/12
        elif action == 4:
            msg.angular.z = math.pi/4
        self.publisher_.publish(msg)
        self.get_logger().info(f'Aktualny kierunek: {action}')

    def save_simulation_duration(self):
        end_time = time.time()
        duration = end_time - self.start_time
        with open(f'simulation_duration_{self.robot_type.lower()}.txt', 'w') as f:
            f.write(f'Simulation Duration: {duration} seconds\n')
        self.get_logger().info(f'Simulation Duration saved to file: {duration} seconds')

def main(args=None):
    rclpy.init(args=args)
    dqn_controller = DQNController()
    try:
        rclpy.spin(dqn_controller)
    except KeyboardInterrupt:
        pass
    finally:
        if not dqn_controller.collision_detected:
            dqn_controller.save_simulation_duration()
        dqn_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
