#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_distance = self.create_subscription(Float32, '/distance_to_obstacle', self.distance_callback, 10)
        self.current_distance = None
        self.model = DQN(in_position=2, out_actions=5)
        self.model.load_state_dict(torch.load('dqn_model.pth'))
        self.model.eval()

    def distance_callback(self, msg):
        self.current_distance = msg.data
        self.make_decision()

    def make_decision(self):
        if self.current_distance is not None:
            state = torch.tensor([[self.current_distance, 0]], dtype=torch.float32)
            with torch.no_grad():
                action = self.model(state).max(1)[1].item()
            self.publish_velocity(action)

    def publish_velocity(self, action):
        msg = Twist()
        msg.linear.x = 0.5  #Stała prędkość liniowa
        if action == 0:
            msg.angular.z = -1.0  #Mocny skręt w lewo
        elif action == 1:
            msg.angular.z = -0.5  #Skręt w lewo
        elif action == 2:
            msg.angular.z = 0.0  #Jazda na wprost
        elif action == 3:
            msg.angular.z = 0.5  #Skręt w prawo
        elif action == 4:
            msg.angular.z = 1.0  #Mocny skręt w prawo
        self.publisher_.publish(msg)
        self.get_logger().info(f'Aktualny kierunek: {action}')

def main(args=None):
    rclpy.init(args=args)
    dqn_controller = DQNController()
    rclpy.spin(dqn_controller)
    dqn_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()