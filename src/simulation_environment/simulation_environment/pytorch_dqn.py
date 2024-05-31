#!/usr/bin/env python3
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("START")

#parametry
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 200
LR = 0.001
MEMORY_SIZE = 10000

# Mapowanie stringów na liczby
distance_mapping = {
    "hit": 0,
    "very close": 1,
    "close": 2,
    "safe": 3,
    "far": 4
}

direction_mapping = {
    "hard left": -2,
    "left": -1,
    "forward": 0,
    "right": 1,
    "hard right": 2
}

with open('driving_data.json', 'r') as f:
    driving_data = json.load(f)

states = []
actions = []
rewards = []
next_states = []

for entry in driving_data:
    state = [distance_mapping[entry['distance_to_obstacle']], direction_mapping[entry['current_direction']]]
    action = direction_mapping[entry['current_direction']]
    reward = entry['reward']
    next_state = state
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    next_states.append(next_state)

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

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        #zapis doświadczenia do bufora
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        #losow wybiera batch_size doświadczeń z bufora
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

n_actions = 5 #mocno w lewo lub prawo, skręt w lewo lub prawo, prosto
robot_state = 2 #odleglosc od przeszkody i kierunek

policy_net = DQN(robot_state, n_actions).to(device)
target_net = DQN(robot_state, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
        math.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long, device=device)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool, device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

class DQNTrain(Node):
    def __init__(self):
        super().__init__('dqn_train')
        self.subscription = self.create_subscription(
            Bool,
            '/stop_lidar',
            self.listener_callback,
            10)
        self.subscription
        self.training_started = False
    
    def listener_callback(self,msg):
        if msg.data and not self.training_started:
            self.training_started = True
            self.train_dqn()
    
    def train_dqn(self):
        self.get_logger().info(f'TRAIN DQN STARTING...')

        num_episodes = 50
        for i_episode in range(num_episodes):
            for t in range(len(states)):
                state = torch.tensor([states[t]], dtype=torch.float32).to(device)
                action = torch.tensor([[actions[t]]], dtype=torch.long).to(device)
                reward = torch.tensor([rewards[t]], dtype=torch.float32).to(device)
                next_state = torch.tensor([next_states[t]], dtype=torch.float32).to(device)

                memory.push(state, action, reward, next_state)

                optimize_model()
            
            if i_episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())
        self.get_logger().info(f'TRAIN DQN FINISH!')

torch.save(policy_net.state_dict(), 'dqn_model.pth')

def main(args=None):
    rclpy.init(args=args)
    dqn_trainer = DQNTrain()
    rclpy.spin(dqn_trainer)
    dqn_trainer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()