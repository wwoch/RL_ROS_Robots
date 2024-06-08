#!/usr/bin/env python3
import json
import os
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
LR = 0.01
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
    "hard left": 0,
    "left": 1,
    "forward": 2,
    "right": 3,
    "hard right": 4
}

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
        self.robot_type = self.declare_parameter('robot_type', 'A').get_parameter_value().string_value
        self.train_dqn()

    def train_dqn(self):
        self.get_logger().info(f'TRAIN DQN STARTING...')

        filenames = ['driving_data_a.json', 'driving_data_b.json']
        driving_data = []
        files_found = 0

        for filename in filenames:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    driving_data.extend(data)
                    files_found += 1

        if driving_data:
            if files_found == 2:
                output_filename = 'driving_data_c.json'
            else:
                output_filename = f'driving_data_{self.robot_type.lower()}.json'
            
            with open(output_filename, 'w') as f:
                json.dump(driving_data, f)

        else:
            self.get_logger().warning(f'No driving data files found.')

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in range(len(driving_data) - 1):
            state = [distance_mapping[driving_data[i]['distance_to_obstacle']], direction_mapping[driving_data[i]['current_direction']]]
            next_state = [distance_mapping[driving_data[i + 1]['distance_to_obstacle']], direction_mapping[driving_data[i + 1]['current_direction']]]
            action = torch.tensor([[direction_mapping[driving_data[i]['current_direction']]]], dtype=torch.long)
            reward = torch.tensor([driving_data[i]['reward']], dtype=torch.float32)
            states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))

        num_episodes = 200
        for i_episode in range(num_episodes):
            for t in range(len(states)):
                state = states[t].to(device)
                action = actions[t].to(device)
                reward = rewards[t].to(device)
                next_state = next_states[t].to(device)

                memory.push(state, action, reward, next_state)

                optimize_model()
            
            if i_episode % 5 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                self.get_logger().info(f'Episode {i_episode}, Updated target network')

        model_a_exists = os.path.exists('dqn_model_a.pth')
        model_b_exists = os.path.exists('dqn_model_b.pth')

        if model_a_exists and model_b_exists:
            model_filename = 'dqn_model_c.pth'
        else:
            model_filename = f'dqn_model_{self.robot_type.lower()}.pth'

        torch.save(policy_net.state_dict(), model_filename)
        self.get_logger().info(f'TRAIN DQN FINISH!')

def main(args=None):
    rclpy.init(args=args)
    dqn_trainer = DQNTrain()
    rclpy.spin(dqn_trainer)
    dqn_trainer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
