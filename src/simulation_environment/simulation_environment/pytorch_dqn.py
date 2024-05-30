#!/usr/bin/env python3
# import json
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import namedtuple, deque

# #Parametry
# BATCH_SIZE = 64
# GAMMA = 0.99
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 200
# MEMORY_SIZE = 10000

# with open('driving_data.json', 'r') as f:
#     driving_data = f.read()

# states = []
# actions = []
# rewards = []
# next_states = []

# for entry in driving_data:
#     state = [entry['distance_to_obstacle'], entry['current_direction']]


