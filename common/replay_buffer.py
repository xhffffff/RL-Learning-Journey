"""经验回放缓冲区"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1, discrete: bool = True):
        self.capacity = capacity
        self.discrete = discrete
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = np.atleast_1d(np.asarray(action, dtype=np.float32))
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        actions = torch.FloatTensor(self.actions[indices])
        if self.discrete:
            actions = actions.long().squeeze(-1)
        return (
            torch.FloatTensor(self.states[indices]),
            actions,
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )

    def __len__(self):
        return self.size
