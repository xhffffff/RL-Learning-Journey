"""
Double DQN 算法实现 (2015)
-------------------------------
论文: Deep Reinforcement Learning with Double Q-learning
作者: van Hasselt, Guez & Silver

核心思想:
解决DQN中Q值过高估计的问题

标准DQN:    y = r + γ·max_a' Q_target(s', a')
Double DQN: y = r + γ·Q_target(s', argmax_a' Q_online(s', a'))

解耦:
- 动作选择: 使用在线网络 argmax Q_online(s', a')
- Q值评估:  使用目标网络 Q_target(s', 选择的动作)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DoubleDQN:
    """Double DQN: 消除Q值过高估计"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 500,
        target_update: int = 10,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self._sync_target()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity, state_dim)
        self.update_count = 0
    
    def _sync_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax().item())
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN关键差异: 用在线网络选动作，目标网络评价值
            online_next_q = self.q_network(next_states)
            best_actions = online_next_q.argmax(dim=1, keepdim=True)
            
            target_next_q = self.target_network(next_states)
            max_target_q = target_next_q.gather(1, best_actions).squeeze(1)
            
            target_q = rewards + self.gamma * max_target_q * (1 - dones)
        
        loss = F.mse_loss(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self._sync_target()
        
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1.0 * self.update_count / self.epsilon_decay)
        )
        
        return loss.item()
