"""
Dueling DQN 算法实现 (2016)
-------------------------------
论文: Dueling Network Architectures for Deep Reinforcement Learning
作者: Wang et al.

核心创新: 将Q网络拆分为两个独立流
- Value Stream (V):  估计状态的固有价值 V(s)
- Advantage Stream (A): 估计每个动作的相对优势 A(s,a)

组合公式: Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

为什么有效:
- V(s) 学习快: 很多状态共享相似的价值
- 不需要为每个动作都学习完整的Q值
- 对于动作价值接近的状态尤为有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.replay_buffer import ReplayBuffer


class DuelingQNetwork(nn.Module):
    """Dueling网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势函数流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + [A(s,a) - mean(A)]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQN:
    """Dueling DQN: 分解V和A"""
    
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
        
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity, state_dim)
        self.update_count = 0
    
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
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = F.mse_loss(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1.0 * self.update_count / self.epsilon_decay)
        )
        
        return loss.item()
