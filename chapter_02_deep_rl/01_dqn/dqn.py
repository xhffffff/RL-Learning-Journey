"""
DQN (Deep Q-Network) 算法实现
-------------------------------
Deep Q-Network (2015) - DeepMind, Nature
Human-level control through deep reinforcement learning

核心创新:
1. 经验回放: 打破数据相关性
2. 目标网络: 稳定训练
3. 深度网络: 处理高维输入

算法流程:
1. 智能体与环境交互，存储 (s,a,r,s',done)
2. 从回放缓冲区采样批次
3. 计算TD目标: y = r + γ max_a' Q_target(s', a')
4. 最小化 MSE Loss: (y - Q_online(s,a))²
5. 定期同步目标网络参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """DQN的Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """
    Deep Q-Network
    
    参数:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        learning_rate: 学习率
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最小探索率
        epsilon_decay: 探索率衰减
        target_update: 目标网络更新频率
        buffer_capacity: 回放缓冲区容量
        batch_size: 训练批次大小
        device: 计算设备
    """
    
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
        
        # Epsilon 退火
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 在线网络和目标网络
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity, state_dim)
        
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-Greedy 动作选择"""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax().item())
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """执行一次梯度更新"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 当前Q值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = F.mse_loss(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1.0 * self.update_count / self.epsilon_decay)
        )
        
        return loss.item()
    
    def save(self, filepath: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
