"""
TD3 (Twin Delayed DDPG) 算法实现 (2018)
-----------------------------------------------------------
论文: Addressing Function Approximation Error in Actor-Critic Methods
作者: Fujimoto et al.

三大核心改进 (相对于DDPG):

1. Clipped Double Q-Learning
   使用两个Q网络中较小的值来减少过估计
   y = r + γ min(Q1', Q2')

2. Delayed Policy Updates
   策略更新频率低于Q网络 (通常每次Q更新2次才更新1次策略)
   减少Actor从错误的Q值中学习

3. Target Policy Smoothing
   给目标动作加噪声，使Q函数更平滑
   a' = clip(π_target(s') + noise, a_low, a_high)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """确定性策略网络 μ(s)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Q网络 Q(s,a)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1架构
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2架构 (双Q网络)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        return self.fc3(q1)


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient
    
    参数:
        state_dim: 状态维度
        action_dim: 动作维度
        max_action: 最大动作值
        lr: 学习率
        gamma: 折扣因子
        tau: 软更新系数
        policy_noise: 目标策略平滑噪声
        noise_clip: 噪声裁剪范围
        policy_freq: 策略更新频率
        device: 计算设备
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.device = device
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(buffer_capacity, state_dim, action_dim, discrete=False)
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        if not eval_mode:
            noise = np.random.normal(0, self.max_action * 0.1, size=action.shape[1])
            action = (action.cpu().numpy()[0] + noise).clip(-self.max_action, self.max_action)
            return action
        else:
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """TD3更新"""
        self.total_it += 1
        
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # ---- TD3改进1: 目标策略平滑 ----
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # ---- TD3改进2: Clipped Double Q-Learning ----
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # Critic更新
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---- TD3改进3: 延迟策略更新 ----
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
            for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        
        return {'critic_loss': critic_loss.item()}
