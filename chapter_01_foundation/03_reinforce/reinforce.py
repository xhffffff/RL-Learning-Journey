"""
REINFORCE 算法实现 (1992)
-------------------------------
论文: Simple Statistical Gradient-Following Algorithms for Connectionist
      Reinforcement Learning
作者: Ronald J. Williams

核心思想:
- 蒙特卡洛策略梯度方法
- 直接参数化策略 π_θ(a|s)
- 通过梯度上升优化期望回报

策略梯度定理:
    ∇_θ J = E[∇_θ log π_θ(a|s) · G_t]

其中 G_t = Σ_{k=0}^{T-t} γ^k r_{t+k+1} 是从时间t开始的折扣回报
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    """
    策略网络
    输入状态，输出动作概率分布
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class REINFORCE:
    """
    REINFORCE (蒙特卡洛策略梯度)
    
    算法流程:
    1. 使用当前策略收集完整的 episode
    2. 对于 episode 中的每一步:
       - 计算折扣回报 G_t
       - 更新: θ ← θ + α ∇_θ log π_θ(a_t|s_t) · G_t
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.01,
        gamma: float = 0.99
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # 存储 episode 数据
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        根据策略采样动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        
        if eval_mode:
            return int(torch.argmax(probs).item())
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        if not eval_mode:
            self.states.append(state_tensor)
            self.actions.append(action)
        
        return int(action.item())
    
    def store_reward(self, reward: float):
        """存储即时奖励"""
        self.rewards.append(reward)
    
    def update(self) -> float:
        """
        使用完整 episode 数据更新策略
        
        Returns:
            total_loss: 平均损失
        """
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # 标准化回报（减少方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度
        policy_loss = []
        for state, action, G_val in zip(self.states, self.actions, returns):
            probs = self.policy_network(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            policy_loss.append(-log_prob * G_val)
        
        loss = torch.stack(policy_loss).mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空 buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        
        return loss.item()


class REINFORCEWithBaseline:
    """
    带基线的 REINFORCE
    
    使用价值网络 V(s) 作为基线来减少方差:
    ∇_θ J = E[∇_θ log π_θ(a|s) · (G_t - V(s_t))]
    
    (G_t - V(s_t)) 称为优势估计
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.01,
        gamma: float = 0.99
    ):
        self.gamma = gamma
        
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )
        
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        
        if eval_mode:
            return int(torch.argmax(probs).item()), 0.0
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        if not eval_mode:
            self.states.append(state_tensor)
            self.actions.append(action)
        
        return int(action.item()), dist.entropy().item()
    
    def store_reward(self, reward: float):
        self.rewards.append(reward)
    
    def update(self) -> Tuple[float, float]:
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # 计算价值估计
        states_tensor = torch.cat(self.states)
        values = self.value_network(states_tensor)
        
        # 优势估计
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略损失
        policy_loss = []
        for state, action, adv in zip(self.states, self.actions, advantages):
            probs = self.policy_network(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            policy_loss.append(-log_prob * adv)
        
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = F.mse_loss(values, returns)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        
        return policy_loss.item(), value_loss.item()
