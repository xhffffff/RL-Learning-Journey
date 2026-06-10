"""
A3C / A2C 算法实现 (2016)
-------------------------------
论文: Asynchronous Methods for Deep Reinforcement Learning
作者: Mnih et al.

核心架构:
- Actor网络: 输出动作概率 π(a|s)
- Critic网络: 输出状态价值 V(s)
- 优势函数: A(s,a) = Q(s,a) - V(s)

A2C vs A3C:
- A2C: 同步更新，等待所有worker完成
- A3C: 异步更新，每个worker独立更新全局网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    """Actor-Critic 共享网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value


class A2C:
    """
    Advantage Actor-Critic (A2C)
    
    使用n-step优势估计:
    A(s_t, a_t) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t)
    
    两个损失:
    - Actor Loss:  -log π(a|s) * A  (策略梯度)
    - Critic Loss: MSE(V(s), R)     (价值估计)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
        
        if eval_mode:
            return int(probs.argmax().item())
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        self.states.append(state)
        self.actions.append(action.item())
        self.values.append(value.item())
        self.log_probs.append(dist.log_prob(action).item())
        
        return int(action.item())
    
    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(float(done))
    
    def update(self) -> tuple:
        """在完整episode后更新"""
        # 计算n-step回报
        returns = []
        R = 0
        
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R * (1 - self.dones[i])
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # 优势估计
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor损失
        actor_loss = (-log_probs * advantages.detach()).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(values, returns)
        
        # 熵损失 (鼓励探索)
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        logits, _ = self.network(states_tensor)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        # 清空缓冲区
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
