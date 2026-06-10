"""
TRPO (Trust Region Policy Optimization) 算法实现
-----------------------------------------------------------
论文: Trust Region Policy Optimization (Schulman et al., 2015)

核心思想:
- 使用KL散度约束限制策略更新幅度
- 保证策略单调提升 (Monotonic Improvement Guarantee)
- 自然策略梯度 (Natural Policy Gradient)

注意：本实现采用简化版TRPO，使用KL散度惩罚系数来近似信任区域约束，
而非原论文中的共轭梯度 + 线搜索方法。
完整的TRPO实现需使用二阶优化（共轭梯度求解自然梯度方向 + 回溯线搜索）。
本简化版在概念上更接近PPO的KL惩罚变体。

TRPO vs PPO:
- TRPO: 使用二阶优化 + KL约束
- PPO:  使用一阶优化 + 裁剪目标 (简化版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TRPONetwork(nn.Module):
    """TRPO的Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(TRPONetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        logits = self.actor(state)
        value = self.critic(state)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy


class TRPO:
    """
    Trust Region Policy Optimization
    
    注意: 完整TRPO使用共轭梯度+线搜索，这里采用简化版
    使用KL惩罚系数来近似信任区域约束
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_kl: float = 0.01,
        damping: float = 0.1,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.device = device
        
        self.network = TRPONetwork(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=critic_lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action(state_tensor, deterministic)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return int(action.item())
    
    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(float(done))
    
    def _compute_gae(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [0])
        rewards = np.array(self.rewards)
        
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self) -> dict:
        advantages, returns = self._compute_gae()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        
        # ---- Critic更新 ----
        values = self.network.critic(states_t).squeeze(-1)
        critic_loss = F.mse_loss(values, ret_t)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---- Actor更新 (TRPO风格) ----
        logits = self.network.actor(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_t)
        
        # 重要性采样比率
        ratio = (new_log_probs - old_log_probs_t).exp()
        
        # 策略损失 (不裁剪的纯粹策略梯度)
        surr_loss = -(ratio * adv_t).mean()
        
        # KL惩罚项 (TRPO简化为KL惩罚)
        with torch.no_grad():
            old_dist = torch.distributions.Categorical(
                logits=self.network.actor(states_t).detach())
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        
        # KL惩罚损失
        actor_loss = surr_loss + self.damping * kl
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return {
            'actor_loss': surr_loss.item(),
            'critic_loss': critic_loss.item(),
            'kl': kl.item()
        }
