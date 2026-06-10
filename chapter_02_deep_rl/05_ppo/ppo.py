"""
PPO (Proximal Policy Optimization) 算法实现 (2017)
-----------------------------------------------------------
论文: Proximal Policy Optimization Algorithms
作者: John Schulman et al. (OpenAI)

核心创新:
使用裁剪目标函数限制策略更新幅度，避免破坏性更新。

关键公式:

Clipped Surrogate 目标:
  L_CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
  其中 r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)

GAE (Generalized Advantage Estimation):
  A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
  δ_t = r_t + γV(s_{t+1}) - V(s_t)

PPO为何如此成功:
- 简单: 只需几行代码
- 稳定: 裁剪机制防止过大的策略更新
- 高效: 可以多轮使用同一批数据
- 通用: 适用于离散和连续动作空间
- LLM训练: RLHF的标准算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPONetwork(nn.Module):
    """PPO的Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 continuous: bool = False):
        super(PPONetwork, self).__init__()
        self.continuous = continuous
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        if continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor_logits = nn.Linear(hidden_dim, action_dim)
        
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        value = self.critic(features)
        
        if self.continuous:
            action_mean = self.actor_mean(features)
            return action_mean, value
        else:
            action_logits = self.actor_mean(features)
            return action_logits, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        if self.continuous:
            action_mean, value = self.forward(x)
            action_std = self.actor_logstd.exp().expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, action_std)
        else:
            logits, value = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1) if not self.continuous else action_mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        if not self.continuous:
            log_prob = log_prob
        
        return action, log_prob, value, dist.entropy()


class PPO:
    """Proximal Policy Optimization"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.network = PPONetwork(state_dim, action_dim, hidden_dim, continuous).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout 缓冲区
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
        self.actions.append(action.item() if not self.network.continuous else action.cpu().numpy()[0])
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return int(action.item()) if not self.network.continuous else action.cpu().numpy()[0]
    
    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(float(done))
    
    def _compute_gae(self, last_value: float = 0.0):
        """计算GAE优势估计"""
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
        """PPO更新步骤"""
        advantages, returns = self._compute_gae()
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为Tensor
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        adv_tensor = torch.FloatTensor(advantages).to(self.device)
        ret_tensor = torch.FloatTensor(returns).to(self.device)
        
        total_losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}
        
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_adv = adv_tensor[batch_indices]
                batch_ret = ret_tensor[batch_indices]
                
                # 计算新的对数概率和价值
                logits_or_mean, values = self.network(batch_states)
                if self.network.continuous:
                    action_std = self.network.actor_logstd.exp().expand_as(logits_or_mean)
                    dist = torch.distributions.Normal(logits_or_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits_or_mean)
                    new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 重要性采样比率
                ratio = (new_log_probs - batch_log_probs).exp()
                
                # 裁剪目标
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(-1), batch_ret)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_losses['policy'].append(policy_loss.item())
                total_losses['value'].append(value_loss.item())
                total_losses['entropy'].append(entropy.item())
                total_losses['total'].append(loss.item())
        
        # 清空缓冲区
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return {k: np.mean(v) for k, v in total_losses.items()}
    
    def save(self, path: str):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
