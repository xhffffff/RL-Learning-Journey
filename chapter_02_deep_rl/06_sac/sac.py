"""
SAC (Soft Actor-Critic) 算法实现 (2018)
-----------------------------------------------------------
论文: Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
作者: Haarnoja et al. (UC Berkeley & Google)

核心创新: 最大熵强化学习
目标函数中加入策略的熵项，鼓励探索

J(π) = Σ_t E[r_t + α H(π(·|s_t))]

三大关键组件:
1. 软Q函数:  两个Q网络 + 目标网络
2. 软V函数:  用Q函数最小化近似(实践中通常省略)
3. 软策略:   参数化高斯分布

自动温度调节:
α 的损失: L(α) = E[-α(log π(a|s) + 目标熵)]

SAC的优势:
- 样本效率: Off-policy + 最大熵 → 高效探索
- 稳定性: 双Q网络减少过估计
- 通用性: 适用于连续和离散动作空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.replay_buffer import ReplayBuffer


class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample(mean.shape).to(state.device)
        action = mean + std * z
        action = torch.tanh(action)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean


class QNetwork(nn.Module):
    """双Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC:
    """Soft Actor-Critic"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = None,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Critic网络 (双Q)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Actor网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        
        # 温度参数
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = alpha
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.memory = ReplayBuffer(buffer_capacity, state_dim, action_dim, discrete=False)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if eval_mode:
            with torch.no_grad():
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean)
            return action.cpu().numpy()[0]
        
        with torch.no_grad():
            action, _, _ = self.policy.sample(state_tensor)
        return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self) -> dict:
        if len(self.memory) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # ---- Critic更新 ----
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        q1_loss = F.mse_loss(self.q1(states, actions), target_q)
        q2_loss = F.mse_loss(self.q2(states, actions), target_q)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ---- Actor更新 ----
        sampled_actions, log_probs, _ = self.policy.sample(states)
        q1_new = self.q1(states, sampled_actions)
        q2_new = self.q2(states, sampled_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ---- Alpha更新 ----
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # ---- 软更新目标网络 ----
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item()
        }
