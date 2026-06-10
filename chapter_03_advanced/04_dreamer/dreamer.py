"""
Dreamer 世界模型核心组件实现
-----------------------------------------------------------
论文: Mastering Diverse Domains through World Models (DreamerV3, 2023)
作者: Hafner et al. (DeepMind & TU/e)

Dreamer核心架构:
1. Encoder: 观察 → 隐变量
2. RSSM: 循环状态空间模型 (世界模型)
3. Decoder: 隐变量 → 观察
4. Reward Predictor: 隐变量 → 奖励
5. Actor/Critic: 在隐空间中学习策略

Dreamer三大版本:
- V1 (2020): 基础世界模型 + Actor-Critic
- V2 (2021): 离散隐变量 + KL平衡
- V3 (2023): 无需超参数调节，通用性强

本实现聚焦于RSSM核心组件的教学
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RSSM(nn.Module):
    """
    循环状态空间模型 (Recurrent State-Space Model)
    Dreamer的核心组件
    
    状态分解:
    - 确定性状态 h_t: 通过RNN建模
    - 随机状态 z_t:  通过离散/连续分布建模
    
    转移模型:
    p(z_t | h_t)
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    
    表示模型:
    q(z_t | h_t, o_t)
    """
    
    def __init__(
        self,
        action_dim: int,
        deterministic_dim: int = 200,
        stochastic_dim: int = 32,
        class_dim: int = 32,
        hidden_dim: int = 200,
        device: str = "cpu"
    ):
        super(RSSM, self).__init__()
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.class_dim = class_dim
        self.device = device
        
        self.rnn = nn.GRUCell(stochastic_dim * class_dim + action_dim, deterministic_dim)
        
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * class_dim)
        )
        
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + 1024, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * class_dim)
        )
    
    def prior(self, h: torch.Tensor):
        """先验分布 p(z|h)"""
        logits = self.prior_net(h)
        logits = logits.view(-1, self.stochastic_dim, self.class_dim)
        return torch.distributions.Categorical(logits=logits)
    
    def posterior(self, h: torch.Tensor, obs_embed: torch.Tensor):
        """后验分布 q(z|h,o)"""
        inputs = torch.cat([h, obs_embed], dim=-1)
        logits = self.posterior_net(inputs)
        logits = logits.view(-1, self.stochastic_dim, self.class_dim)
        return torch.distributions.Categorical(logits=logits)
    
    def forward(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """一步前向: (h_t, z_t, a_t) → (h_{t+1}, z_{t+1})"""
        if z.dim() == 2:
            z = F.one_hot(z.long(), self.class_dim).float().view(-1, self.stochastic_dim * self.class_dim)
        else:
            z_flat = z.view(-1, self.stochastic_dim * self.class_dim)
            z = z_flat
        rnn_input = torch.cat([z, action], dim=-1)
        h_new = self.rnn(rnn_input, h)
        
        prior_dist = self.prior(h_new)
        z_new = prior_dist.sample()
        
        return h_new, z_new


class DreamerEncoder(nn.Module):
    """
    编码器: 观察 → 嵌入
    将原始观察压缩为低维特征
    """
    
    def __init__(self, obs_dim: int, embed_dim: int = 1024, hidden_dim: int = 400):
        super(DreamerEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, obs: torch.Tensor):
        return self.network(obs)


class DreamerDecoder(nn.Module):
    """
    解码器: 隐变量 → 观察
    从隐变量重构原始观察
    """
    
    def __init__(self, stochastic_dim: int, class_dim: int, obs_dim: int, hidden_dim: int = 400):
        super(DreamerDecoder, self).__init__()
        self.class_dim = class_dim
        self.network = nn.Sequential(
            nn.Linear(stochastic_dim * class_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, z: torch.Tensor):
        batch_size = z.shape[0]
        if z.dim() == 2:
            z = F.one_hot(z.long(), self.class_dim).float().view(batch_size, -1)
        z_flat = z.view(batch_size, -1)
        return self.network(z_flat)


class RewardPredictor(nn.Module):
    """奖励预测器: 隐变量 → 奖励"""
    
    def __init__(self, stochastic_dim: int, class_dim: int, hidden_dim: int = 400):
        super(RewardPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(stochastic_dim * class_dim + 200, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor, h: torch.Tensor):
        if z.dim() == 2:
            z = F.one_hot(z.long(), 32).float().view(z.shape[0], -1)
        z_flat = z.view(z.shape[0], -1)
        return self.network(torch.cat([z_flat, h], dim=-1))


class DreamerWorldModel(nn.Module):
    """
    Dreamer世界模型
    整合RSSM + Encoder + Decoder + Reward
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 1024,
        deterministic_dim: int = 200,
        stochastic_dim: int = 32,
        class_dim: int = 32
    ):
        super(DreamerWorldModel, self).__init__()
        
        self.encoder = DreamerEncoder(obs_dim, embed_dim)
        self.rssm = RSSM(action_dim, deterministic_dim, stochastic_dim, class_dim)
        self.decoder = DreamerDecoder(stochastic_dim, class_dim, obs_dim)
        self.reward_predictor = RewardPredictor(stochastic_dim, class_dim)
    
    def imagine(
        self, h: torch.Tensor, z: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        世界模型中想象 (Rollout)
        """
        horizons = []
        zs = []
        rewards = []

        batch_size = h.shape[0]
        for a in actions:
            h, z = self.rssm(a, h, z)
            reward = self.reward_predictor(z, h)

            horizons.append(h)
            zs.append(z)
            rewards.append(reward)

        return torch.stack(horizons), torch.stack(zs), torch.stack(rewards)


if __name__ == "__main__":
    print("Dreamer World Model Components")
    print("=" * 40)
    
    obs_dim = 64
    action_dim = 4
    batch_size = 2
    
    model = DreamerWorldModel(obs_dim, action_dim)
    
    obs = torch.randn(batch_size, obs_dim)
    embed = model.encoder(obs)
    print(f"观察 → 嵌入: {obs.shape} → {embed.shape}")
    
    h = torch.zeros(batch_size, 200)
    prior = model.rssm.prior(h)
    z = prior.sample()
    print(f"先验分布: {prior.batch_shape}, z形状: {z.shape}")
    
    obs_recon = model.decoder(z)
    print(f"重构观察: {obs_recon.shape}")
    
    # 想象10步
    actions = torch.randn(10, batch_size, action_dim)
    h_im, z_im, r_im = model.imagine(h, z, actions)
    print(f"\n想象展开 (10步):")
    print(f"  隐藏状态: {h_im.shape}")
    print(f"  随机状态: {z_im.shape}")
    print(f"  预测奖励: {r_im.shape}")
