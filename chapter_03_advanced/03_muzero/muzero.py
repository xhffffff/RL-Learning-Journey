"""
MuZero 核心组件实现
-----------------------------------------------------------
论文: Mastering Atari, Go, chess and shogi by planning
      with a learned model (DeepMind, 2019)

完整MuZero包含:
1. 表示函数 (Representation)
2. 动态函数 (Dynamics)
3. 预测函数 (Prediction)
4. MCTS (Monte Carlo Tree Search)

由于完整MuZero实现较复杂，本文件聚焦于核心网络组件的教学实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class MuZeroRepresentation(nn.Module):
    """
    表示函数 h: o_t → s^0
    
    将原始观察映射到隐藏状态
    原始论文使用ResNet，这里用简化的MLP+CNN
    """
    
    def __init__(self, obs_shape, hidden_dim: int = 256):
        super(MuZeroRepresentation, self).__init__()
        self.hidden_dim = hidden_dim
        
        if len(obs_shape) == 1:
            # 向量观测
            input_dim = obs_shape[0]
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                conv_out = conv_net(dummy)
                conv_dim = conv_out.view(1, -1).size(1)
            self.network = nn.Sequential(
                conv_net,
                nn.Flatten(),
                nn.Linear(conv_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class MuZeroDynamics(nn.Module):
    """
    动态函数 g: (s^{k-1}, a^k) → (r^k, s^k)
    
    预测:
    - 即时奖励 r
    - 下一个隐藏状态 s'
    """
    
    def __init__(self, hidden_dim: int, action_dim: int, reward_support_size: int = 601):
        super(MuZeroDynamics, self).__init__()
        self.hidden_dim = hidden_dim
        self.reward_support_size = reward_support_size
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.reward_head = nn.Linear(hidden_dim, reward_support_size)
        self.state_head = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden_state, action], dim=-1)
        features = self.network(x)
        
        reward_logits = self.reward_head(features)
        next_hidden = self.state_head(features)
        
        return reward_logits, next_hidden


class MuZeroPrediction(nn.Module):
    """
    预测函数 f: s^k → (p^k, v^k)
    
    预测:
    - 策略分布 p (用于MCTS)
    - 价值 v (预期回报)
    """
    
    def __init__(self, hidden_dim: int, action_dim: int, value_support_size: int = 601):
        super(MuZeroPrediction, self).__init__()
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, value_support_size)
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network(hidden_state)
        
        policy_logits = self.policy_head(features)
        value_logits = self.value_head(features)
        
        return policy_logits, value_logits


class MuZeroNetwork(nn.Module):
    """MuZero完整网络"""
    
    def __init__(self, obs_shape, action_dim: int, hidden_dim: int = 256):
        super(MuZeroNetwork, self).__init__()
        
        self.representation = MuZeroRepresentation(obs_shape, hidden_dim)
        self.dynamics = MuZeroDynamics(hidden_dim, action_dim)
        self.prediction = MuZeroPrediction(hidden_dim, action_dim)
    
    def initial_inference(self, obs: torch.Tensor):
        """初始推理"""
        hidden = self.representation(obs)
        policy, value = self.prediction(hidden)
        return hidden, policy, value
    
    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor):
        """递归推理（展开一步）"""
        reward, next_hidden = self.dynamics(hidden, action)
        policy, value = self.prediction(next_hidden)
        return reward, next_hidden, policy, value


class MCTSNode:
    """
    MCTS节点 (简化版)
    完整实现包含PUCT公式和回溯更新
    """
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def select_child(self, c1: float = 1.25, c2: float = 19652.0):
        """PUCT选择公式"""
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            ucb_score = child.value() + \
                c1 * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count) * \
                (c2 + math.log((self.visit_count + c2 + 1) / c2))
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child


if __name__ == "__main__":
    print("MuZero核心网络组件")
    print("=" * 40)
    
    # 测试网络
    obs_shape = (3, 96, 96)
    action_dim = 4
    
    model = MuZeroNetwork(obs_shape, action_dim)
    
    dummy_obs = torch.randn(1, *obs_shape)
    hidden, policy, value = model.initial_inference(dummy_obs)
    
    print(f"隐藏状态形状: {hidden.shape}")
    print(f"策略形状: {policy.shape}")
    print(f"价值形状: {value.shape}")
    
    # 测试展开
    action = torch.eye(action_dim)[0:1]
    reward, next_hidden, next_policy, next_value = model.recurrent_inference(hidden, action)
    
    print(f"\n展开后:")
    print(f"奖励形状: {reward.shape}")
    print(f"下一个隐藏状态形状: {next_hidden.shape}")
