"""
RLOO (REINFORCE Leave-One-Out) 核心实现
-----------------------------------------------------------------
论文: Back to Basics: Revisiting REINFORCE Style Optimization for
      Learning from Human Feedback in LLMs
作者: Ahmadian et al. (Cohere For AI, Feb 2024, arXiv:2402.14740)

RLOO是一种极简但高效的LLM对齐算法，回归到最基础的REINFORCE方法。

关键创新 vs PPO:

标准PPO:
  需要4个模型: Policy, Reference, Critic, Reward Model
  使用GAE + Critic网络估计优势
  PPO裁剪机制 + 多轮更新

RLOO:
  只需2个模型: Policy, Reference
  使用Leave-One-Out基线估计优势 (无需Critic!)
  无需PPO裁剪 — 纯REINFORCE
  **节省约33%内存** (去掉了Critic网络)

核心公式:
  对每个prompt s，采样K个独立响应a_1,...,a_K
  每个获得奖励r(s, a_k)
  
  Leave-One-Out基线: b_{-k} = (1/(K-1)) * Σ_{j≠k} r(s, a_j)
  
  优势(等价形式):
  A(s, a_k) = (K/(K-1)) * (r(s, a_k) - (1/K) * Σ_{j=1}^{K} r(s, a_j))

RLOO损失:
  L_RLOO = -(1/K) * Σ_{k=1}^{K} [A(s, a_k) * log π_θ(a_k|s)]
           + β * KL(π_θ || π_ref)

关键特性:
  - 无偏梯度估计 (Critic基线是有偏的)
  - 极简实现: 核心代码~20行
  - K值通常取2-4 (本实现K=4)

奖励设计:
  准确率奖励: 答案是否正确
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class RLOOPolicy(nn.Module):
    """RLOO策略模型 (简化版)"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 50):
        super(RLOOPolicy, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.shared(x))


class RLOOTrainer:
    """
    REINFORCE Leave-One-Out (RLOO)
    
    核心特点:
    1. 无需Critic网络 (用Leave-One-Out基线替代)
    2. 无偏梯度估计 (LOO基线是无偏的)
    3. 极简实现 (纯REINFORCE，无需裁剪)
    4. 轻量级 (比PPO少约33%参数)
    
    参数:
        policy: 训练中的策略模型
        ref_policy: 参考模型 (冻结)
        rloo_k: 每个prompt采样响应数 K
        beta: KL惩罚系数
        lr: 学习率
    """
    
    def __init__(
        self,
        policy: RLOOPolicy,
        ref_policy: RLOOPolicy,
        rloo_k: int = 4,
        beta: float = 0.04,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.rloo_k = rloo_k
        self.beta = beta
        self.device = device
        
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.stats = {'loss': [], 'reward': [], 'kl': []}
    
    def leave_one_out_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        RLOO的核心: Leave-One-Out优势估计
        
        对每个prompt，采样K个响应，计算:
        A_k = (K/(K-1)) * (r_k - mean({r_1,...,r_K}))
        
        等价于 r_k - b_{-k}，其中b_{-k}是除k外所有响应的平均奖励
        
        LOO基线的优势:
        - 无偏: E[b_{-k}] = 真实的基线期望
        - 减少方差: K个样本互相作为基线
        - 无需Critic: 完全基于采样的奖励
        """
        K = rewards.shape[-1]
        mean_r = rewards.mean(dim=-1, keepdim=True)
        return (K / (K - 1)) * (rewards - mean_r)
    
    def compute_reward(
        self, responses: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        规则奖励函数
        
        简化为分类准确率:
        响应与目标匹配 → 奖励=1
        响应与目标不匹配 → 奖励=0
        """
        accuracy = (responses == target.unsqueeze(1)).float()
        return accuracy
    
    def compute_kl(self, prompt: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """KL散度惩罚"""
        with torch.no_grad():
            ref_logits = self.ref_policy(prompt)
        
        p = F.softmax(logits, dim=-1)
        ref_p = F.softmax(ref_logits, dim=-1)
        
        kl = (p * (torch.log(p + 1e-8) - torch.log(ref_p + 1e-8))).sum(dim=-1)
        return kl
    
    def train_step(
        self, prompts: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """
        单步RLOO训练
        
        RLOO算法流程:
        1. 对每个prompt采样K个响应
        2. 计算规则奖励
        3. 计算Leave-One-Out优势
        4. 纯REINFORCE损失 (无裁剪!) + KL惩罚
        """
        batch_size = prompts.shape[0]
        K = self.rloo_k
        
        # 1. 对每个prompt采样K个响应（采样时存储old_log_probs）
        self.policy.eval()
        all_responses = []
        all_old_log_probs = []
        all_rewards = []
        
        with torch.no_grad():
            logits = self.policy(prompts)
            for _ in range(K):
                dist = torch.distributions.Categorical(logits=logits)
                response = dist.sample()
                log_prob = dist.log_prob(response)
                reward = (response == targets).float()
                
                all_responses.append(response)
                all_old_log_probs.append(log_prob)
                all_rewards.append(reward)
        
        # 2. 堆叠为 (batch, K)
        responses = torch.stack(all_responses, dim=1)          # (batch, K)
        old_log_probs = torch.stack(all_old_log_probs, dim=1)  # (batch, K)
        rewards = torch.stack(all_rewards, dim=1)              # (batch, K)
        
        # 3. 计算Leave-One-Out优势（在K维度上）
        advantages = self.leave_one_out_advantage(rewards)     # (batch, K)
        
        # 4. 优化阶段：纯REINFORCE（无裁剪!）
        self.policy.train()
        self.optimizer.zero_grad()
        
        logits = self.policy(prompts)  # (batch, vocab)
        dist = torch.distributions.Categorical(logits=logits.unsqueeze(1))
        new_log_probs = dist.log_prob(responses)  # (batch, K)
        
        # RLOO核心损失: 纯REINFORCE，无PPO裁剪
        policy_loss = -(new_log_probs * advantages).mean()
        
        # 5. KL惩罚
        kl = self.compute_kl(prompts, logits).mean()
        
        total_loss = policy_loss + self.beta * kl
        
        total_loss.backward()
        self.optimizer.step()
        
        self.stats['loss'].append(total_loss.item())
        self.stats['reward'].append(rewards.mean().item())
        self.stats['kl'].append(kl.item())
        
        return {
            'loss': total_loss.item(),
            'reward': rewards.mean().item(),
            'kl': kl.item(),
            'advantage_std': advantages.std().item()
        }


def demo():
    """RLOO训练演示"""
    print("=" * 60)
    print("RLOO (REINFORCE Leave-One-Out)")
    print("Back to Basics: 纯REINFORCE + LOO基线 算法演示")
    print("=" * 60)
    
    print("\nRLOO vs PPO 对比:")
    print("  PPO:  需要 Policy + Reference + Critic + Reward Model")
    print("  RLOO: 只需 Policy + Reference (无需Critic!)")
    print()
    print("RLOO 核心优势:")
    print("  - LOO优势: A_k = (K/(K-1)) * (r_k - mean)")
    print("  - 无偏梯度: LOO基线是无偏的 (Critic有偏)")
    print("  - 无需PPO裁剪: 纯REINFORCE损失")
    print("  - 节省约33%内存: 去掉了Critic网络")
    print()
    
    input_dim = 64
    hidden_dim = 256
    output_dim = 50
    
    policy = RLOOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy = RLOOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy.load_state_dict(policy.state_dict())
    
    trainer = RLOOTrainer(policy, ref_policy, rloo_k=4, beta=0.04)
    
    steps = 200
    for step in range(steps):
        prompts = torch.randn(8, input_dim)
        targets = torch.randint(0, output_dim, (8,))
        
        stats = trainer.train_step(prompts, targets)
        
        if step % 40 == 0:
            print(
                f"Step {step:3d} | "
                f"Loss: {stats['loss']:.4f} | "
                f"Reward: {stats['reward']:.3f} | "
                f"KL: {stats['kl']:.4f} | "
                f"Adv Std: {stats['advantage_std']:.3f}"
            )
    
    print(f"\nRLOO训练完成!")
    print(f"最终奖励: {stats['reward']:.3f}")
    print(f"最终KL: {stats['kl']:.4f}")
    
    return trainer.stats


if __name__ == "__main__":
    stats = demo()
