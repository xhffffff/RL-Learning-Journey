"""
GRPO (Group Relative Policy Optimization) 核心实现
-----------------------------------------------------------------
论文: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
作者: DeepSeek-AI (2025)

GRPO是DeepSeek-R1的核心训练算法，用于激发LLM的推理能力。

关键创新 vs PPO:

标准PPO:
  需要4个模型: Policy, Reference, Critic, Reward Model
  使用GAE + Critic网络估计优势

GRPO:
  只需2个模型: Policy, Reference
  使用组内相对奖励作为优势估计 (无需Critic!)
  
核心公式:
  A_i = (R_i - mean_group(R)) / std_group(R)
  L = -1/G Σ min(r_i·A_i, clip(r_i, 1-ε, 1+ε)·A_i)

奖励设计:
  准确率奖励: 答案是否正确
  格式奖励: 输出是否符合格式要求
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class GRPOPolicy(nn.Module):
    """GRPO策略模型 (简化版)"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 50):
        super(GRPOPolicy, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.shared(x))


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO)
    
    核心特点:
    1. 无需Critic网络 (用组内统计量替代)
    2. 轻量级 (比PPO少一半模型参数)
    3. 适合推理任务 (用规则奖励替代复杂奖励模型)
    
    参数:
        policy: 训练中的策略模型
        ref_policy: 参考模型 (冻结)
        group_size: 每组采样的响应数 G
        clip_epsilon: 裁剪参数 ε
        beta: KL惩罚系数
        lr: 学习率
    """
    
    def __init__(
        self,
        policy: GRPOPolicy,
        ref_policy: GRPOPolicy,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        beta: float = 0.04,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.device = device
        
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.stats = {'loss': [], 'reward': [], 'kl': []}
    
    def group_relative_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        GRPO的核心: 组内相对优势
        
        对每个prompt，采样G个响应，计算:
        A_i = (R_i - mean({R_1,...,R_G})) / (std({R_1,...,R_G}) + ε)
        
        优势归一化的好处:
        - 移除奖励的绝对值偏差
        - 强调组内相对好坏
        """
        mean_r = rewards.mean(dim=-1, keepdim=True)
        std_r = rewards.std(dim=-1, keepdim=True)
        return (rewards - mean_r) / (std_r + 1e-8)
    
    def compute_reward(
        self, responses: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        规则奖励函数
        
        DeepSeek-R1使用的两类奖励:
        1. 准确率奖励: 答案与标准答案匹配
        2. 格式奖励: 输出格式符合要求
        
        这里简化为分类准确率
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
        单步GRPO训练
        
        GRPO算法流程:
        1. 对每个prompt采样G个响应
        2. 计算规则奖励
        3. 计算组内相对优势
        4. 使用裁剪目标 + KL惩罚更新策略
        """
        batch_size = prompts.shape[0]
        G = self.group_size
        
        # 1. 对每个prompt采样G个响应（采样时存储old_log_probs）
        self.policy.eval()
        all_responses = []
        all_old_log_probs = []
        all_rewards = []
        
        with torch.no_grad():
            logits = self.policy(prompts)
            for _ in range(G):
                dist = torch.distributions.Categorical(logits=logits)
                response = dist.sample()
                log_prob = dist.log_prob(response)
                reward = (response == targets).float()
                
                all_responses.append(response)
                all_old_log_probs.append(log_prob)
                all_rewards.append(reward)
        
        # 2. 堆叠为 (batch, G)
        responses = torch.stack(all_responses, dim=1)      # (batch, G)
        old_log_probs = torch.stack(all_old_log_probs, dim=1)  # (batch, G)
        rewards = torch.stack(all_rewards, dim=1)              # (batch, G)
        
        # 3. 计算组内相对优势（在group_size维度上归一化）
        advantages = self.group_relative_advantage(rewards)    # (batch, G)
        
        # 4. 优化阶段：重新计算新策略的log_probs
        self.policy.train()
        self.optimizer.zero_grad()
        
        logits = self.policy(prompts)  # (batch, vocab)
        # 将logits扩展到(batch, 1, vocab)，使Categorical分布接受(batch, G)的responses
        dist = torch.distributions.Categorical(logits=logits.unsqueeze(1))
        new_log_probs = dist.log_prob(responses)  # (batch, G)
        
        ratio = (new_log_probs - old_log_probs).exp()
        
        # 5. 裁剪目标 (与PPO相同)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 6. KL惩罚
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
    """GRPO训练演示"""
    print("=" * 60)
    print("GRPO (Group Relative Policy Optimization)")
    print("DeepSeek-R1 核心算法演示")
    print("=" * 60)
    
    print("\nGRPO vs PPO 对比:")
    print("  PPO:  需要 Policy + Reference + Critic + Reward Model")
    print("  GRPO: 只需 Policy + Reference (无需Critic!)")
    print()
    print("GRPO 核心优势:")
    print("  - 组内相对优势: A_i = (R_i - mean) / std")
    print("  - 无需Critic网络: 减少50%参数量")
    print("  - 规则奖励: 准确率 + 格式 (无需RM)")
    print()
    
    input_dim = 64
    hidden_dim = 256
    output_dim = 50
    
    policy = GRPOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy = GRPOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy.load_state_dict(policy.state_dict())
    
    trainer = GRPOTrainer(policy, ref_policy, group_size=4, beta=0.04)
    
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
    
    print(f"\nGRPO训练完成!")
    print(f"最终奖励: {stats['reward']:.3f}")
    print(f"最终KL: {stats['kl']:.4f}")
    
    return trainer.stats


if __name__ == "__main__":
    stats = demo()
