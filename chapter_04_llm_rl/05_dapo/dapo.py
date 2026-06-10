"""
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 核心实现
-----------------------------------------------------------------
论文: DAPO: An Open-Source LLM RL System at Scale
作者: ByteDance Seed + Tsinghua AIR (2025, arXiv:2503.14476)

DAPO是从GRPO/PPO演化而来的LLM对齐算法，关键结果:
在AIME 2024上达到50%准确率（DeepSeek-R1为47%，且只用一半训练步数）

四大核心创新 vs GRPO/PPO:

1. Clip-Higher (解耦裁剪):
   标准PPO/GRPO: clip(r, 1-ε, 1+ε)  对称裁剪
   DAPO:       clip(r, 1-ε_low, 1+ε_high)  上限更高
   让模型有更大的探索空间，更自由地提升好响应的概率

2. Dynamic Sampling (动态采样过滤):
   过滤掉所有G个响应奖励相同的prompt（std=0）
   只保留奖励有差异的prompt → 更有效的梯度更新

3. Token-Level Loss (逐token损失):
   损失分母 = Σ|a_i| (总token数) 而非 G (响应数)
   在简化MLP设置中，使用所有保留样本贡献的平均

4. Overlong Reward Shaping (过长惩罚):
   R_shaped = R_original - α * penalty_factor
   对低质量/高不确定性的响应施加软惩罚

核心公式:
  A_i = (R_i - mean_group(R)) / std_group(R)
  J_DAPO = -(1/|V|) Σ_{i∈V} min(r_i·A_i, clip(r_i, 1-ε_low, 1+ε_high)·A_i) + β·KL
  其中 V 是经Dynamic Sampling保留的prompt集合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class DAPOPolicy(nn.Module):
    """DAPO策略模型 (简化版)"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 50):
        super(DAPOPolicy, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.shared(x))


class DAPOTrainer:
    """
    Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)
    
    核心特点:
    1. 解耦裁剪: 更高的上界(ε_high > ε_low)鼓励探索
    2. 动态采样: 过滤无信息量的prompt (所有响应奖励相同)
    3. 逐token损失: 更精细的梯度归一化
    4. 过长惩罚: 降低低质量响应的奖励
    
    参数:
        policy: 训练中的策略模型
        ref_policy: 参考模型 (冻结)
        group_size: 每组采样的响应数 G
        clip_epsilon_low: 裁剪下界 ε_low
        clip_epsilon_high: 裁剪上界 ε_high
        beta: KL惩罚系数
        overlong_alpha: 过长惩罚系数 α
        lr: 学习率
    """
    
    def __init__(
        self,
        policy: DAPOPolicy,
        ref_policy: DAPOPolicy,
        group_size: int = 8,
        clip_epsilon_low: float = 0.2,
        clip_epsilon_high: float = 0.28,
        beta: float = 0.04,
        overlong_alpha: float = 0.05,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.group_size = group_size
        self.clip_epsilon_low = clip_epsilon_low
        self.clip_epsilon_high = clip_epsilon_high
        self.beta = beta
        self.overlong_alpha = overlong_alpha
        self.device = device
        
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.stats = {'loss': [], 'reward': [], 'kl': [], 'kept_ratio': []}
    
    def group_relative_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        组内相对优势 (与GRPO相同)
        
        对每个prompt，采样G个响应，计算:
        A_i = (R_i - mean({R_1,...,R_G})) / (std({R_1,...,R_G}) + ε)
        
        优势归一化的好处:
        - 移除奖励的绝对值偏差
        - 强调组内相对好坏
        """
        mean_r = rewards.mean(dim=-1, keepdim=True)
        std_r = rewards.std(dim=-1, keepdim=True)
        return (rewards - mean_r) / (std_r + 1e-8)
    
    def dynamic_sampling_filter(
        self, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        DAPO核心: 动态采样过滤
        
        过滤掉所有G个响应奖励相同的prompt (std=0)
        只保留奖励有方差的prompt用于梯度更新
        
        原理: 如果所有G个响应奖励相同，说明该prompt
        当前策略下没有区分度，梯度更新无意义
        """
        reward_std = rewards.std(dim=-1)  # (batch,)
        valid_mask = reward_std > 1e-8
        return valid_mask
    
    def compute_reward(
        self, responses: torch.Tensor, target: torch.Tensor,
        old_log_probs: torch.Tensor, apply_overlong: bool = True
    ) -> torch.Tensor:
        """
        规则奖励函数 + DAPO过长惩罚
        
        DAPO的奖励包含两部分:
        1. 准确率奖励: 答案与标准答案匹配
        2. 过长惩罚 (Overlong Reward Shaping):
           R_shaped = R_original - α * penalty_factor
           其中 penalty_factor = |old_log_prob| (不确定性越大惩罚越重)
        
        【简化说明】
        在MLP设置中，用负对数概率模拟"响应长度/质量"，
        概率越低的采样越被惩罚，鼓励模型产出高置信度答案。
        """
        accuracy = (responses == target.unsqueeze(1)).float()
        
        if apply_overlong:
            penalty_factor = torch.abs(old_log_probs)
            rewards = accuracy - self.overlong_alpha * penalty_factor
        else:
            rewards = accuracy
        
        return rewards
    
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
        单步DAPO训练
        
        DAPO算法流程 (四大创新标注 ★):
        1. 对每个prompt采样G个响应
        2. 存储old_log_probs
        3. 计算奖励 + ★ 过长惩罚
        4. ★ 动态采样过滤
        5. 计算组内相对优势
        6. 计算new_log_probs + ratio
        7. ★ 解耦裁剪: clip(ratio, 1-ε_low, 1+ε_high)
        8. ★ 逐token损失 + KL惩罚
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
        responses = torch.stack(all_responses, dim=1)           # (batch, G)
        old_log_probs = torch.stack(all_old_log_probs, dim=1)   # (batch, G)
        rewards_raw = torch.stack(all_rewards, dim=1)           # (batch, G)
        
        # 3. ★ 过长惩罚 (Overlong Reward Shaping)
        rewards = self.compute_reward(
            responses, targets, old_log_probs, apply_overlong=True
        )
        
        # 4. ★ 动态采样过滤（基于原始奖励，过滤所有响应都正确/都错误的prompt）
        valid_mask = self.dynamic_sampling_filter(rewards_raw)  # (batch,)
        num_kept = valid_mask.sum().item()
        kept_ratio = num_kept / batch_size if batch_size > 0 else 0.0
        
        # 如果没有可用的prompt，跳过本次更新
        if num_kept == 0:
            self.stats['loss'].append(0.0)
            self.stats['reward'].append(rewards_raw.mean().item())
            self.stats['kl'].append(0.0)
            self.stats['kept_ratio'].append(kept_ratio)
            return {
                'loss': 0.0,
                'reward': rewards_raw.mean().item(),
                'kl': 0.0,
                'kept_ratio': kept_ratio
            }
        
        # 保留有效prompt
        kept_prompts = prompts[valid_mask]            # (num_kept, input_dim)
        kept_responses = responses[valid_mask]        # (num_kept, G)
        kept_old_log_probs = old_log_probs[valid_mask]  # (num_kept, G)
        kept_rewards = rewards[valid_mask]            # (num_kept, G)
        
        # 5. 计算组内相对优势（在group_size维度上归一化）
        advantages = self.group_relative_advantage(kept_rewards)  # (num_kept, G)
        
        # 6. 优化阶段：重新计算新策略的log_probs
        self.policy.train()
        self.optimizer.zero_grad()
        
        logits = self.policy(kept_prompts)  # (num_kept, vocab)
        dist = torch.distributions.Categorical(logits=logits.unsqueeze(1))
        new_log_probs = dist.log_prob(kept_responses)  # (num_kept, G)
        
        ratio = (new_log_probs - kept_old_log_probs).exp()
        
        # 7. ★ 解耦裁剪 (Clip-Higher)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon_low,
            1 + self.clip_epsilon_high
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 8. KL惩罚
        kl = self.compute_kl(kept_prompts, logits).mean()
        
        total_loss = policy_loss + self.beta * kl
        
        total_loss.backward()
        self.optimizer.step()
        
        self.stats['loss'].append(total_loss.item())
        self.stats['reward'].append(kept_rewards.mean().item())
        self.stats['kl'].append(kl.item())
        self.stats['kept_ratio'].append(kept_ratio)
        
        return {
            'loss': total_loss.item(),
            'reward': kept_rewards.mean().item(),
            'kl': kl.item(),
            'kept_ratio': kept_ratio,
            'advantage_std': advantages.std().item()
        }


def demo():
    """DAPO训练演示"""
    print("=" * 60)
    print("DAPO (Decoupled Clip and Dynamic sAmpling")
    print("      Policy Optimization)")
    print("ByteDance Seed + Tsinghua AIR (2025)")
    print("=" * 60)
    
    print("\nDAPO vs GRPO/PPO 对比:")
    print("  PPO:   Policy + Reference + Critic + Reward Model")
    print("  GRPO:  Policy + Reference (组内优势)")
    print("  DAPO:  Policy + Reference + 四大创新")
    print()
    print("DAPO 四大核心创新 ★:")
    print("  ★1. Clip-Higher: ε_low=0.2, ε_high=0.28 (非对称)")
    print("  ★2. Dynamic Sampling: 过滤std=0的prompt")
    print("  ★3. Token-Level Loss: 逐token归一化")
    print("  ★4. Overlong Shaping: 惩罚不确定性")
    print()
    
    input_dim = 64
    hidden_dim = 256
    output_dim = 50
    
    policy = DAPOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy = DAPOPolicy(input_dim, hidden_dim, output_dim)
    ref_policy.load_state_dict(policy.state_dict())
    
    trainer = DAPOTrainer(
        policy, ref_policy,
        group_size=8,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        beta=0.04,
        overlong_alpha=0.05
    )
    
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
                f"Kept: {stats['kept_ratio']:.2%}"
            )
    
    print(f"\nDAPO训练完成!")
    print(f"最终奖励: {stats['reward']:.3f}")
    print(f"最终KL: {stats['kl']:.4f}")
    print(f"最终保留率: {stats['kept_ratio']:.2%}")
    
    return trainer.stats


if __name__ == "__main__":
    stats = demo()
