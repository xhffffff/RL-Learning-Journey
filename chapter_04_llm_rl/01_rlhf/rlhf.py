"""
RLHF (Reinforcement Learning from Human Feedback) 核心逻辑实现
-----------------------------------------------------------------
论文: Training language models to follow instructions with human feedback
作者: OpenAI (2022)

本实现聚焦于RLHF的训练范式核心概念，使用一个简化的
"语言模型" (实际上是MLP策略) 在偏好学习任务上演示RLHF流程。

RLHF三阶段:
1. SFT (Supervised Fine-Tuning): 克隆人类高质量响应
2. RM (Reward Model): 从人类偏好中学习奖励函数
3. PPO Fine-Tuning: 使用RM奖励 + KL惩罚微调模型

完整LLM训练中，所有这些步骤在数TB的文本数据上进行，
本实现提供缩小的教育版本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from collections import namedtuple


# ============================================================
# 阶段1: SFT (Supervised Fine-Tuning)
# ============================================================

class SmallLanguageModel(nn.Module):
    """
    简化的"语言模型"
    在实际LLM训练中，这会是一个Transformer(GPT/LLaMA)
    这里用MLP在简单分类任务上演示
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, vocab_size: int = 100):
        super(SmallLanguageModel, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.lm_head(features)
    
    def get_output_distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs)


def sft_training(
    model: SmallLanguageModel,
    prompt_response_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 100,
    lr: float = 1e-3
) -> SmallLanguageModel:
    """
    阶段1: 监督微调
    模仿学习 - 学习从 prompt 到高质量 response 的映射
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for prompt, target_response in prompt_response_pairs:
            optimizer.zero_grad()
            logits = model(prompt)
            loss = F.cross_entropy(logits, target_response)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"SFT Epoch {epoch:3d} | Loss: {total_loss:.4f}")
    
    return model


# ============================================================
# 阶段2: RM (Reward Model Training)
# ============================================================

class RewardModel(nn.Module):
    """
    奖励模型
    学习预测人类偏好分数 r(y|x)
    
    输入: prompt + response 的表示
    输出: 标量奖励分数
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super(RewardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def reward_model_training(
    rm: RewardModel,
    preferences: List[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 200,
    lr: float = 1e-3
) -> RewardModel:
    """
    阶段2: 奖励模型训练
    使用 Bradley-Terry 偏好模型
    
    P(1 > 2) = σ(r(chosen) - r(rejected))
    
    损失: -log σ(r(chosen) - r(rejected))
    """
    optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for chosen, rejected in preferences:
            optimizer.zero_grad()
            
            r_chosen = rm(chosen)
            r_rejected = rm(rejected)
            
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (r_chosen > r_rejected).float().mean().item()
        
        if epoch % 40 == 0:
            acc = correct / len(preferences)
            print(f"RM Epoch {epoch:3d} | Loss: {total_loss:.4f} | Acc: {acc:.2%}")
    
    return rm


# ============================================================
# 阶段3: PPO Fine-Tuning (RLHF核心)
# ============================================================

class RLHFTrainer:
    """
    阶段3: 使用RLHF微调语言模型
    
    RLHF目标:
    max_θ E_{x~D, y~π_θ(·|x)} [r_φ(y|x) - β·log π_θ(y|x)/π_ref(y|x)]
    
    其中:
    - r_φ: 奖励模型评分
    - β: KL惩罚系数
    - π_ref: 参考模型 (SFT后的模型，冻结)
    
    实际计算:
    - 奖励: r_total = r_rm - β * KL_penalty
    - 使用PPO更新策略
    """
    
    def __init__(
        self,
        policy: SmallLanguageModel,
        ref_policy: SmallLanguageModel,
        reward_model: RewardModel,
        beta: float = 0.1,      # KL惩罚系数
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.reward_model = reward_model.to(device)
        self.beta = beta
        self.device = device
        
        # 冻结参考模型
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.stats = {
            'reward_rm': [],
            'kl_penalty': [],
            'total_reward': []
        }
    
    def compute_kl_penalty(
        self, prompt: torch.Tensor, response_logits: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度惩罚"""
        with torch.no_grad():
            ref_logits = self.ref_policy(prompt)
        
        policy_probs = F.softmax(response_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        # KL(policy || ref)
        kl = (policy_probs * (torch.log(policy_probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1)
        return kl.mean()
    
    def train_step(self, prompts: torch.Tensor) -> dict:
        """
        单步PPO训练在RLHF中
        """
        self.optimizer.zero_grad()
        
        # 1. 从当前策略采样响应
        logits = self.policy(prompts)
        dist = torch.distributions.Categorical(logits=logits)
        responses = dist.sample()
        
        # 2. 计算奖励模型奖励
        # 【简化】RM输入为单token one-hot，真实场景中RM应接收完整生成序列
        response_one_hot = F.one_hot(responses, num_classes=self.policy.vocab_size).float()
        rm_reward = self.reward_model(response_one_hot)
        
        # 3. 计算KL惩罚
        # 【简化】KL在prompt层面计算，真实场景在完整response序列层面计算
        kl_penalty = self.compute_kl_penalty(prompts, logits)
        
        # 4. RLHF总奖励
        total_reward = rm_reward.mean() - self.beta * kl_penalty
        
        # 5. 简化REINFORCE损失（完整PPO需要重要性采样比率+裁剪机制）
        log_probs = dist.log_prob(responses)
        policy_loss = -(log_probs * total_reward.detach()).mean()
        
        policy_loss.backward()
        self.optimizer.step()
        
        self.stats['reward_rm'].append(rm_reward.mean().item())
        self.stats['kl_penalty'].append(kl_penalty.item())
        self.stats['total_reward'].append(total_reward.item())
        
        return {
            'reward_rm': rm_reward.mean().item(),
            'kl_penalty': kl_penalty.item(),
            'total_reward': total_reward.item()
        }


# ============================================================
# 演示脚本
# ============================================================

def demo():
    """RLHF流程完整演示"""
    print("=" * 60)
    print("RLHF (Reinforcement Learning from Human Feedback)")
    print("演示: 简化版RLHF三阶段训练")
    print("=" * 60)
    
    input_dim = 64
    hidden_dim = 256
    vocab_size = 50
    
    # ---- 阶段1: SFT ----
    print("\n[阶段1] SFT - 监督微调")
    print("-" * 40)
    
    sft_model = SmallLanguageModel(input_dim, hidden_dim, vocab_size)
    
    # 生成模拟的SFT数据
    sft_data = []
    for i in range(100):
        prompt = torch.randn(1, input_dim)
        target = torch.randint(0, vocab_size, (1,))
        sft_data.append((prompt, target))
    
    sft_model = sft_training(sft_model, sft_data, epochs=100)
    
    # 保存SFT模型作为参考
    ref_model = SmallLanguageModel(input_dim, hidden_dim, vocab_size)
    ref_model.load_state_dict(sft_model.state_dict())
    
    # ---- 阶段2: 奖励模型训练 ----
    print("\n[阶段2] RM - 奖励模型训练")
    print("-" * 40)
    
    reward_model = RewardModel(input_dim=vocab_size)
    
    # 生成模拟的偏好数据
    preferences = []
    for i in range(200):
        chosen = torch.randn(1, vocab_size)   # 好的响应
        rejected = torch.randn(1, vocab_size)  # 差的响应
        preferences.append((chosen, rejected))
    
    reward_model = reward_model_training(reward_model, preferences, epochs=200)
    
    # ---- 阶段3: PPO Fine-Tuning ----
    print("\n[阶段3] RLHF - PPO Fine-Tuning")
    print("-" * 40)
    
    trainer = RLHFTrainer(sft_model, ref_model, reward_model, beta=0.1)
    
    for step in range(100):
        prompts = torch.randn(4, input_dim)
        stats = trainer.train_step(prompts)
        
        if step % 20 == 0:
            print(
                f"RLHF Step {step:3d} | "
                f"RM Reward: {stats['reward_rm']:.4f} | "
                f"KL: {stats['kl_penalty']:.4f} | "
                f"Total: {stats['total_reward']:.4f}"
            )
    
    print("\n" + "=" * 60)
    print("RLHF 三阶段训练流程演示完成!")
    print("完整流程: SFT → RM Training → PPO Fine-Tuning")
    print("=" * 60)
    
    return trainer.stats


if __name__ == "__main__":
    stats = demo()
