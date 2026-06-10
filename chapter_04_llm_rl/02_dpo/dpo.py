"""
DPO (Direct Preference Optimization) 核心实现
-----------------------------------------------------------
论文: Direct Preference Optimization: Your Language Model is
      Secretly a Reward Model
作者: Rafailov et al. (Stanford, 2023)

核心思想:
直接从偏好数据优化策略，无需显式奖励模型

DPO损失函数:
L_DPO = -log σ( β·log[π_θ(y_w|x)/π_ref(y_w|x)]
               - β·log[π_θ(y_l|x)/π_ref(y_l|x)] )

其中:
- π_θ: 正在训练的策略模型
- π_ref: 参考模型 (SFT模型，冻结)
- y_w: 偏好响应 (chosen/preferred)
- y_l: 非偏好响应 (rejected/dispreferred)
- β: 温度参数，控制偏离参考模型的程度

DPO vs RLHF:
+ 无需训练奖励模型
+ 训练更稳定
+ 计算效率更高
- 偏好需要离线收集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PreferenceBatch:
    """偏好数据批次"""
    prompt: torch.Tensor      # 提示
    chosen: torch.Tensor      # 偏好响应
    rejected: torch.Tensor    # 非偏好响应


class DPOModel(nn.Module):
    """
    DPO策略模型
    在实际LLM中，这就是语言模型本身 (GPT/LLaMA等)
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 50):
        super(DPOModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.shared(x))
    
    def get_log_probs(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算给定标签的对数概率
        log π_θ(labels|x)
        """
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)


class DPOTrainer:
    """
    Direct Preference Optimization 训练器
    
    【简化说明】
    本实现使用MLP + 单token分类来演示DPO损失。
    真实DPO中，log π(y|x) 是对整个响应序列所有token的对数概率求和：
    log π(y|x) = Σ_t log π(y_t | x, y_{<t})
    这里的实现将"响应"简化为单个token标签进行分类。
    
    参数:
        policy: 训练中的策略模型
        ref_model: 参考模型 (SFT后，冻结)
        beta: 温度参数 (通常0.1-0.5)
        lr: 学习率
    """
    
    def __init__(
        self,
        policy: DPOModel,
        ref_model: DPOModel,
        beta: float = 0.1,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy.to(device)
        self.ref_model = ref_model.to(device)
        self.beta = beta
        self.device = device
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.stats = {
            'loss': [],
            'chosen_logps': [],
            'rejected_logps': [],
            'accuracy': []
        }
    
    def dpo_loss(
        self, batch: PreferenceBatch
    ) -> Tuple[torch.Tensor, dict]:
        """
        核心DPO损失函数
        
        L_DPO = -log σ(β·[log(π_w/π_ref_w) - log(π_l/π_ref_l)])
        """
        # 策略模型的对数概率
        policy_chosen_logps = self.policy.get_log_probs(batch.prompt, batch.chosen)
        policy_rejected_logps = self.policy.get_log_probs(batch.prompt, batch.rejected)
        
        # 参考模型的对数概率
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.get_log_probs(batch.prompt, batch.chosen)
            ref_rejected_logps = self.ref_model.get_log_probs(batch.prompt, batch.rejected)
        
        # 对数比率
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
        
        # DPO损失
        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()
        
        # 统计信息
        with torch.no_grad():
            acc = (log_ratio_chosen > log_ratio_rejected).float().mean()
        
        stats = {
            'chosen_logps': policy_chosen_logps.mean().item(),
            'rejected_logps': policy_rejected_logps.mean().item(),
            'log_ratio_chosen': log_ratio_chosen.mean().item(),
            'log_ratio_rejected': log_ratio_rejected.mean().item(),
            'accuracy': acc.item()
        }
        
        return loss, stats
    
    def train_step(self, batch: PreferenceBatch) -> dict:
        self.optimizer.zero_grad()
        
        loss, stats = self.dpo_loss(batch)
        loss.backward()
        self.optimizer.step()
        
        self.stats['loss'].append(loss.item())
        self.stats['chosen_logps'].append(stats['chosen_logps'])
        self.stats['rejected_logps'].append(stats['rejected_logps'])
        self.stats['accuracy'].append(stats['accuracy'])
        
        return {
            'loss': loss.item(),
            'chosen_logps': stats['chosen_logps'],
            'rejected_logps': stats['rejected_logps'],
            'accuracy': stats['accuracy']
        }


def demo():
    """DPO训练演示"""
    print("=" * 60)
    print("DPO (Direct Preference Optimization)")
    print("演示: 直接从偏好数据学习")
    print("=" * 60)
    
    input_dim = 64
    hidden_dim = 256
    output_dim = 50
    
    # 初始化策略模型和参考模型
    policy = DPOModel(input_dim, hidden_dim, output_dim)
    ref_model = DPOModel(input_dim, hidden_dim, output_dim)
    ref_model.load_state_dict(policy.state_dict())
    
    trainer = DPOTrainer(policy, ref_model, beta=0.1)
    
    # 演示不同 beta 值对 KL 惩罚的影响
    print("\nDPO核心公式解析:")
    print("  L = -log σ(β · [log(π_w/π_ref_w) - log(π_l/π_ref_l)])")
    print("  β=0.0: 不惩罚 (模型自由偏离参考)")
    print("  β=0.1: 适度惩罚 (推荐)")
    print("  β=1.0: 强惩罚   (更接近参考模型)")
    print()
    
    batches = 200
    for step in range(batches):
        # 生成模拟偏好数据
        batch = PreferenceBatch(
            prompt=torch.randn(8, input_dim),
            chosen=torch.randint(0, output_dim, (8,)),
            rejected=torch.randint(0, output_dim, (8,))
        )
        
        stats = trainer.train_step(batch)
        
        if step % 40 == 0:
            print(
                f"Step {step:3d} | "
                f"Loss: {stats['loss']:.4f} | "
                f"Chosen logP: {stats['chosen_logps']:.2f} | "
                f"Rejected logP: {stats['rejected_logps']:.2f} | "
                f"Acc: {stats['accuracy']:.2%}"
            )
    
    print(f"\n最终准确率: {stats['accuracy']:.2%}")
    print("DPO训练完成! 模型已学会区分偏好和非偏好响应")
    
    return trainer.stats


if __name__ == "__main__":
    stats = demo()
