"""
第四章测试: RLHF, DPO, GRPO
(LLM时代的强化学习)
"""
import sys, os
import importlib.util
import numpy as np
import torch

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = os.path.dirname(__file__)

print('=' * 70)
print('第四章 大模型时代的强化学习 (2022-至今)')
print('=' * 70)
print()
print('本章与前三章的本质区别:')
print('  前三章: 智能体与环境交互获取 reward (游戏/机器人)')
print('  第四章: 从人类偏好数据中学习 (文本质量/安全性/推理)')
print('  奖励来源: 环境规则 -> 人类偏好 -> 规则奖励')
print()

# ============ 1. RLHF ============
print('--- 1. RLHF (2022, OpenAI - InstructGPT) ---')
print()
print('【数据获取方式 —— 三阶段训练，每阶段数据不同!】')
print()
print('  阶段1: SFT (监督微调)')
print('    数据: 人类编写的高质量 (prompt, response) 对')
print('    格式: {"prompt": "解释机器学习", "response": "机器学习是AI分支..."}')
print('    来源: 人工标注 / 高质量数据筛选')
print('    目标: 让模型学会模仿人类风格')
print()
print('  阶段2: RM (奖励模型训练)')
print('    数据: 人类偏好对 (chosen, rejected)')
print('    格式: {"prompt": "...", "chosen": "好的回答", "rejected": "差的回答"}')
print('    来源: 人工排序 / 标注员评分')
print('    目标: 训练一个能判断回答好坏的打分模型')
print()
print('  阶段3: PPO Fine-Tuning')
print('    数据: 大量无标注 prompt (不需要人类标注!)')
print('    强化信号: 来自阶段2训练好的奖励模型')
print('    奖励公式: total_reward = RM_score - beta * KL(policy || ref_model)')
print('    KL惩罚: 防止模型偏离太远 (变成"奖励黑客")')
print()
print('  测试环境: 简化版MLP演示 (实际LLM是GPT/LLaMA级, 千亿参数)')
print()

mod = load_module('rlhf', os.path.join(ROOT, 'chapter_04_llm_rl', '01_rlhf', 'rlhf.py'))
SmallLanguageModel = mod.SmallLanguageModel
RewardModel = mod.RewardModel
sft_training = mod.sft_training
reward_model_training = mod.reward_model_training
RLHFTrainer = mod.RLHFTrainer

input_dim, hidden_dim, vocab_size = 64, 128, 50

# 阶段1: SFT
print('  [阶段1] SFT - 监督微调')
sft_model = SmallLanguageModel(input_dim, hidden_dim, vocab_size)
sft_data = [(torch.randn(1, input_dim), torch.randint(0, vocab_size, (1,))) for _ in range(50)]
sft_model = sft_training(sft_model, sft_data, epochs=40)
print(f'   SFT完成! 模型参数量: {sum(p.numel() for p in sft_model.parameters()):,}')

# 阶段2: RM
print('  [阶段2] RM - 奖励模型训练')
rm = RewardModel(input_dim=vocab_size)
prefs = [(torch.randn(1, vocab_size), torch.randn(1, vocab_size)) for _ in range(100)]
rm = reward_model_training(rm, prefs, epochs=40)
print(f'   RM完成!')

# 阶段3: PPO
print('  [阶段3] PPO - RLHF微调')
ref = SmallLanguageModel(input_dim, hidden_dim, vocab_size)
ref.load_state_dict(sft_model.state_dict())
trainer = RLHFTrainer(sft_model, ref, rm, beta=0.1)
for step in range(20):
    prompts = torch.randn(4, input_dim)
    stats = trainer.train_step(prompts)
print(f'   RLHF RM Reward: {stats["reward_rm"]:.4f}')
print(f'   RLHF KL Penalty: {stats["kl_penalty"]:.4f}')
print(f'   RLHF Total: {stats["total_reward"]:.4f}')
print('  RLHF => OK')

# ============ 2. DPO ============
print()
print('--- 2. DPO (2023, Rafailov et al., Stanford) ---')
print()
print('【数据获取方式 —— 比RLHF更简单!】')
print('  DPO只需一种数据: 偏好对 (chosen, rejected)')
print('  不需要显式训练奖励模型!')
print()
print('  数据格式:')
print('    {"prompt": x, "chosen": y_w, "rejected": y_l}')
print('    chosen = 人类更喜欢的回答')
print('    rejected = 人类不喜欢的回答')
print()
print('  损失函数: L_DPO = -log sigma(beta * [log(pi_w/ref_w) - log(pi_l/ref_l)])')
print('  直接增大chosen概率，减小rejected概率')
print()
print('  优势 vs RLHF:')
print('    DPO: 1个模型 + 偏好数据 = 对齐 (简单)')
print('    RLHF: 3个模型(SFT+RM+PPO) + 3种数据 = 对齐 (复杂)')
print()

mod = load_module('dpo', os.path.join(ROOT, 'chapter_04_llm_rl', '02_dpo', 'dpo.py'))
DPOModel = mod.DPOModel
DPOTrainer = mod.DPOTrainer
PreferenceBatch = mod.PreferenceBatch

policy = DPOModel(input_dim, hidden_dim, vocab_size)
ref_policy = DPOModel(input_dim, hidden_dim, vocab_size)
ref_policy.load_state_dict(policy.state_dict())

dpo_trainer = DPOTrainer(policy, ref_policy, beta=0.1)
for step in range(40):
    batch = PreferenceBatch(
        prompt=torch.randn(8, input_dim),
        chosen=torch.randint(0, vocab_size, (8,)),
        rejected=torch.randint(0, vocab_size, (8,))
    )
    stats = dpo_trainer.train_step(batch)
print(f'  DPO Loss: {stats["loss"]:.4f}')
print(f'  DPO Chosen logP: {stats["chosen_logps"]:.2f}')
print(f'  DPO Rejected logP: {stats["rejected_logps"]:.2f}')
print(f'  DPO Acc: {stats["accuracy"]:.2%}')
print('  DPO => OK')

# ============ 3. GRPO ============
print()
print('--- 3. GRPO (2025, DeepSeek-AI - DeepSeek-R1) ---')
print()
print('【数据获取方式 —— 规则奖励! 无需人类标注!】')
print('  GRPO既不需要环境也不需要人类标注!')
print('  数据来源:')
print('    1. 大量数学/编程题目 (prompts)')
print('    2. 规则奖励: 自动判断答案是否正确!')
print('       - 准确率奖励: 对比标准答案')
print('       - 格式奖励: 检查输出格式 (如是否包含<think>标签)')
print()
print('  关键创新: 组内相对优势')
print('    对每个prompt采样G=4个响应')
print('    A_i = (R_i - mean_group_R) / std_group_R')
print('    无需Critic网络! 用组内统计量替代')
print()
print('  GRPO vs PPO:')
print('    PPO: 需要 Policy + Ref + Critic + Reward Model (4个模型)')
print('    GRPO: 只需 Policy + Ref (2个模型)')
print('    减少50%参数量!')
print()
print('  DeepSeek-R1的核心训练算法，激发推理能力(CoT)')
print('  测试: 简化版MLP演示')
print()

mod = load_module('grpo', os.path.join(ROOT, 'chapter_04_llm_rl', '03_grpo', 'grpo.py'))
GRPOPolicy = mod.GRPOPolicy
GRPOTrainer = mod.GRPOTrainer

policy2 = GRPOPolicy(input_dim, hidden_dim, vocab_size)
ref2 = GRPOPolicy(input_dim, hidden_dim, vocab_size)
ref2.load_state_dict(policy2.state_dict())

grpo_trainer = GRPOTrainer(policy2, ref2, group_size=4, beta=0.04)
for step in range(40):
    prompts = torch.randn(8, input_dim)
    targets = torch.randint(0, vocab_size, (8,))
    stats = grpo_trainer.train_step(prompts, targets)
print(f'  GRPO Loss: {stats["loss"]:.4f}')
print(f'  GRPO Rule Reward: {stats["reward"]:.3f}')
print(f'  GRPO KL: {stats["kl"]:.4f}')
print(f'  GRPO Adv Std: {stats["advantage_std"]:.3f}')
print('  GRPO => OK')

print()
print('=' * 70)
print('第四章全部通过! RLHF(三阶段) -> DPO(简化版) -> GRPO(DeepSeek-R1)')
print('=' * 70)
