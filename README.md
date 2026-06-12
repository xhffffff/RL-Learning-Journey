
<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/framework-PyTorch-red.svg" alt="PyTorch"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/chapter-4-blue?label=总章节" alt="4 chapters">
  <img src="https://img.shields.io/badge/algorithms-16-brightgreen?label=涵盖算法" alt="16 algorithms">
  <img src="https://img.shields.io/badge/focus-RLHF%20%7C%20DPO%20%7C%20GRPO-orange?label=重点" alt="Focus: RLHF">
</p>

<h1 align="center">从 Q-Learning 到 DAPO：强化学习算法演进全记录</h1>

---

## 📖 目录

- [🎯 项目介绍](#-项目介绍)
- [🧠 核心理论速览](#-核心理论速览)
- [📂 项目结构](#-项目结构)
- [📘 章节概览](#-章节概览)
  - [第一章：基础算法](#第一章基础算法-1980s-2013)
  - [第二章：深度强化学习](#第二章深度强化学习-2013-2017)
  - [第三章：高级算法](#第三章高级算法-2015-2023)
  - [第四章：LLM 时代的强化学习 ⭐](#第四章llm时代的强化学习-2022至今-⭐)
- [📊 算法对比表](#-算法对比表)
- [🗺️ 学习路线图](#️-学习路线图)
- [⚡ 快速开始](#-快速开始)
- [📚 延伸阅读](#-延伸阅读)
- [🙏 致谢](#-致谢)

---

## 🎯 项目介绍

**RL-Learning-Journey** 是一个按照强化学习发展脉络组织的教学项目，覆盖从 **Q-Learning (1989)** 到 **DAPO (2025)** 共 16 个核心算法。每个算法都配有**可运行的最小实现**，帮助你"看到"算法的工作原理，而不仅仅是"读懂"公式。

### 🎓 目标受众

本项目专为具备以下基础、希望进入 **LLM 对齐（Alignment）领域** 的学习者设计：

| 预备知识 | 所需程度 | 自检问题 |
|---------|---------|----------|
| **深度学习** | 熟悉神经网络、反向传播、PyTorch | 你能用 PyTorch 写一个训练循环吗？ |
| **高等数学/线性代数** | 理解梯度、概率分布、矩阵运算 | 你知道 softmax 和 cross-entropy 的关系吗？ |
| **大语言模型（LLM）** | 了解 Transformer、SFT、Prompt | 你理解"下一个 token 预测"的训练目标吗？ |

> **如果你符合以上条件，并希望理解 RLHF / DPO / GRPO 背后的强化学习原理——本项目就是为你准备的。**

### ✨ 项目特色

- **发展脉络驱动**：按时间线组织，呈现算法之间的演进关系
- **最小可运行实现**：每个算法 <500 行核心代码，专注于算法精髓
- **统一代码接口**：公共模块 [`common/`](common/) 提供 `BaseAgent` 与 `ReplayBuffer`，降低认知负担
- **LLM RL 重点突出**：第四章深入讲解 RLHF → DPO → GRPO 的完整链路，包含"为什么需要RL"动机分析、算法对比、决策流程图、FAQ
- **中文注释 & 文档**：所有代码和文档使用中文，降低语言障碍
- **理论与实践分离**：每个算法明确标注"简化说明"，让你清楚哪些是算法本质、哪些是工程实现

### 🔬 为什么 LLM 工程师需要学 RL？

```
SFT（监督微调） → 模仿人类，上限是人类水平
RLHF  /  DPO    → 超越人类标注，优化"人类偏好"本身
GRPO            → 无需偏好标注，模型通过自我对比进化
```

理解这些技术的核心——强化学习——将使你能够：

- ✅ 理解 ChatGPT、Claude、DeepSeek-R1 的训练原理
- ✅ 参与 LLM 对齐（Alignment）相关研发
- ✅ 读懂 RLHF / DPO / GRPO 的前沿论文
- ✅ 为未来的 AI 训练范式做好准备

---

## 🧠 核心理论速览

在深入算法之前，先快速浏览强化学习的核心概念。以下概念贯穿整个项目，**第四章的 RLHF 本质上就是 PPO 在语言模型上的应用**。

### 1. 马尔可夫决策过程 (MDP)

强化学习的基本数学模型。一个 MDP 由五元组定义：$\langle S, A, P, R, \gamma \rangle$

| 符号 | 含义 | 示例 |
|------|------|------|
| $S$ | 状态空间 | 棋盘局面 |
| $A$ | 动作空间 | 可走的位置 |
| $P(s'\mid s,a)$ | 状态转移概率 | 落子后的新局面 |
| $R(s,a)$ | 奖励函数 | 赢棋 +1 |
| $\gamma \in [0,1]$ | 折扣因子 | 平衡短期与长期回报 |

**累积回报（Return）**：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 2. 价值函数与 Bellman 方程

**状态价值函数**：在状态 $s$ 下，遵循策略 $\pi$ 的期望回报

$$V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

**动作价值函数**：在状态 $s$ 采取动作 $a$ 后，遵循策略 $\pi$ 的期望回报

$$Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$

**Bellman 期望方程**（RL 最重要的公式之一）：

$$Q_\pi(s,a) = \mathbb{E}_{s'}\left[R(s,a) + \gamma \sum_{a'} \pi(a'|s') Q_\pi(s',a')\right]$$

> 💻 代码入口：Q-Learning 实现见 [`chapter_01_foundation/01_q_learning/q_learning.py`](chapter_01_foundation/01_q_learning/q_learning.py) 中的 `QLearningAgent.update()` 方法。

### 3. TD 学习（时序差分学习）

TD 是 Q-Learning 和 SARSA 的理论基础，核心思想是**从当前估计中学习**：

**TD(0) 更新**：

$$V(S_t) \leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

其中 $\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD Target}} - \underbrace{V(S_t)}_{\text{Current Estimate}}$ 称为 **TD 误差 (TD Error)**。

**Q-Learning (Off-Policy)**：

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

> 💻 代码入口：见 [`chapter_01_foundation/01_q_learning/q_learning.py:84`](chapter_01_foundation/01_q_learning/q_learning.py)

**SARSA (On-Policy)**：

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma Q(s',a') - Q(s,a)\right]$$

> 💻 代码入口：见 [`chapter_01_foundation/02_sarsa/sarsa.py:59`](chapter_01_foundation/02_sarsa/sarsa.py)

### 4. 策略梯度定理 (Policy Gradient Theorem)

直接优化策略参数 $\theta$ 以最大化期望回报：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \Psi_t\right]$$

其中 $\Psi_t$ 可以是：
- **总回报** $G_t$（REINFORCE）
- **优势函数** $A(s,a) = Q(s,a) - V(s)$（A2C/PPO）
- **TD 误差** $\delta_t$（Actor-Critic）

> 💻 代码入口：REINFORCE 实现见 [`chapter_01_foundation/03_reinforce/reinforce.py:98`](chapter_01_foundation/03_reinforce/reinforce.py)

### 5. 从经典 RL 到 RLHF 的桥梁

RLHF 的目标函数本质上是**带 KL 约束的 PPO**：

$$L_{\text{RLHF}} = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[r_\phi(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)\right)\right]$$

| 经典 RL | RLHF 对应 |
|---------|-----------|
| 环境 → 状态 $s$ | 用户 → 提示词 $x$ |
| 策略 → 动作 $a$ | 语言模型 → 响应 $y$ |
| 奖励函数 $R(s,a)$ | 奖励模型 $r_\phi(x,y)$ |
| 状态转移 $P(s'\mid s,a)$ | 确定性（下一个请求） |
| PPO 裁剪 | PPO 裁剪（防止失控） |
| 无约束 | **KL 惩罚**（防止遗忘） |

> 💻 代码入口：见 [`chapter_04_llm_rl/01_rlhf/rlhf.py:218`](chapter_04_llm_rl/01_rlhf/rlhf.py)

---

## 📂 项目结构

```
RL-Learning-Journey/
├── common/                         🔧 公共基础设施
│   ├── base_agent.py               # Agent 抽象基类
│   └── replay_buffer.py            # 经验回放缓冲区
│
├── environments/                   🎮 自定义 Gym 环境
│   └── custom_envs.py
│
├── chapter_01_foundation/          📗 第一章：经典 RL 基础
│   ├── 01_q_learning/              # Q-Learning (Off-Policy TD)
│   ├── 02_sarsa/                   # SARSA (On-Policy TD)
│   └── 03_reinforce/               # REINFORCE (MC Policy Gradient)
│
├── chapter_02_deep_rl/             📙 第二章：深度强化学习
│   ├── 01_dqn/                     # DQN (Nature 2015)
│   ├── 02_double_dqn/              # Double DQN
│   ├── 03_dueling_dqn/             # Dueling DQN
│   ├── 04_a3c/                     # Advantage Actor-Critic (A2C)
│   ├── 05_ppo/                     # Proximal Policy Optimization
│   └── 06_sac/                     # Soft Actor-Critic
│
├── chapter_03_advanced/            📕 第三章：前沿 RL 算法
│   ├── 01_trpo/                    # Trust Region Policy Optimization
│   ├── 02_td3/                     # Twin Delayed DDPG
│   ├── 03_muzero/                  # MuZero (Model-based + Planning)
│   └── 04_dreamer/                 # Dreamer (World Models)
│
├── chapter_04_llm_rl/              ⭐ 第四章：LLM 时代的强化学习
│   ├── 01_rlhf/                    # RLHF (InstructGPT)
│   ├── 02_dpo/                     # Direct Preference Optimization
│   ├── 03_grpo/                    # Group Relative Policy Optimization
│   ├── 04_rloo/                    # REINFORCE Leave-One-Out (Cohere, 2024)
│   └── 05_dapo/                    # DAPO (ByteDance+Tsinghua, 2025)
│
├── test_chapter01.py ~ test_chapter04.py   🧪 单元测试
├── requirements.txt                📦 依赖
├── setup.py                        ⚙️ 包配置
└── LICENSE                         📄 MIT License
```

---

## 📘 章节概览

### 第一章：基础算法 (1980s-2013)

> 🎯 **学习目标**：理解 RL 的核心机制——值函数、TD 学习、策略梯度

在深度神经网络尚未流行的时代，RL 算法通过**表格**或**线性函数**进行学习。本章是理解一切现代 RL 算法的**必修基础**。

| # | 算法 | 年份 | 类型 | 核心思想 | 代码 |
|---|------|------|------|---------|------|
| 1 | **Q-Learning** | 1989 | Off-Policy TD | 学最优 Q 值，与行为策略无关 | [`q_learning.py`](chapter_01_foundation/01_q_learning/q_learning.py) |
| 2 | **SARSA** | 1994 | On-Policy TD | 学当前策略的 Q 值，探索更安全 | [`sarsa.py`](chapter_01_foundation/02_sarsa/sarsa.py) |
| 3 | **REINFORCE** | 1992 | Policy Gradient | 蒙特卡洛采样直接优化策略 | [`reinforce.py`](chapter_01_foundation/03_reinforce/reinforce.py) |

**关键公式**：

Q-Learning 更新规则：
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

REINFORCE 梯度：
$$\nabla_\theta J \approx \frac{1}{N} \sum_{i} \sum_{t} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot G_t^i$$

> 📖 详细文档：[`chapter_01_foundation/README.md`](chapter_01_foundation/README.md)（如果存在）

---

### 第二章：深度强化学习 (2013-2017)

> 🎯 **学习目标**：掌握神经网络 + RL 的组合拳，理解 DQN → PPO 的技术演进

深度学习赋予了 RL 处理**高维状态空间**（图像、连续控制）的能力，开启了 DeepRL 时代。**PPO 是 RLHF 的核心引擎**，务必彻底理解。

| # | 算法 | 年份 | 类型 | 核心思想 | 代码 |
|---|------|------|------|---------|------|
| 4 | **DQN** ⭐ | 2015 | Value-based | 经验回放 + 目标网络 = 稳定训练 | [`dqn.py`](chapter_02_deep_rl/01_dqn/dqn.py) |
| 5 | **Double DQN** | 2015 | Value-based | 解耦动作选择与评估，缓解过估计 | [`double_dqn.py`](chapter_02_deep_rl/02_double_dqn/double_dqn.py) |
| 6 | **Dueling DQN** | 2016 | Value-based | $Q = V + A$，分离状态价值与动作优势 | [`dueling_dqn.py`](chapter_02_deep_rl/03_dueling_dqn/dueling_dqn.py) |
| 7 | **A2C** | 2016 | Actor-Critic | 优势函数 $A(s,a)$ 降低方差 | [`a2c.py`](chapter_02_deep_rl/04_a3c/a2c.py) |
| 8 | **PPO** ⭐⭐ | 2017 | Policy Gradient | 裁剪目标函数，限制策略更新幅度 | [`ppo.py`](chapter_02_deep_rl/05_ppo/ppo.py) |
| 9 | **SAC** | 2018 | Actor-Critic | 最大熵框架，自动探索与利用平衡 | [`sac.py`](chapter_02_deep_rl/06_sac/sac.py) |

**PPO 裁剪目标**（RLHF 的理论基础）：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比。

> ⚠️ **与 RLHF 的关系**：OpenAI 的 InstructGPT 直接使用 PPO 进行 RLHF 训练。理解 PPO 的裁剪机制，是理解 RLHF 训练稳定性的关键。

---

### 第三章：高级算法 (2015-2023)

> 🎯 **学习目标**：了解 RL 的前沿方向——信任区域约束、世界模型、基于模型的规划

这些算法代表了 RL 能力的边界拓展。虽然不是 RLHF 的直接依赖，但理解它们有助于建立完整的 RL 知识体系。

💡 包含针对初学者的直觉解释和常见误区提示。详细文档：[`chapter_03_advanced/README.md`](chapter_03_advanced/README.md)

| # | 算法 | 年份 | 类型 | 核心思想 | 代码 |
|---|------|------|------|---------|------|
| 10 | **TRPO** | 2015 | Policy Gradient | KL 散度约束，保证策略单调提升 | [`trpo.py`](chapter_03_advanced/01_trpo/trpo.py) |
| 11 | **TD3** | 2018 | Actor-Critic | 双 Q 网络 + 延迟更新 + 目标平滑 | [`td3.py`](chapter_03_advanced/02_td3/td3.py) |
| 12 | **MuZero** | 2019 | Model-based | 无需环境规则，学习隐式模型 + MCTS 规划 | [`muzero.py`](chapter_03_advanced/03_muzero/muzero.py) |
| 13 | **Dreamer** | 2020 | Model-based | 在世界模型的隐空间中"做梦"学习策略 | [`dreamer.py`](chapter_03_advanced/04_dreamer/dreamer.py) |

**TRPO 约束优化**（PPO 的前身）：

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s,a)\right] \quad \text{s.t.} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta)] \leq \delta$$

> 💡 **演进线索**：TRPO（精确 KL 约束）→ PPO（近似裁剪）→ RLHF-PPO（裁剪 + KL 惩罚）→ GRPO（组内归一化 + 无 Critic）

---

### 第四章：LLM 时代的强化学习 (2022-至今) ⭐

> 🎯 **学习目标**：彻底弄懂 RLHF → DPO → GRPO → RLOO → DAPO 的技术演进，掌握 LLM 对齐的核心方法
>
> 🔥 **本章是本项目的核心重点。** 详细文档：[`chapter_04_llm_rl/README.md`](chapter_04_llm_rl/README.md)

详细文档包含：为什么需要 RL、SFT vs RLHF 对比、渐进式代码走读、DPO 拔河比喻、GRPO 推理机制、RLOO 无偏基线、DAPO 工程改进、算法决策流程图、FAQ 等。

#### 4.1 RLHF — 强化学习从人类反馈

**三阶段训练流程**：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1    │ → │   Stage 2    │ → │   Stage 3    │
│   SFT       │    │   RM 训练     │    │   PPO 微调    │
│ (监督微调)   │    │ (奖励建模)    │    │ (强化学习)    │
└─────────────┘    └──────────────┘    └──────────────┘
  模仿高质量回复       学习人类偏好         对齐偏好
```

**RLHF-PPO 目标函数**：

$$L = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[r_\phi(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)\right)\right]$$

- $r_\phi(x,y)$：奖励模型对生成响应 $y$ 的评分
- $\beta$：KL 惩罚系数（典型值 0.02-0.1），**防止模型"钻奖励模型的空子"**
- $\pi_{\text{ref}}$：SFT 模型作为参考锚点

> 💻 代码入口：RLHF 三阶段实现见 [`chapter_04_llm_rl/01_rlhf/rlhf.py`](chapter_04_llm_rl/01_rlhf/rlhf.py)

**核心论文**：*Training language models to follow instructions with human feedback* (Ouyang et al., NeurIPS 2022)

#### 4.2 DPO — 直接偏好优化

**核心洞察**：RLHF 的奖励模型在最优策略下有解析解，因此**可以跳过奖励模型，直接从偏好数据优化策略**。

**DPO 损失函数**：

$$L_{\text{DPO}} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

| 符号 | 含义 |
|------|------|
| $y_w$ | **Chosen**——人类偏好的回答 |
| $y_l$ | **Rejected**——人类不偏好的回答 |
| $\beta$ | 控制偏离参考模型的程度（典型值 0.1） |
| $\pi_{\text{ref}}$ | 参考模型（通常是 SFT 模型） |

**DPO vs RLHF 对比**：

| | RLHF (PPO) | DPO |
|---|-----------|-----|
| 奖励模型 | ✅ 需要 | ❌ 不需要 |
| 训练阶段 | 3 阶段 | 2 阶段 |
| 采样生成 | ✅ 需要（PPO 在线采样） | ❌ 不需要（离线数据） |
| 训练稳定性 | ⚠️ 需要调参 | ✅ 较稳定 |
| 计算开销 | 🔴 高 | 🟢 低 |

> 💻 代码入口：DPO 实现在 [`chapter_04_llm_rl/02_dpo/dpo.py:154`](chapter_04_llm_rl/02_dpo/dpo.py)

**核心论文**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., NeurIPS 2023)

#### 4.3 GRPO — 组内相对策略优化 ⭐ 最新

**DeepSeek-R1 的核心训练算法**。GRPO 最大的创新是**彻底抛弃了 Critic 网络（价值网络）**，使用组内奖励的相对位置来估计优势。

**GRPO 目标函数**：

$$L_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \min\left(r_i A_i, \ \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i\right)$$

其中优势函数通过组内归一化得到：

$$A_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

**GRPO 三大创新**：

```
1. 无 Critic 网络  →  节省 50% 以上显存
2. 组内归一化     →  无需学值函数，直接用奖励分布
3. 规则化奖励     →  无需训练奖励模型（准确率 + 格式）
```

> 💻 代码入口：GRPO 实现在 [`chapter_04_llm_rl/03_grpo/grpo.py:134`](chapter_04_llm_rl/03_grpo/grpo.py)

**核心论文**: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (DeepSeek-AI, 2025)

#### 4.4 RLOO — REINFORCE Leave-One-Out

**"PPO 对 RLHF 来说是大炮打蚊子"**。RLOO 回到最基础的 REINFORCE 算法，用一个简单的 Leave-One-Out 基线替代 Critic 网络。

**RLOO 优势函数**：

$$A_k = \frac{K}{K-1}\left(R_k - \frac{1}{K}\sum_{j=1}^{K} R_j\right)$$

**RLOO 损失**：

$$\mathcal{L}_{\text{RLOO}} = -\frac{1}{K} \sum_{k=1}^{K} A_k \cdot \log \pi_\theta(a_k|s) + \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$$

**三大优势**：
- 无 Critic 网络（节省 ~33% 显存）
- 无 PPO 裁剪（极简实现，核心代码 ~20 行）
- **无偏梯度估计**（LOO 基线在数学上是无偏的）

> 💻 代码入口: [`chapter_04_llm_rl/04_rloo/rloo.py`](chapter_04_llm_rl/04_rloo/rloo.py)

**核心论文**: *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (Ahmadian et al., Cohere For AI, 2024)

#### 4.5 DAPO — 解耦裁剪与动态采样策略优化

**"GRPO 的四大工程补丁"**。ByteDance Seed + 清华 AIR 团队发现 GRPO 在竞赛级推理上有四个致命缺陷，并逐一修复。

**DAPO 四大创新**：

| 创新 | 问题 | 解决方案 |
|------|------|----------|
| **Clip-Higher** | PPO 对称裁剪限制探索 | 非对称裁剪 clip(r, 1-ε_low, 1+ε_high) |
| **Dynamic Sampling** | 全对/全错样本无学习信号 | 过滤 std=0 的 prompt |
| **Token-Level Loss** | 样本级损失对长短序列不公 | 总 token 数归一的逐 token 损失 |
| **Overlong Shaping** | 硬截断导致长度崩塌 | 软惩罚 R·exp(-α·超出比例) |

**AIME 2024 成绩**: 50%（DeepSeek-R1 为 47%，仅需一半训练步数）

> 💻 代码入口: [`chapter_04_llm_rl/05_dapo/dapo.py`](chapter_04_llm_rl/05_dapo/dapo.py)

**核心论文**: *DAPO: An Open-Source LLM RL System at Scale* (ByteDance Seed + Tsinghua AIR, 2025)

---

## 📊 算法对比表

| 算法 | 年份 | 作者 | 类型 | 核心创新 | 应用场景 |
|------|------|------|------|---------|---------|
| **Q-Learning** | 1989 | Watkins | Off-Policy TD | Bootstrap + 最优 Q 值 | 离散控制、游戏 |
| **SARSA** | 1994 | Rummery | On-Policy TD | 学当前策略的 Q 值 | 安全关键场景 |
| **REINFORCE** | 1992 | Williams | MC Policy Gradient | 直接策略优化 | 连续控制基础 |
| **DQN** | 2015 | DeepMind | Value + 深度网络 | 经验回放 + 目标网络 | Atari 游戏 |
| **Double DQN** | 2015 | van Hasselt | Value + 深度网络 | 解耦选择与评估 | 减少过估计 |
| **Dueling DQN** | 2016 | Wang et al. | Value + 深度网络 | $Q = V + A$ 分解 | 状态识别 |
| **A2C** | 2016 | Mnih et al. | Actor-Critic | 优势函数降方差 | 通用 RL 任务 |
| **PPO** | 2017 | OpenAI | Actor-Critic | 裁剪目标函数 | **RLHF、机器人** |
| **SAC** | 2018 | Haarnoja | Actor-Critic | 最大熵探索 | 连续控制 |
| **TRPO** | 2015 | Schulman | Actor-Critic | KL 散度约束 | PPO 前身 |
| **TD3** | 2018 | Fujimoto | Actor-Critic | 双 Q + 延迟更新 | 连续控制 |
| **MuZero** | 2019 | DeepMind | Model-based | 隐式模型 + MCTS | 围棋、Atari |
| **Dreamer** | 2020 | Hafner | Model-based | 隐空间学习 | 机器人想象 |
| **RLHF** | 2022 | OpenAI | LLM + PPO | 三阶段人类对齐 | **ChatGPT** |
| **DPO** | 2023 | Stanford | LLM + 偏好 | 无需奖励模型 | **LLM 对齐** |
| **GRPO** | 2025 | DeepSeek | LLM + 组优化 | 无 Critic + 组内基线 | **DeepSeek-R1** |
| **RLOO** | 2024 | Cohere | LLM + REINFORCE | LOO 无偏基线 | 通用 RLHF |
| **DAPO** | 2025 | ByteDance+Tsinghua | LLM + 组优化 | 四大工程改进 | 竞赛级推理 |

---

## 🗺️ 学习路线图

```
1989 ─ Q-Learning ─────────────────────────────────────┐
1992 ─ REINFORCE ──────────────────────────────────────┤
1994 ─ SARSA ──────────────────────────────────────────┤
                                                         │  经典 RL 时代
0000 ──────────────────────────────────────────────────┤  (表格方法)
                                                         │
2015 ─ DQN ─────────────┐───────────────────────────────┘
2015 ─ Double DQN ──────┤
2016 ─ Dueling DQN ─────┤  深度 RL 崛起
2016 ─ A2C ─────────────┤  (神经网络 + RL)
2017 ─ PPO ─────────────┤── 成为 RLHF 核心引擎
2018 ─ SAC ─────────────┤
2018 ─ TD3 ─────────────┤
2019 ─ MuZero ──────────┤  基于模型
2020 ─ Dreamer ─────────┘
                                                         ┐
2022 ─ RLHF (InstructGPT) ─── PPO + KL 约束 + 奖励模型  │
                           ─── ChatGPT 训练方法          │
2023 ─ DPO ─────────────────── 跳过奖励模型，直接偏好   │  LLM 时代的 RL
                           ─── 更简单、更稳定            │  ⭐ 重点
2025 ─ GRPO ────────────────── 无 Critic + 组内归一化    │
                           ─── DeepSeek-R1 推理突破       │
2024 ─ RLOO ─────────────────── 纯 REINFORCE + LOO 基线  │
                           ─── 无偏梯度，极简实现          │
2025 ─ DAPO ────────────────── GRPO 四大工程改进            │
                           ─── AIME 50% 超越 DeepSeek-R1   │
                                                         ┘
        ← 从零开始 →       ← 经典掌握 →       ← 前沿对齐 →
        第一章              第二、三章           第四章
```

---

## ⚡ 快速开始

### 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | **3.10+** | 需要 match-case 等新语法 |
| PyTorch | **2.0+** | 推荐 2.1+，支持 compile |
| Gymnasium | **0.29+** | OpenAI Gym 的继任者 |
| CUDA | 11.8+（可选） | GPU 加速训练 |

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/xhffffff/RL-Learning-Journey.git
cd RL-Learning-Journey

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. （可选）以包形式安装，方便跨目录导入
pip install -e .
```

### 验证安装

安装完成后，运行以下命令验证环境：

```bash
# 检查 Python 版本
python --version  # 应该显示 3.10 或更高

# 检查 PyTorch 是否正确安装
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 检查 Gymnasium
python -c "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')"

# 运行基础测试
python test_chapter01.py
```

### 运行第一个算法

```bash
# 入门：Q-Learning 在 FrozenLake 上
cd chapter_01_foundation/01_q_learning
python train.py
```

### 各章运行命令

<details>
<summary><b>第一章 · 基础算法</b></summary>

```bash
# Q-Learning
cd chapter_01_foundation/01_q_learning
python train.py

# SARSA
cd chapter_01_foundation/02_sarsa
python train.py

# REINFORCE
cd chapter_01_foundation/03_reinforce
python train.py

# REINFORCE 对比（带/不带 baseline）
cd chapter_01_foundation/03_reinforce
python visualize_baseline.py
```
</details>

<details>
<summary><b>第二章 · 深度强化学习</b></summary>

```bash
# DQN
cd chapter_02_deep_rl/01_dqn
python train.py --env CartPole-v1 --episodes 500

# Double DQN
cd chapter_02_deep_rl/02_double_dqn
python train.py --env CartPole-v1 --episodes 500

# Dueling DQN
cd chapter_02_deep_rl/03_dueling_dqn
python train.py --env CartPole-v1 --episodes 500

# A2C
cd chapter_02_deep_rl/04_a3c
python train.py --env CartPole-v1 --episodes 1000

# PPO
cd chapter_02_deep_rl/05_ppo
python train.py --env CartPole-v1 --episodes 1000

# SAC
cd chapter_02_deep_rl/06_sac
python train.py --env Pendulum-v1 --episodes 200
```
</details>

<details>
<summary><b>第三章 · 高级算法</b></summary>

```bash
# TRPO
cd chapter_03_advanced/01_trpo
python train.py --env CartPole-v1

# TD3
cd chapter_03_advanced/02_td3
python train.py --env Pendulum-v1 --episodes 200
```
</details>

<details open>
<summary><b>第四章 · LLM 时代的 RL ⭐</b></summary>

```bash
# RLHF 演示
cd chapter_04_llm_rl/01_rlhf
python rlhf.py
# 或
python train.py --demo

# DPO 演示
cd chapter_04_llm_rl/02_dpo
python dpo.py
# 或
python train.py --demo

# GRPO 演示
cd chapter_04_llm_rl/03_grpo
python grpo.py
# 或
python train.py --demo

# RLOO 演示
cd chapter_04_llm_rl/04_rloo
python rloo.py

# DAPO 演示
cd chapter_04_llm_rl/05_dapo
python dapo.py
```
</details>

### 运行测试

```bash
# 测试所有章节
python test_chapter01.py
python test_chapter02.py
python test_chapter03.py
python test_chapter04.py
```

---

## 📚 延伸阅读

### 📄 必读论文（按学习顺序）

| 序号 | 论文 | 链接 | 关键贡献 |
|------|------|------|---------|
| 1 | Playing Atari with Deep RL (DQN) | [arXiv 1312.5602](https://arxiv.org/abs/1312.5602) | 经验回放 + 目标网络 |
| 2 | Human-level control through DRL (Nature DQN) | [Nature 2015](https://www.nature.com/articles/nature14236) | DQN 登 Nature |
| 3 | Asynchronous Methods for Deep RL (A3C) | [arXiv 1602.01783](https://arxiv.org/abs/1602.01783) | Actor-Critic 并行化 |
| 4 | Proximal Policy Optimization (PPO) | [arXiv 1707.06347](https://arxiv.org/abs/1707.06347) | 裁剪目标、RLHF 引擎 |
| 5 | Soft Actor-Critic (SAC) | [arXiv 1801.01290](https://arxiv.org/abs/1801.01290) | 最大熵 RL |
| 6 | **Training language models to follow instructions (RLHF)** | [arXiv 2203.02155](https://arxiv.org/abs/2203.02155) | InstructGPT / ChatGPT 方法 |
| 7 | **Direct Preference Optimization (DPO)** | [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) | 无需奖励模型的偏好学习 |
| 8 | **DeepSeek-R1 (GRPO)** | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | 推理强化学习突破 |
| 9 | Mastering Atari, Go, Chess (MuZero) | [Nature 2020](https://www.nature.com/articles/s41586-020-03051-4) | 基于模型的 RL |
| 10 | DreamerV3 | [arXiv 2301.04104](https://arxiv.org/abs/2301.04104) | 通用世界模型 |
| 11 | **Back to Basics: REINFORCE Style Optimization (RLOO)** | [arXiv 2402.14740](https://arxiv.org/abs/2402.14740) | LOO 基线替代 Critic |
| 12 | **DAPO: An Open-Source LLM RL System** | [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) | GRPO 的四大工程改进 |

### 📖 推荐教材

| 资源 | 适合阶段 | 说明 |
|------|---------|------|
| [**Sutton & Barto - Reinforcement Learning: An Introduction**](http://incompleteideas.net/book/the-book-2nd.html) | 第一、二章 | RL 圣经，免费在线 |
| [**Spinning Up in Deep RL (OpenAI)**](https://spinningup.openai.com/) | 第二、三章 | 最佳 Deep RL 入门教程 |
| [**HuggingFace Deep RL Course**](https://huggingface.co/learn/deep-rl-course/) | 第一章入门 | 互动式 RL 教学 |
| [**Lil'Log - Policy Gradient algorithms**](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) | 第二、三章 | 策略梯度算法综述 |
| [**RLHF 详解 (HuggingFace)**](https://huggingface.co/blog/rlhf) | 第四章 | RLHF 图解说明 |
| [**Andrej Karpathy - DeepSeek-R1**](https://karpathy.ai/deepseek-r1.html) | 第四章 | GRPO 通俗解读 |

### 🎥 视频资源

| 资源 | 说明 |
|------|------|
| [**David Silver RL Course**](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) | AlphaGo 之父的 RL 经典课程 |
| [**CS 285: Deep RL (UC Berkeley)**](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine 的深度 RL 课程 |
| [**Stanford CS224N: RLHF Lecture**](https://web.stanford.edu/class/cs224n/) | NLP 视角下的 RLHF 讲解 |

### 🔗 推荐 GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [openai/spinningup](https://github.com/openai/spinningup) | OpenAI 官方 Deep RL 教育库 |
| [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) | PPO / A2C 经典实现 |
| [huggingface/trl](https://github.com/huggingface/trl) | Transformer RL——RLHF / DPO 生产级实现 |
| [DeepSeek-AI/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | GRPO 官方实现 |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | RLHF 开源框架 |

---

## 🙏 致谢

本项目深受以下优秀项目和资源的启发，在此表示感谢：

- **Richard Sutton & Andrew Barto** —《Reinforcement Learning: An Introduction》，RL 领域的奠基之作
- **OpenAI Spinning Up** — 开创性的 Deep RL 教育资源，奠定了"从代码学 RL"的范式
- **HuggingFace TRL Team** — 将 RLHF 工程化并开源，推动了整个社区的发展
- **John Schulman** — PPO 算法的设计者，间接塑造了 ChatGPT 的训练方式
- **DeepSeek-AI** — 开源 DeepSeek-R1 与 GRPO，推动推理 RL 的前沿
- **Stanford CRFM** (Rafailov et al.) — 提出 DPO，极大简化了偏好对齐的流程
- 所有为本项目贡献代码和反馈的学习者

---

<p align="center">
  <sub>Made with ❤️ for the RL community | Licensed under <a href="LICENSE">MIT</a></sub>
  <br>
  <sub>⭐ 如果本项目对你有帮助，请给一个 Star！</sub>
</p>
