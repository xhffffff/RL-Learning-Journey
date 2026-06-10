# 第四章：大模型时代的强化学习 — LLM 对齐 (2022-至今)

> 🎯 **本章是本课程最核心的章节，目标是 RLHF 入门学者。**
>
> 本章系统讲解强化学习在大型语言模型（LLM）对齐中的应用：从经典的 RLHF 三阶段，到简洁优雅的 DPO，从 DeepSeek-R1 背后的 GRPO，到"返璞归真"的 RLOO，再到"百尺竿头更进一步"的 DAPO。所有实现均使用简化的 MLP 模型来演示核心算法逻辑，帮助你聚焦于理解公式背后的直觉。

---

## 0. 为什么需要强化学习？—— 监督微调的局限性

很多 LLM 工程师会困惑：**"SFT（监督微调）不是已经能对齐了吗，为什么还要搞 RL？"** 这是一个非常好的问题。让我们用一个具体例子来回答。

### 场景：教模型回答"如何学习编程？"

**SFT 的做法（行为克隆）**：
> 收集 1000 条"高质量回答"，让模型逐 token 模仿。
> 问题：模型只是学会了**复刻训练数据的模式**，但无法区分"好回答"和"更好回答"之间的细微差异。

**RL 的做法（偏好优化）**：
> 收集 1000 组"对比数据"（回答A > 回答B），让模型学会**最大化人类偏好**。
> 优势：模型开始在抽象的"偏好空间"中搜索，可能发现训练数据中没有出现过的、但人类更喜欢的回答方式。

### SFT vs RLHF 的本质区别

| 维度 | SFT（监督微调） | RLHF / DPO |
|------|---------------|------------|
| **学习信号** | "这是正确答案，照着学" | "A比B好，朝A的方向走" |
| **目标** | 最大化 $\log P(y \mid x)$ | 最大化**人类偏好分数** |
| **天花板** | 标注数据质量（人写得多好，模型就多好） | **超越标注者**——模型可能发现比标注更好的回答 |
| **优化空间** | 狭窄——只在正样本附近 | 广阔——在整个"偏好梯度"上搜索 |
| **典型失败模式** | 过度自信、重复模板、缺乏多样性 | 奖励黑客、语言退化（通过 KL 约束缓解） |

### 一个具体例子

假设 prompt 是"解释强化学习"：

| 回答 | SFT 的处理 | RLHF/DPO 的处理 |
|------|-----------|----------------|
| A："RL是ML的一个分支..."（标准但平淡） | 正样本，鼓励 | 如果是 chosen，相对 rejected 提权 |
| B："就像训练狗：坐下→零食→学会坐下，RL也一样"（生动但不够严谨） | 若不在训练集则不学 | 人类若偏好 B > A，模型朝"生动"方向移动 |
| C："RL就是强化学习"（太简短） | 若在训练集则也学 | 很可能是 rejected，降低此类概率 |

> 💡 **核心洞察**：SFT 告诉你"抄这份作业"，RLHF 告诉你"大家更喜欢这种风格，你朝这个方向努力"。前者的上限是作业质量，后者的上限是**模型创造力 + 偏好信号的组合**。

### 为什么直接评分不行？

你可能会想："那我直接让人类给每个回答打分（1-5分），然后做回归不就行了？" 这样做有三个致命问题：

1. **评分不一致**：不同人对"4分"的理解完全不同，偏好比较（pairwise）要稳定得多
2. **分数不校准**：人类打分时存在"宽严差异"，导致信号噪声极大
3. **绝对分数难以优化**：模型需要知道"往哪个方向改"能让分数提高，Pairwise 比较天然给出梯度方向

---

## 1. LLM 对齐问题概述

### 从游戏 RL 到语言 RL 的范式转移

传统强化学习解决的是 **游戏型任务**（Atari、围棋、机器人控制）——奖励函数明确、状态空间可定义、目标单一。而 LLM 训练面对的是一个全新的问题：

| 维度 | 游戏 RL | LLM 对齐 RL |
|------|---------|-------------|
| 奖励信号 | 环境自动给出（得分、存活） | **人类偏好**（模糊、主观、不一致） |
| 动作空间 | 离散/连续动作 | **token 序列**（$10^4 \sim 10^5$ 词汇量） |
| 状态空间 | 物理状态 | **语义空间**（不可直接观测） |
| 目标 | 最大化累计奖励 | 生成**有用、无害、诚实（HHH）**的文本 |
| 约束 | 物理约束 | **不能遗忘预训练知识**（KL 约束） |
| 关键挑战 | 探索-利用平衡 | **奖励信号极度稀疏**（只能在生成完成后评估） |

在经典 RL 中，智能体与 MDP 交互：$S \times A \to R, S'$。在 LLM 对齐中：

- **状态** $s_t$ = 到目前为止生成的 token 序列 $(x, y_1, ..., y_{t-1})$
- **动作** $a_t$ = 下一个 token $y_t \in \mathcal{V}$
- **奖励** $R$ = 人类（或奖励模型）对**整个生成文本** $y$ 的评分
- **策略** $\pi_\theta$ = 语言模型本身

核心挑战：**奖励信号稀疏且主观**——只有完整生成后才能评估"回答得好不好"，而且"多好算好"因人而异。

### 五类对齐算法概览

| 算法 | 核心理念 | 需要几个模型 | 复杂度 |
|------|----------|-------------|--------|
| **RLHF** | 先训练奖励模型，再用 PPO 优化 | 4 个（Policy + Reference + Critic + Reward Model） | 🔴 高 |
| **DPO** | 直接从偏好数据优化，数学上消去奖励模型 | 2 个（Policy + Reference） | 🟢 低 |
| **RLOO** | 纯 REINFORCE + Leave-One-Out 基线，无 Critic 无裁剪 | 2 个（Policy + Reference） | 🟢 低 |
| **GRPO** | 用组内相对奖励替代 Critic + 奖励模型 | 2 个（Policy + Reference） | 🟡 中 |
| **DAPO** | GRPO 的四大工程改进，专攻竞赛级推理 | 2 个（Policy + Reference） | 🟡 中 |

---

## 2. RLHF — Reinforcement Learning from Human Feedback

### 论文信息
- **核心论文**: *Training language models to follow instructions with human feedback* (OpenAI, 2022)
- **关键论文**: *Learning to Summarize from Human Feedback* (OpenAI, 2020)
- **链接**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

### 概述

RLHF 是 ChatGPT 背后的核心技术，包含三个顺序阶段。它成功地将人类偏好注入到原本只会"预测下一个 token"的语言模型中。

```
阶段1 (SFT)         阶段2 (RM Training)      阶段3 (PPO Fine-Tuning)
┌──────────┐        ┌──────────┐            ┌──────────┐
│预训练 LM  │  ──→  │SFT 模型  │  ──→       │对齐模型   │
│          │        │          │            │          │
│ "原始"   │  高质量 │ "礼貌"   │  人类偏好  │ "有用"    │
│          │  指令数据│          │  比较数据  │          │
└──────────┘        └──────────┘            └──────────┘
                            ↓
                     ┌──────────┐
                     │奖励模型 RM│
                     └──────────┘
```

---

### 🔍 渐进式代码走读：从高层到底层理解 RLHF

在深入各阶段细节之前，先以**鸟瞰视角**理解 RLHF 的完整训练循环：

```
高层流程（你只需要理解这个）：
1. SFT：用标准交叉熵损失微调 → 模型学会"像人类那样回答"
2. RM训练：用比较数据训练奖励模型 → RM学会"判断哪个回答更好"
3. PPO微调：RM打分 + KL约束 → 模型学会"生成讨人类喜欢的回答"

中层流程（看代码时的导航）：
  sft_training()  →  返回 sft_model（保存为 π_ref）
  reward_model_training()  →  返回 rm
  RLHFTrainer(policy=sft_model, ref_policy=sft_model, reward_model=rm)
    └── train_step() 循环  →  最终对齐模型

底层细节（关键数学）：
  RM损失: -log σ(r_chosen - r_rejected)
  RLHF目标: max [r_φ(x,y) - β·KL(π_θ || π_ref)]
  KL计算: Σ π_θ·log(π_θ / π_ref)
```

---

### 阶段 1：SFT — Supervised Fine-Tuning（监督微调）

#### 目标

让预训练语言模型学会"按照指令格式给出高质量回答"。本质上是一个 **行为克隆（Behavior Cloning）** 过程。

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{SFT}}}\left[\log \pi_\theta(y \mid x)\right]$$

其中 $(x, y)$ 是 **指令-高质量回答** 对（由人类专家标注）。

#### 数据格式

```
指令 x:  "请用三句话解释什么是强化学习"
回答 y:  "强化学习是机器学习的一个分支，智能体通过与环境交互、
         获得奖励或惩罚信号来学习最优策略。其核心概念包括状态、
         动作、奖励和策略。常见算法有Q-Learning和策略梯度方法。"
```

#### 📝 代码走读（从外到内）

**第1层 — 调用入口**（`rlhf.py` 主训练流程）：
```python
# 顶层：调用SFT训练函数
sft_model = sft_training(base_model, prompt_response_pairs)
```

**第2层 — 训练循环**（`rlhf.py` sft_training 函数，第 61-86 行）：
```python
def sft_training(model, prompt_response_pairs, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for prompt, target_response in prompt_response_pairs:
            optimizer.zero_grad()
            logits = model(prompt)                    # 模型前向
            loss = F.cross_entropy(logits, target_response)  # 标准分类损失
            loss.backward()
            optimizer.step()
    return model
```

**第3层 — 损失函数本质**：
这个交叉熵损失就是让模型对"正确回答"的预测概率最大化。如果 prompt 是"1+1=?"，target 是"2"，那么 loss 驱动模型给"2"更高的 softmax 概率。

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 SFT |
|------|--------|----------|
| 模型 | MLP 分类器（~100K 参数） | GPT-3/LLaMA（175B+ 参数） |
| 数据量 | 几十条模拟指令 | 数万-数十万条人工标注指令 |
| 序列处理 | 单 token 分类 | 自回归生成：对每个 token 计算交叉熵并求和 |
| 损失 | `CrossEntropyLoss(logits, label)` | $\sum_t \text{CrossEntropy}(\pi_\theta(\cdot \mid x, y_{<t}), y_t)$ |

SFT 后，保存模型权重作为后续阶段的**参考模型** $\pi_{\text{ref}}$：
```python
ref_model = SmallLanguageModel(input_dim, hidden_dim, vocab_size)
ref_model.load_state_dict(sft_model.state_dict())  # 冻结快照
```

---

### 阶段 2：Reward Model Training（奖励模型训练）

#### 核心问题

RL 需要一个奖励函数，但在 LLM 场景中，我们无法为每个可能的回答编写代码来评分——"回答得多好"是无法用规则衡量的。

**解决方案**：让人类对同一 prompt 下的两个回答进行比较排序，然后用这些比较数据训练一个**奖励模型**来模拟人类的偏好判断。

#### Bradley-Terry 偏好模型

假设对于 prompt $x$，有两个回答 $y_w$（更好的/chosen）和 $y_l$（更差的/rejected），人类偏好概率用 Bradley-Terry 模型建模：

$$P(y_w \succ y_l \mid x) = \frac{\exp\left(r_\phi(x, y_w)\right)}{\exp\left(r_\phi(x, y_w)\right) + \exp\left(r_\phi(x, y_l)\right)} = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)$$

其中 $r_\phi(x, y)$ 是奖励模型为回答 $y$ 打出的标量分数，$\sigma$ 是 sigmoid 函数。

直观理解：**两个回答的奖励分数差距越大，模型越确信更好的那个确实是更好的**。

#### 奖励模型训练损失

通过对偏好数据做最大似然估计，得到 RM 的损失函数：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}_{\text{pref}}}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

最小化这个损失 = 让 chosen 的分数显著高于 rejected 的分数。

#### 📝 代码走读（从外到内）

**第1层 — RM 网络结构**（`rlhf.py` RewardModel 类）：
```python
class RewardModel(nn.Module):
    """输入: prompt+response 表示 → 输出: 标量分数"""
    def __init__(self, input_dim=128, hidden_dim=256):
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)            # 输出标量
        )
```

**第2层 — 训练循环**（`rlhf.py` reward_model_training 函数）：
```python
def reward_model_training(rm, preferences, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
    for epoch in range(epochs):
        for chosen, rejected in preferences:
            r_chosen = rm(chosen)            # 打分：好回答
            r_rejected = rm(rejected)        # 打分：差回答
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            loss.backward()
            optimizer.step()
```

**第3层 — 损失函数直觉**：
- 当 `r_chosen - r_rejected` 很大（正数） → sigmoid → 1 → `-log(1)` = 0，无惩罚 ✅
- 当 `r_chosen - r_rejected` 很小/负数 → sigmoid → ≈0 → `-log(ε)` 很大，强惩罚 ❌

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 RM 训练 |
|------|--------|-------------|
| 初始化 | 随机初始化 MLP | 从 SFT 模型初始化（共享 Transformer 主体，只换输出头） |
| 输入表示 | 单 token 的 one-hot 向量 | prompt + 完整 response 序列的嵌入 |
| 数据量 | 模拟的几十对 | 数十万对人工比较数据 |
| 评分粒度 | 整个 response 一个分数 | 可以是 token 级别的细粒度奖励 |

---

### 阶段 3：PPO Fine-Tuning（强化学习微调）

#### RLHF 的强化学习目标

有了奖励模型 $r_\phi$ 后，PPO 阶段的目标是在**不偏离参考模型太远**的前提下，最大化奖励：

$$\max_\theta \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot \mid x)}\left[r_\phi(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\right)\right]$$

其中：
- $r_\phi(x, y)$：奖励模型对生成的回答 $y$ 打出的分数
- $\beta$：KL 惩罚系数（控制偏离参考模型的程度），典型值 0.02-0.1
- $\pi_{\text{ref}}$：参考模型（SFT 后的模型，**冻结**），作为"不要偏离太远"的锚点
- $D_{KL}$：KL 散度，衡量新策略与参考策略的差异

#### 🔑 KL 惩罚为何至关重要

没有 KL 惩罚时，模型会迅速学会"欺骗"奖励模型：

1. **奖励黑客（Reward Hacking）**：生成语法不通但对齐关键词的文本来骗取高分
2. **知识遗忘**：为追求高奖励，遗忘预训练学到的语言能力
3. **模式崩溃（Mode Collapse）**：反复生成同一种"安全但无聊"的回答

> 💡 直觉类比：奖励模型像一个"不完美的老师"，KL 惩罚是一条"安全带"——你被鼓励讨好老师，但不能放飞自我。

#### PPO 裁剪目标

PPO 的裁剪机制用于稳定训练，核心是限制策略更新的幅度。设重要性采样比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

- 当 $r_t > 1+\epsilon$ 且 $\hat{A}_t > 0$：裁剪阻止策略变得过于贪婪
- 当 $r_t < 1-\epsilon$ 且 $\hat{A}_t < 0$：裁剪阻止策略变得过于保守

PPO 的核心直觉：**鼓励好的改变，但限制改变幅度**。

#### 📝 代码走读（从外到内）

**第1层 — RLHFTrainer 初始化**（`rlhf.py` RLHFTrainer 类）：
```python
class RLHFTrainer:
    def __init__(self, policy, ref_policy, reward_model, beta=0.1):
        self.policy = policy         # 可训练的策略模型
        self.ref_policy = ref_policy # 冻结的SFT模型（用于KL约束）
        self.reward_model = reward_model  # 冻结的奖励模型
        self.beta = beta             # KL惩罚系数
```

**第2层 — 训练步**（`rlhf.py` train_step 方法）：
```python
def train_step(self, prompts):
    # Step 1: 从当前策略采样响应
    logits = self.policy(prompts)
    dist = torch.distributions.Categorical(logits=logits)
    responses = dist.sample()

    # Step 2: 奖励模型打分
    rm_reward = self.reward_model(response_one_hot)

    # Step 3: 计算KL惩罚（防止偏离SFT模型）
    kl_penalty = self.compute_kl_penalty(prompts, logits)

    # Step 4: 总奖励 = RM分数 - β × KL惩罚
    total_reward = rm_reward.mean() - self.beta * kl_penalty

    # Step 5: 策略梯度（简化REINFORCE）
    log_probs = dist.log_prob(responses)
    policy_loss = -(log_probs * total_reward.detach()).mean()
```

**第3层 — KL 惩罚计算**（`rlhf.py` compute_kl_penalty 方法）：
```python
def compute_kl_penalty(self, prompt, response_logits):
    with torch.no_grad():
        ref_logits = self.ref_policy(prompt)
    policy_probs = F.softmax(response_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    # KL(policy || ref) = Σ policy_probs · log(policy_probs / ref_probs)
    kl = (policy_probs * (torch.log(policy_probs + 1e-8)
          - torch.log(ref_probs + 1e-8))).sum(dim=-1)
    return kl.mean()
```

### 🔬 公式与代码一一对应

下表展示了 RLHF 三阶段中每个数学公式与 `rlhf.py` 中代码的精确对应关系：

| 公式 | 数学表达 | 代码位置 | 代码片段 |
|------|----------|----------|----------|
| **SFT 损失** | $$\mathcal{L}_{\text{SFT}} = -\mathbb{E}\left[\log \pi_\theta(y \mid x)\right]$$ | `sft_training()` (第78行) | `loss = F.cross_entropy(logits, target_response)` |
| **RM 损失** | $$\mathcal{L}_{\text{RM}} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$ | `reward_model_training()` (第142行) | `loss = -F.logsigmoid(r_chosen - r_rejected).mean()` |
| **RLHF 总奖励** | $$\mathcal{R}_{\text{total}} = \bar{r}_{\phi} - \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$$ | `RLHFTrainer.train_step()` (第239行) | `total_reward = rm_reward.mean() - self.beta * kl_penalty` |
| **KL 散度** | $$D_{KL} = \sum_j \pi_\theta(j) \cdot \log\frac{\pi_\theta(j)}{\pi_{\text{ref}}(j)}$$ | `RLHFTrainer.compute_kl_penalty()` (第215行) | `kl = (policy_probs * (torch.log(policy_probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1)` |
| **REINFORCE 损失** | $$\mathcal{L}_{\text{RL}} = -\mathbb{E}\left[\log\pi_\theta(y|x) \cdot \mathcal{R}_{\text{total}}\right]$$ | `RLHFTrainer.train_step()` (第243行) | `policy_loss = -(log_probs * total_reward.detach()).mean()` |

**公式逐行解读:**

1. **SFT 损失** $\mathcal{L}_{\text{SFT}} = -\log \pi_\theta(y|x)$：
   - 第一步：`logits = model(prompt)` — 模型前向传播得到 logits（第77行）
   - 第二步：`F.cross_entropy(logits, target_response)` — PyTorch 内部执行 `-log(softmax(logits)[target])`，等价于最大化正确答案的对数概率
   - 直觉：SFT 本质上就是**监督分类**——强迫模型给"正确答案"最高的 softmax 概率

2. **RM 损失** $-\log\sigma(r_{\text{chosen}} - r_{\text{rejected}})$：
   - 第一步：`r_chosen = rm(chosen)` — 奖励模型对好回答打分（第139行）
   - 第二步：`r_rejected = rm(rejected)` — 奖励模型对差回答打分（第140行）
   - 第三步：`-F.logsigmoid(r_chosen - r_rejected).mean()` — 当 $r_{\text{chosen}} \gg r_{\text{rejected}}$ 时，`logsigmoid(大正数) ≈ 0`，损失接近 0；当两者接近时损失很大
   - 直觉：RM 训练就是一个**二分类问题**——学会判断"哪个更好"

3. **KL 散度** $\sum \pi_\theta \log(\pi_\theta / \pi_{\text{ref}})$：
   - 第一步：`ref_logits = self.ref_policy(prompt)` — 冻结参考模型前向（第209行，`torch.no_grad` 下）
   - 第二步：`policy_probs = F.softmax(response_logits, dim=-1)` — 当前策略概率分布（第211行）
   - 第三步：`ref_probs = F.softmax(ref_logits, dim=-1)` — 参考策略概率分布（第212行）
   - 第四步：`policy_probs * (torch.log(policy_probs + 1e-8) - torch.log(ref_probs + 1e-8))` — 逐元素计算 $p \cdot \log(p/q)$，`.sum(-1)` 对词表维度求和（第215行）
   - 直觉：如果 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 分布完全相同，KL = 0；差异越大 KL 越大

4. **RLHF 总奖励 + REINFORCE 更新**：
   - 第一步：`rm_reward.mean()` — batch 内 RM 奖励的平均值（第239行）
   - 第二步：`self.beta * kl_penalty` — β 加权 KL 惩罚（第239行）
   - 第三步：`total_reward = rm_reward.mean() - self.beta * kl_penalty` — 净奖励 = 讨好 RM - 不要太飘
   - 第四步：`log_probs = dist.log_prob(responses)` — 采样动作的对数概率（第242行）
   - 第五步：`(log_probs * total_reward.detach())` — 优势加权的对数概率（`detach()` 阻止奖励通过梯度回传到 RM）
   - 第六步：取负 `.mean()` → 最大化期望奖励 = 最小化负期望奖励
   - 直觉：高总奖励的采样被强化（梯度推高其 log_prob），低总奖励的被抑制

> 📐 **RLHF 三阶段串联逻辑**：
> ```
> SFT:  minimize -log P(correct|prompt)         → 学会"像人类一样回答"
> RM:   minimize -log σ(r_chosen - r_rejected)  → 学会"判断哪个更好"  
> PPO:  maximize E[R_RM - β·KL]                 → 学会"生成讨喜的回答"
> ```
> 三个阶段**串行依赖**：PPO 依赖 RM 的评分和 SFT 的参考分布，缺一不可。

### 💻 关键代码速查

| # | 代码行 | 为什么关键 |
|---|--------|-----------|
| 1 | `loss = -F.logsigmoid(r_chosen - r_rejected).mean()` | **RM 训练核心**：这是整个 RLHF 偏好学习的数学根基，Bradley-Terry 模型的直接代码化 |
| 2 | `total_reward = rm_reward.mean() - self.beta * kl_penalty` | **RLHF 目标**：奖励最大化与 KL 约束的平衡点，β 控制"讨好 RM"与"保持清醒"的权衡 |
| 3 | `kl = (policy_probs * (torch.log(policy_probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1)` | **KL 惩罚实现**：这是防止奖励黑客的数学安全带，直接对应 $D_{KL}(P \parallel Q)$ 的离散形式 |

#### ⚠️ 简化说明（理论与实践差距）

| 简化项 | 本实现 | 真实 RLHF |
|--------|--------|-----------|
| 策略模型 | MLP 分类器 | GPT-3/LLaMA (175B+ 参数) |
| 序列建模 | 单 token 一步分类 | 完整自回归序列生成 |
| PPO 实现 | 简化 REINFORCE 损失（无裁剪） | 完整 PPO-CLIP + GAE + mini-batch |
| Critic 网络 | ❌ 无 | ✅ 完整价值网络 |
| RM 输入 | 单 token one-hot | prompt + 完整 response 嵌入 |
| KL 惩罚 | prompt 层面概率分布 KL | 逐 token 累加的序列级 KL |
| 训练规模 | 数百步，模拟数据 | 数十万步，真实人类偏好数据 |
| 硬件需求 | CPU 即可 | 数百-数千 GPU |

---

## 3. DPO — Direct Preference Optimization

### 论文信息
- **标题**: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- **作者**: Rafailov et al. (Stanford, 2023)
- **链接**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **荣誉**: NeurIPS 2023 Oral

### 核心洞察：隐式奖励

DPO 的重大发现是：**语言模型本身隐式地定义了一个奖励函数**。通过简单的数学推导，可以将 Bradley-Terry 偏好模型中的显式奖励函数 $r_\phi$ 替换为策略的对数比率：

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

这意味着：**我们不需要训练一个独立的奖励模型！** 策略模型 $\pi_\theta$ 相对于参考模型 $\pi_{\text{ref}}$ 的回答概率提升程度，天然衡量了"人类有多喜欢这个回答"。

### 📐 DPO 损失函数 — "拔河"比喻

DPO 损失函数：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

这个公式看起来复杂，但用**"拔河比赛"**的比喻就一目了然：

```
         π_θ(y_w|x)                        π_θ(y_l|x)
    log  ──────────    vs    log  ──────────
         π_ref(y_w|x)                     π_ref(y_l|x)

         ↑ "好回答"概率变化               ↑ "差回答"概率变化

DPO希望: 好回答的"概率提升"  >>  差回答的"概率提升"
```

具体来说：

| 情况 | 含义 | DPO 损失 |
|------|------|----------|
| `chosen_log_ratio` ≫ `rejected_log_ratio` | 模型更偏好好回答 | 损失 ≈ 0 ✅ |
| `chosen_log_ratio` ≈ `rejected_log_ratio` | 模型不区分好坏 | 损失 ≈ 0.693 |
| `chosen_log_ratio` ≪ `rejected_log_ratio` | 模型偏好了差回答！ | 损失 很大 ❌ |

> 💡 **拔河比喻总结**：有两个队伍——"好回答队"和"差回答队"。DPO 损失就是**拉大两队的得分差距**。当"好回答队"遥遥领先时（得分差很大），sigmoid 接近 1，loss 接近 0——模型已经学会了。当两队旗鼓相当时，loss 迫使模型在"好回答"上花更多概率质量。

### 📝 代码走读（从外到内）

**第1层 — DPO 类初始化**（`dpo.py` DPOTrainer 类）：
```python
class DPOTrainer:
    def __init__(self, policy, ref_model, beta=0.1):
        self.policy = policy      # 可训练的策略模型
        self.ref_model = ref_model  # 冻结的SFT参考模型
        self.beta = beta          # 控制偏离参考模型的程度
```

**第2层 — 核心损失函数**（`dpo.py` dpo_loss 方法）：
```python
def dpo_loss(self, batch):
    # 策略模型对 chosen/rejected 的对数概率
    policy_chosen_logps = self.policy.get_log_probs(batch.prompt, batch.chosen)
    policy_rejected_logps = self.policy.get_log_probs(batch.prompt, batch.rejected)

    # 参考模型对 chosen/rejected 的对数概率（冻结，无梯度）
    with torch.no_grad():
        ref_chosen_logps = self.ref_model.get_log_probs(batch.prompt, batch.chosen)
        ref_rejected_logps = self.ref_model.get_log_probs(batch.prompt, batch.rejected)

    # 计算对数比率（"拔河"的两端）
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

    # DPO损失：拉大 chosen 和 rejected 的比率差距
    loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()
```

**第3层 — 获取对数概率**（`dpo.py` get_log_probs 方法）：
```python
def get_log_probs(self, x, labels):
    logits = self.forward(x)
    return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)
```

### DPO vs RLHF 对比表

| 维度 | RLHF | DPO |
|------|------|-----|
| 需要奖励模型 | ✅ 需要单独训练 RM | ❌ 不需要（隐式奖励） |
| 训练阶段数 | 3 阶段（SFT → RM → PPO） | 2 阶段（SFT → DPO） |
| PPO 训练 | ✅ 需要（不稳定、超参敏感） | ❌ 不需要 |
| 人类偏好数据 | 比较数据（pairwise） | 比较数据（pairwise） |
| 奖励信号 | 显式（RM 评分） | 隐式（对数比率） |
| 训练稳定性 | 较差（RL + LM 双系统） | 较好（纯监督学习） |
| 计算开销 | 高（4 个模型 + RL 采样） | 低（2 个模型 + 直接梯度） |
| 在线/离线 | 通常在线（self-generated） | 通常离线（固定偏好数据） |
| 适用场景 | 大规模在线训练（ChatGPT） | 离线偏好优化（开源模型） |
| 典型使用者 | OpenAI, Anthropic | Meta (Llama 3), Mistral, 社区 |

### 🔬 公式与代码一一对应

下表展示了 DPO 中每个数学公式与 `dpo.py` 中代码的精确对应关系：

| 公式 | 数学表达 | 代码位置 | 代码片段 |
|------|----------|----------|----------|
| **隐式奖励** | $$r(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ | `DPOTrainer.dpo_loss()` (第134行) | `log_ratio_chosen = policy_chosen_logps - ref_chosen_logps` |
| **策略对数概率** | $$\log \pi_\theta(\text{label} \mid x)$$ | `DPOModel.get_log_probs()` (第70行) | `F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)` |
| **DPO 损失** | $$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta \left[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right]\right)$$ | `DPOTrainer.dpo_loss()` (第138行) | `loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()` |
| **"拔河"两端** | $$\Delta_{\text{chosen}} = \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)},\; \Delta_{\text{rejected}} = \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$$ | `DPOTrainer.dpo_loss()` (第134-135行) | `log_ratio_chosen = policy_chosen_logps - ref_chosen_logps` / `log_ratio_rejected = policy_rejected_logps - ref_rejected_logps` |
| **训练准确率** | $$\mathbf{1}\left[\Delta_{\text{chosen}} > \Delta_{\text{rejected}}\right]$$ | `DPOTrainer.dpo_loss()` (第142行) | `acc = (log_ratio_chosen > log_ratio_rejected).float().mean()` |

**公式逐行解读:**

1. **隐式奖励** $r(x,y) = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$：
   - 第一步：`policy_chosen_logps = self.policy.get_log_probs(batch.prompt, batch.chosen)` — 当前策略对 chosen 的对数概率（第125行）
   - 第二步：`ref_chosen_logps = self.ref_model.get_log_probs(batch.prompt, batch.chosen)` — 参考模型对 chosen 的对数概率，在 `torch.no_grad()` 下（第130行）
   - 第三步：`log_ratio_chosen = policy_chosen_logps - ref_chosen_logps` — 对数比率，其数学本质是 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$（第134行）
   - 直觉：如果策略相比参考模型**提升了** chosen 的概率 → `log_ratio_chosen > 0` → 隐式奖励为正（说明策略在这个回答上"进步"了）

2. **策略对数概率** $\log \pi_\theta(\text{label} \mid x)$：
   - 第一步：`logits = self.forward(x)` — 模型前向得到 logits（第69行）
   - 第二步：`F.log_softmax(logits, dim=-1)` — 对 logits 做 log-softmax，得到每个类别的对数概率（第70行）
   - 第三步：`.gather(1, labels.unsqueeze(-1)).squeeze(-1)` — 取出标签对应位置的对数概率值（第70行）
   - 直觉：这一步拿到的是 $\log P(\text{正确答案})$，越大越好

3. **DPO 损失** $\mathcal{L}_{\text{DPO}}$ — "拔河"的完整过程：
   - 第一步：计算 `log_ratio_chosen` 和 `log_ratio_rejected`（第134-135行）— 两队的"得分"
   - 第二步：`log_ratio_chosen - log_ratio_rejected` — 两队的"得分差"（第138行）
   - 第三步：`self.beta * (差值)` — β 控制分差的放大倍数（β 越大，小分差也会产生大损失，相当于"比赛更严格"）（第138行）
   - 第四步：`F.logsigmoid(β·分差)` — 分差越大 → sigmoid → 1 → logsigmoid → 0 → 损失小（第138行）
   - 第五步：取负 `.mean()` → 最小化损失 = 最大化分差
   - 直觉：DPO 就是一个**带温度控制的二分类**——分差够大就万事大吉，分差不够就罚

4. **"拔河"两端** — 为什么称为拔河：
   - 左端 `log_ratio_chosen`：策略对 chosen 的相对提升程度
   - 右端 `log_ratio_rejected`：策略对 rejected 的相对提升程度
   - DPO 希望左端 **远远大于** 右端：即策略应该大幅提升 chosen 的概率（相对参考），同时**不要提升** rejected 的概率
   - 如果 `log_ratio_chosen > log_ratio_rejected`：准确率 = 1，模型**已经在区分好坏**
   - 如果 `log_ratio_chosen ≤ log_ratio_rejected`：模型要么不区分、要么更偏好 rejected

> 📐 **DPO 的优雅之处**：
> ```
> RLHF:  SFT → RM训练 → PPO优化  (3个阶段，4个模型)
> DPO:   SFT → DPO直接优化       (2个阶段，2个模型)
> 
> DPO 消去了 RM，因为数学上 r(x,y) = β·log(π/π_ref)  
> 这意味着：你不需要训练一个奖励模型来给回答打分，
> 策略模型相对于参考模型的"概率提升"本身就是奖励信号。
> ```

### 💻 关键代码速查

| # | 代码行 | 为什么关键 |
|---|--------|-----------|
| 1 | `log_ratio_chosen = policy_chosen_logps - ref_chosen_logps` | **隐式奖励核心**：这一行是 DPO 数学洞察的代码化身——用概率比率替代了显式 RM |
| 2 | `loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()` | **DPO 损失**：整篇论文的结论收敛为这一行代码，"拔河"比喻的终极实现 |
| 3 | `acc = (log_ratio_chosen > log_ratio_rejected).float().mean()` | **训练监控**：这一行不是优化目标，但直接告诉你模型是否已学会区分好坏答案 |

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 DPO |
|------|--------|----------|
| 模型 | MLP 分类器 | Transformer（如 Llama 7B-70B） |
| 序列对数概率 | 单 token log_prob | $\sum_t \log \pi(y_t \mid x, y_{<t})$ |
| 偏好数据 | 随机生成的模拟 chosen/rejected | 数万条真实人工标注的 pairwise 数据 |
| 参考模型 | 简单 MLP | SFT 后的完整 Transformer |
| 训练规模 | 几百步 | 数万步，需要多 GPU |

---

## 4. GRPO — Group Relative Policy Optimization

### 论文信息
- **标题**: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*
- **作者**: DeepSeek-AI (2025)
- **链接**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

### 历史意义

GRPO 是 **DeepSeek-R1** 的核心训练算法，它成功地让 LLM 涌现出类人的推理能力（o1 级别的 Chain-of-Thought）。在 GRPO 出现之前，几乎没有 RL 算法能让纯语言模型自发学会"思考"。

### 🎯 为什么 DeepSeek 选择 GRPO 而不是 PPO？

这个问题是理解 GRPO 价值的核心。PPO 用于 RLHF 有三个实际痛点：

| 痛点 | PPO (标准RLHF) | GRPO 的解决方案 |
|------|---------------|----------------|
| **Critic 网络开销** | 需要训练一个与策略同样大的 Critic 网络 | 完全不需要 Critic，节省约 50% 显存 |
| **奖励模型训练** | 需要单独训练和部署 RM | 使用**规则化奖励**（准确率+格式检查） |
| **GAE 计算复杂** | 需要维护价值函数做 GAE 优势估计 | 用**组内相对优势**（同一 prompt 的 G 个回答互相比较） |

对于 DeepSeek 这样的推理任务（数学、编程），奖励天然是**可编程的**——答案对就是对，错就是错。这种情况下，训练一个神经奖励模型不仅是浪费，还可能引入噪声。GRPO 恰好利用了这一点。

### 核心创新：无 Critic 的强化学习

GRPO 最大的创新是**完全不需要 Critic 网络（价值函数）**。那它如何估计优势函数呢？

**答案：组内相对优势（Group Relative Advantage）**。

对于同一个 prompt，模型采样 $G$ 个不同的响应，计算每个响应的奖励，然后以整组的统计量作为归一化基准：

$$A_i = \frac{R_i - \text{mean}\left(\{R_1, R_2, ..., R_G\}\right)}{\text{std}\left(\{R_1, R_2, ..., R_G\}\right) + \varepsilon}$$

其中：
- $G$：组大小（Group Size），通常为 4-16
- $R_i$：第 $i$ 个响应的奖励（可以是规则奖励）
- $A_i$：标准化后的相对优势

**直觉**：如果你在同一次采样中比"平均水平"表现更好，$A_i > 0$，策略被鼓励；反之被抑制。这天然地避免了需要训练一个价值函数来估计基线。

### 🔮 GRPO 对推理能力的影响

GRPO 之所以能催生 DeepSeek-R1 的"推理涌现"，关键在于**组内对比 + 规则奖励**的组合效应：

1. **对比学习效应**：模型同时生成 G=8 个回答，通过互相对比，模型学会了"什么样的推理路径更有效"
2. **格式奖励催生 CoT**：如果规则奖励要求用 `  response  ` 格式，模型会自动学会先思考再回答
3. **自我修正的涌现**：当模型发现"先写过程再写答案"比"直接写答案"的组内排名更高时，它自发地学会了思维链
4. **冷启动路径**：即使初始模型推理能力很弱，经过充分探索后，组归一化会放大偶发的正确推理的优势信号

### GRPO 目标函数

GRPO 使用与 PPO 相同的裁剪目标，但优势来自组内相对奖励：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \min\left(r_i A_i,\;
\text{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i\right) + \beta \cdot D_{KL}\left(\pi_\theta \parallel \pi_{\text{ref}}\right)$$

其中：
- $r_i = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$：重要性采样比率
- $A_i$：组内相对优势（见上文）
- $\epsilon$：裁剪参数（通常 0.2）
- $\beta$：KL 惩罚系数（通常 0.04）

### 📝 代码走读（从外到内）

**第1层 — 组内相对优势计算**（`grpo.py` group_relative_advantage 方法）：
```python
def group_relative_advantage(self, rewards):
    """
    对每个prompt的G个回答，计算标准化优势
    A_i = (R_i - mean({R_1,...,R_G})) / (std({R_1,...,R_G}) + ε)
    """
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean_r) / (std_r + 1e-8)
```

**第2层 — 训练步**（`grpo.py` train_step 方法）：
```python
def train_step(self, prompts, targets):
    # 1. 对每个prompt采样G个响应，记录old_log_probs
    for _ in range(G):
        dist = torch.distributions.Categorical(logits=logits)
        response = dist.sample()
        log_prob = dist.log_prob(response)
        # ...存储...

    # 2. 堆叠为 (batch, G)，计算奖励
    responses = torch.stack(all_responses, dim=1)
    rewards = self.compute_reward(responses, targets)

    # 3. 组内相对优势（GRPO的核心创新）
    advantages = self.group_relative_advantage(rewards)

    # 4. PPO裁剪目标
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 5. KL惩罚 + 总损失
    kl = self.compute_kl(prompts, logits).mean()
    total_loss = policy_loss + self.beta * kl
```

**第3层 — 规则奖励**（`grpo.py` compute_reward 方法）：
```python
def compute_reward(self, responses, target):
    """简化的规则奖励：直接比较分类准确率"""
    accuracy = (responses == target.unsqueeze(1)).float()
    return accuracy  # 1.0 正确，0.0 错误
```

### 规则奖励（Rule-based Rewards）

GRPO 的另一大特色是使用**可编程的规则奖励**而不是训练一个神经奖励模型：

**DeepSeek-R1 使用的两类奖励**：

| 奖励类型 | 定义 | 示例 |
|----------|------|------|
| **准确率奖励** | 最终答案是否与标准答案匹配 | 数学题答案 = 42 → 1.0 |
| **格式奖励** | 输出是否包含要求的格式标记 | 是否用 `  think  ` 和 `  answer  ` 包裹 |

这种设计的精妙之处在于：**对于推理任务，正确答案天然可验证——不需要人来判断"这个推理好不好"**。

### GRPO vs PPO vs DPO vs RLOO — 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    PPO (标准RLHF) — 4个模型                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐         │
│  │ Policy  │  │Reference│  │ Critic  │  │Reward Model │         │
│  │  (训练)  │  │ (冻结)   │  │ (训练)  │  │  (冻结)     │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────┘         │
│  GAE计算优势 │ KL约束参考  │ 值函数估计  │ RM提供奖励               │
└──────────────────────────────────────────────────────────────────┘
                              vs
┌──────────────────────────────────────────────────────────────────┐
│                      DPO — 2个模型                                │
│  ┌─────────┐  ┌─────────┐                                        │
│  │ Policy  │  │Reference│   隐式奖励: r = β·log(π/π_ref)         │
│  │  (训练)  │  │ (冻结)   │   直接梯度优化，无RL采样               │
│  └─────────┘  └─────────┘                                        │
│  纯监督学习 │ 无需RM │ 无需Critic │ 离线数据                       │
└──────────────────────────────────────────────────────────────────┘
                              vs
┌──────────────────────────────────────────────────────────────────┐
│                     RLOO — 2个模型                                │
│  ┌─────────┐  ┌─────────┐                                        │
│  │ Policy  │  │Reference│   LOO基线: A_k = K/(K-1)·(R_k - μ)    │
│  │  (训练)  │  │ (冻结)   │   纯REINFORCE，无裁剪无GAE             │
│  └─────────┘  └─────────┘                                        │
│  无需Critic │ 无需RM │ 在线采样K=2~4个回答 │ 结构最简单             │
└──────────────────────────────────────────────────────────────────┘
                              vs
┌──────────────────────────────────────────────────────────────────┐
│                     GRPO — 2个模型                                │
│  ┌─────────┐  ┌─────────┐                                        │
│  │ Policy  │  │Reference│   组内相对优势: A_i = (R_i-μ)/σ        │
│  │  (训练)  │  │ (冻结)   │   规则奖励: 准确率 + 格式              │
│  └─────────┘  └─────────┘                                        │
│  无需Critic │ 无需RM │ 组内归一化 │ 在线采样 G=8 个回答             │
└──────────────────────────────────────────────────────────────────┘
                              vs
┌──────────────────────────────────────────────────────────────────┐
│                     DAPO — 2个模型                                │
│  ┌─────────┐  ┌─────────┐                                        │
│  │ Policy  │  │Reference│   四大改进: Decoupled CLIP + Dynamic    │
│  │  (训练)  │  │ (冻结)   │   Sampling + Token-Level + 长度惩罚    │
│  └─────────┘  └─────────┘                                        │
│  无需Critic │ 无需RM │ 组内归一化 │ 在线采样 G 个回答              │
└──────────────────────────────────────────────────────────────────┘
```

### 🔬 公式与代码一一对应

下表展示了 GRPO 中每个数学公式与 `grpo.py` 中代码的精确对应关系：

| 公式 | 数学表达 | 代码位置 | 代码片段 |
|------|----------|----------|----------|
| **规则奖励** | $$R_i = \mathbf{1}[\text{response}_i == \text{target}]$$ | `GRPOTrainer.compute_reward()` (第120行) | `accuracy = (responses == target.unsqueeze(1)).float()` |
| **组内优势** | $$A_i = \frac{R_i - \mu_G}{\sigma_G + \varepsilon}$$ | `GRPOTrainer.group_relative_advantage()` (第104-106行) | `(rewards - mean_r) / (std_r + 1e-8)` |
| **重要性比率** | $$r_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$ | `GRPOTrainer.train_step()` (第184行) | `ratio = (new_log_probs - old_log_probs).exp()` |
| **PPO 裁剪目标** | $$\min\left(r_i A_i,\; \text{clip}(r_i, 1-\varepsilon, 1+\varepsilon) \cdot A_i\right)$$ | `GRPOTrainer.train_step()` (第187-189行) | `surr1 = ratio * advantages` / `surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages` / `-torch.min(surr1, surr2).mean()` |
| **KL 惩罚** | $$D_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) = \sum_j \pi_\theta(j) \log\frac{\pi_\theta(j)}{\pi_{\text{ref}}(j)}$$ | `GRPOTrainer.compute_kl()` (第128-132行) | `kl = (p * (torch.log(p + 1e-8) - torch.log(ref_p + 1e-8))).sum(dim=-1)` |
| **总损失** | $$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum \min(r_i A_i, \text{clip}(r_i)A_i) + \beta \cdot D_{KL}$$ | `GRPOTrainer.train_step()` (第194行) | `total_loss = policy_loss + self.beta * kl` |

**公式逐行解读:**

1. **规则奖励** $R_i = \mathbf{1}[\text{response}_i == \text{target}]$：
   - `responses` 形状 `(batch, G)`，`target.unsqueeze(1)` 形状 `(batch, 1)`
   - 广播比较：每个 response 与 target 逐元素比较 → 正确返回 1.0，错误返回 0.0
   - 直觉：GRPO 的奖励是**非黑即白的**——答案对就是对，错就是错。这正是推理任务的优势所在：奖励天然可编程

2. **组内优势** $A_i = \frac{R_i - \mu_G}{\sigma_G + \varepsilon}$：
   - 第一步：`mean_r = rewards.mean(dim=-1, keepdim=True)` — 沿 G 维度求均值 μ（第104行）
   - 第二步：`std_r = rewards.std(dim=-1, keepdim=True)` — 沿 G 维度求标准差 σ（第105行）
   - 第三步：`(rewards - mean_r) / (std_r + 1e-8)` — z-score 标准化，`1e-8` 防止除零（第106行）
   - 直觉：如果某次采样的奖励高于组内平均水平 → $A_i > 0$ → 策略被鼓励；低于平均 → $A_i < 0$ → 被抑制
   - **关键**：归一化去除了奖励的绝对量纲，强制让组内好/坏回答形成对抗

3. **重要性比率** $r_i = \exp(\log\pi_\theta^{\text{new}} - \log\pi_\theta^{\text{old}})$：
   - `new_log_probs`：当前策略对历史采样的对数概率（第182行）
   - `old_log_probs`：采样时存储的旧策略对数概率（第169行）
   - `(new_log_probs - old_log_probs).exp()`：对数域差值取指数 = 概率比率（第184行）
   - 直觉：$r_i > 1$ 意味着新策略比旧策略**更喜欢**这个响应；$r_i < 1$ 则相反

4. **PPO 裁剪目标** $\min(rA, \text{clip}(r)A)$：
   - `surr1 = ratio * advantages`：未裁剪的原始目标（第187行）
   - `surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages`：将比率限制在 $[1-ε, 1+ε]$ 内（第188行）
   - `-torch.min(surr1, surr2).mean()`：取两者的最小值（悲观估计）（第189行）
   - **裁剪的两个方向**：
     - 当 $A>0$（好回答）且 $r>1+ε$：裁剪阻止策略**过度自信**（防止 catastrophe）
     - 当 $A<0$（差回答）且 $r<1-ε$：裁剪阻止策略**过度惩罚**（防止 collapse）
   - 直觉：PPO 的裁剪是一个**安全阀**——鼓励进步，但限制每一步的改变幅度

5. **GRPO 训练流程概览**：
   ```
   ① 采样 G 个响应 → ② 规则奖励打分 → ③ 组内 z-score 归一化 → 
   ④ 计算新旧比率 → ⑤ PPO 裁剪 + KL  → ⑥ 梯度更新
   ```
   相比于标准 PPO/RLHF，GRPO 砍掉了 Critic 网络和 RM 训练，用**组内统计量**和**规则奖励**替代。

### 💻 关键代码速查

| # | 代码行 | 为什么关键 |
|---|--------|-----------|
| 1 | `(rewards - mean_r) / (std_r + 1e-8)` | **GRPO 的灵魂**：组内 z-score 归一化替代了 Critic 网络和 GAE，是"不需要 Critic"这一核心主张的直接体现 |
| 2 | `surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages` | **PPO 裁剪**：防止单步更新过大导致的训练不稳定，是所有 PPO 变体的通用安全保障 |
| 3 | `accuracy = (responses == target.unsqueeze(1)).float()` | **规则奖励**：GRPO 之所以在推理任务上成功的关键——奖励天然可验证，无需训练 RM |

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 GRPO（DeepSeek-R1） |
|------|--------|--------------------------|
| 模型 | MLP 分类器 | DeepSeek-V3-Base（671B MoE） |
| 序列生成 | 单 token 分类 | 自回归序列生成（可能数万 token） |
| 组采样 G | 重复采样（实际同分布） | 真正的多样性采样（不同输出序列） |
| 规则奖励 | 简单的准确率匹配 | 准确率 + 格式 + 语言一致性等多维度 |
| 训练规模 | 几百步 | 数千 GPU 小时 |
| 推理涌现 | 不适用（任务太简单） | 自发产生 CoT、反思、自我修正 |

---

## 5. RLOO — REINFORCE Leave-One-Out

### 论文信息
- **标题**: *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*
- **作者**: Arash Ahmadian, Chris Cremer, Matthias Gallé et al. (Cohere For AI, 2024)
- **链接**: [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
- **发表时间**: 2024 年 2 月

### 💡 一句话讲透 RLOO

**"PPO 对 RLHF 来说是大炮打蚊子——用最原始的 REINFORCE 算法，加上一个简单的 Leave-One-Out 基线，不仅更简单，效果还不输 PPO。"**

### 核心洞察：RLHF 不需要 PPO

在 2024 年之前，整个社区都把 PPO 视为 RLHF 的"标准配置"——Critic 网络、GAE 优势估计、重要性采样裁剪……这些组件在游戏 RL 中确实必要，但在 LLM 对齐场景中，它们可能是多余的。

Cohere For AI 团队做了一个**思想实验**：如果去掉 PPO 的所有花哨组件，回到最基础的策略梯度（REINFORCE），会发生什么？

答案令人震惊：**不仅没变差，反而更简洁、更稳定。**

### 🔬 从 REINFORCE 到 RLOO 的数学推导

#### REINFORCE 策略梯度基线

最基础的 REINFORCE 策略梯度为：

$$\nabla_\theta \mathcal{J} = \mathbb{E}_{y \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot R(y)\right]$$

其中 $R(y)$ 是生成的响应 $y$ 获得的奖励。REINFORCE 的问题在于**高方差**——奖励信号 $R(y)$ 的绝对值波动很大，导致梯度估计不稳定。

#### 引入基线（Baseline）

为了降低方差，经典做法是减去一个基线 $b$：

$$\nabla_\theta \mathcal{J} = \mathbb{E}_{y \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot (R(y) - b)\right]$$

只要 $b$ 不依赖于动作 $y$，减去它**不会改变梯度的期望**（无偏），但可以大幅降低方差。

传统的做法是用 **Critic 网络**（价值函数 $V(s)$）作为基线——这就是 PPO 的做法。但 Critic 本身是近似估计，**会引入偏差**。

#### 🎯 RLOO 的 LOO 基线：无偏且无需训练

对于同一个 prompt $x$，RLOO 采样 $K$ 个响应 $\{y_1, y_2, ..., y_K\}$，然后用 Leave-One-Out 方式计算每个响应的基线：

$$b_k = \frac{1}{K-1} \sum_{j \neq k} R(y_j)$$

即：**对于第 $k$ 个响应，基线是其余 $K-1$ 个响应的平均奖励。**

那么第 $k$ 个响应的**RLOO 优势**为：

$$A(s, a_k) = R(y_k) - \frac{1}{K-1} \sum_{j \neq k} R(y_j)
= \frac{K}{K-1} \left(R(y_k) - \frac{1}{K} \sum_{j=1}^{K} R(y_j)\right)$$

展开理解：
- 括号内是 $R(y_k)$ 相对于整组均值的偏差
- $\frac{K}{K-1}$ 是 LOO 校正系数（因为除去自身的均值需要放大）

#### 完整的 RLOO 损失函数

$$\mathcal{L}_{\text{RLOO}} = -\frac{1}{K} \sum_{k=1}^{K} \left[\frac{1}{|y_k|} \sum_{t=1}^{|y_k|} \log \pi_\theta(y_{k,t} \mid x, y_{k,<t}) \cdot A_k\right] + \beta \cdot D_{KL}\left(\pi_\theta \parallel \pi_{\text{ref}}\right)$$

其中：
- $A_k$ 是上文定义的 LOO 优势（detach 后使用，不参与梯度传播）
- $\frac{1}{|y_k|} \sum_t$ 表示在 token 级别平均（Token-Level Loss）
- $\beta \cdot D_{KL}$ 是 KL 惩罚项
- 注意：RLOO **没有裁剪机制**，直接用 REINFORCE 的原始形式

### 🆚 为什么 LOO 基线是无偏的？——RLOO vs PPO 的本质差异

这是理解 RLOO 价值的核心问题。我们对比三种基线方法：

| 基线方法 | 来源 | 偏差 | 方差 | 额外成本 |
|----------|------|------|------|----------|
| **无基线** (REINFORCE) | $b=0$ | ✅ 无偏 | 🔴 极高 | 无 |
| **Critic 基线** (PPO) | $b=V(s)$ 神经网络估计 | ❌ 有偏（近似误差） | 🟡 中等 | 训练一个大型 Critic |
| **LOO 基线** (RLOO) | $b = \frac{1}{K-1} \sum_{j \neq k} R(y_j)$ | ✅ **无偏** | 🟢 低 | 无（只需 K≥2） |

**LOO 基线无偏的数学证明**（关键推导）：

因为基线 $b_k = \frac{1}{K-1} \sum_{j \neq k} R(y_j)$ 是通过采样其余 $K-1$ 个响应计算的，而这些响应与 $y_k$ 独立同分布，所以：

$$\mathbb{E}[b_k] = \mathbb{E}[R(y)]$$

但是**关键在于**：$b_k$ **不依赖于** $y_k$（它只依赖于 $j \neq k$ 的响应），因此：

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(y_k|x) \cdot b_k\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(y_k|x)\right] \cdot \mathbb{E}[b_k] = 0 \cdot \mathbb{E}[R] = 0$$

所以减去 LOO 基线不会改变梯度的期望，同时因为 $R(y_k)$ 和 $\frac{1}{K-1}\sum_{j \neq k}R(y_j)$ 高度正相关（都来自同一 prompt），方差大幅降低。

> 💡 **直觉**：Critic 基线像一个"需要不断校准的预测器"，而 LOO 基线像"让同考场其他考生一起帮你估分"——同学的实力接近、不需要额外评委、而且绝对公正（无偏）。

### 🔑 RLOO 的关键创新总结

| 组件 | PPO (标准RLHF) | RLOO | 说明 |
|------|---------------|------|------|
| **Critic 网络** | ✅ 必须 | ❌ 不需要 | 节省 ~50% 显存和训练成本 |
| **GAE 优势估计** | ✅ 必须 | ❌ 不需要 | LOO 直接给出无偏优势 |
| **PPO 裁剪** | ✅ 必须 | ❌ 不需要 | REINFORCE 原始形式 |
| **重要性采样** | ✅ 必须 | ❌ 不需要 | 同一批内直接采样 |
| **基线来源** | Critic 近似 | 组内统计量 | LOO：无偏 vs Critic：有偏 |
| **模型数量** | 4 | 2 | Policy + Reference 仅此而已 |

### 拔河比喻：RLOO 的直觉理解

想象一个班级里的 $K$ 个学生（$K=4$）参加同一场考试：

```
        学生A     学生B     学生C     学生D
        得分92    得分78    得分88    得分72
                                     
RLOO 对 A 说："你看其他三人平均是 (78+88+72)/3 ≈ 79.3，你考了 92，
               你比平均水平高出 12.7 分，干得好！继续朝这个方向努力。"

RLOO 对 D 说："其他三人平均是 (92+78+88)/3 = 86，你只考了 72，
               你比平均水平低了 14 分，需要反思并改变答题策略。"
```

这就是 RLOO 的核心思想：
- 每个人与其他所有人比较（Leave-One-Out）
- 比较基准是**同行考试的真实同伴**，不是某个"预测器"的估值
- 比平均高 → 正优势 → 强化；比平均低 → 负优势 → 抑制
- 没有外在评委（Critic），也没有复杂的打分规则（PPO clipping）——纯靠同伴竞争

### 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $K$ | 2 ~ 4 | 每组采样响应数。$K=2$ 已足够，$K=4$ 更稳定 |
| $\beta$ | 0.01 ~ 0.1 | KL 惩罚系数，与 PPO 一致 |
| 学习率 | $10^{-5} \sim 10^{-6}$ | 标准 LLM 微调学习率 |
| 采样温度 | 0.7 ~ 1.0 | 控制生成多样性 |

> 💡 **为什么 $K=2$ 就足够？** 论文实验表明：$K=2$ 的 RLOO 已经在 InstructGPT 级别任务上匹敌 PPO。$K=4$ 可以进一步降低方差，但边际收益递减。这与 GRPO 需要 $G=8$ 形成对比——RLOO 的 LOO 基线天然比 GRPO 的均值-标准差归一化更样本高效。

### 采用情况

| 框架 | 支持情况 |
|------|----------|
| **HuggingFace TRL** | ✅ `RLOOTrainer` 类，与 PPO 并列的官方支持 |
| **OpenRLHF** | ✅ 支持 RLOO 作为可选算法 |
| **Cohere Command-R** | ✅ 系 RLOO 论文作者团队，Command-R 模型系列使用 RLOO 训练 |
| **Unsloth** | ✅ 社区已适配 RLOO 训练 |

### 📝 代码走读（从外到内）

**第1层 — RLOO 核心：LOO 优势计算**（`rloo.py` LOO 优势函数）：
```python
def loo_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO的核心: Leave-One-Out 优势
    对同一prompt的K个响应，计算:
    A_k = R_k - mean(R_{j≠k}) = K/(K-1) · (R_k - mean(R))
    
    参数:
        rewards: (batch, K) 每个响应的奖励
    返回:
        advantages: (batch, K) LOO优势
    """
    K = rewards.shape[-1]
    mean_all = rewards.mean(dim=-1, keepdim=True)      # (batch, 1)
    # A_k = K/(K-1) · (R_k - mean_all)
    advantages = (K / (K - 1)) * (rewards - mean_all)  # (batch, K)
    return advantages
```

**第2层 — 训练步**（`rloo.py` train_step 方法）：
```python
def train_step(self, prompts):
    """
    RLOO训练步 —— 结构极简，只做三件事：
    1. 采样K个响应
    2. 用LOO计算优势
    3. REINFORCE策略梯度更新
    """
    K = self.K  # 每prompt采样K个响应 (通常K=2~4)
    
    # ---- 阶段1: 采样 ----
    self.policy.eval()
    all_responses = []
    all_rewards = []
    all_log_probs = []
    
    with torch.no_grad():
        logits = self.policy(prompts)
        for _ in range(K):
            dist = torch.distributions.Categorical(logits=logits)
            response = dist.sample()
            log_prob = dist.log_prob(response)
            reward = self.compute_reward(response)  # RM or rule
            
            all_responses.append(response)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
    
    # 堆叠为 (batch, K)
    responses = torch.stack(all_responses, dim=1)
    old_log_probs = torch.stack(all_log_probs, dim=1)
    rewards = torch.stack(all_rewards, dim=1)
    
    # ---- 阶段2: LOO 优势（RLOO的核心创新）----
    advantages = self.loo_advantage(rewards)
    
    # ---- 阶段3: REINFORCE 更新（纯策略梯度，无裁剪！）----
    self.policy.train()
    self.optimizer.zero_grad()
    
    logits = self.policy(prompts)
    dist = torch.distributions.Categorical(logits=logits.unsqueeze(1))
    new_log_probs = dist.log_prob(responses)
    
    # REINFORCE 损失（无重要性采样，无裁剪）
    policy_loss = -(new_log_probs * advantages.detach()).mean()
    
    # KL 惩罚
    kl = self.compute_kl(prompts, logits).mean()
    
    total_loss = policy_loss + self.beta * kl
    total_loss.backward()
    self.optimizer.step()
```

**第3层 — LOO 优势的向量化实现**：
```python
# 等价理解: 逐个计算每个样本的LOO基线
# 对于第k个样本: baseline_k = mean(rewards_except_k)
# advantage_k = reward_k - baseline_k
#
# 数学上等价于: A_k = K/(K-1) · (reward_k - mean_all)
# 下面是向量化版本:
def loo_advantage_vectorized(rewards, K):
    mean_all = rewards.mean()               # 全组均值
    # 对每个k: A_k = K/(K-1) · (R_k - μ)
    return (K / (K - 1)) * (rewards - mean_all)
```

### 🔬 公式与代码一一对应

下表展示了 RLOO 中每个数学公式与 `rloo.py` 中代码的精确对应关系：

| 公式 | 数学表达 | 代码位置 | 代码片段 |
|------|----------|----------|----------|
| **LOO 优势** | $$A_k = \frac{K}{K-1}\left(R_k - \frac{1}{K}\sum_{j=1}^{K} R_j\right)$$ | `RLOOTrainer.leave_one_out_advantage()` (第122-124行) | `(K / (K - 1)) * (rewards - mean_r)` |
| **规则奖励** | $$R_k = \mathbf{1}[\text{response}_k == \text{target}]$$ | `RLOOTrainer.compute_reward()` (第136行) | `accuracy = (responses == target.unsqueeze(1)).float()` |
| **REINFORCE 损失** | $$\mathcal{L}_{\text{REINFORCE}} = -\frac{1}{K}\sum_{k=1}^{K} \log\pi_\theta(y_k \mid x) \cdot A_k$$ | `RLOOTrainer.train_step()` (第200行) | `policy_loss = -(new_log_probs * advantages).mean()` |
| **KL 惩罚** | $$D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$$ | `RLOOTrainer.compute_kl()` (第147行) | `kl = (p * (torch.log(p + 1e-8) - torch.log(ref_p + 1e-8))).sum(dim=-1)` |
| **总损失** | $$\mathcal{L}_{\text{RLOO}} = \mathcal{L}_{\text{REINFORCE}} + \beta \cdot D_{KL}$$ | `RLOOTrainer.train_step()` (第205行) | `total_loss = policy_loss + self.beta * kl` |

**公式逐行解读:**

1. **LOO 优势** $A_k = \frac{K}{K-1}(R_k - \mu)$：
   - 第一步：`K = rewards.shape[-1]` — 获取采样响应数 K（第122行）
   - 第二步：`mean_r = rewards.mean(dim=-1, keepdim=True)` — 组内均值 $\mu = \frac{1}{K}\sum R_j$（第123行）
   - 第三步：`(K / (K - 1)) * (rewards - mean_r)` — LOO 校正后优势（第124行）
   - **为什么是 $\frac{K}{K-1}$？** 因为 Leave-One-Out 排除了自身，其余 $K-1$ 个的均值天然需要放大因子 $\frac{K}{K-1}$ 来恢复到全组尺度
   - **等价理解**：$A_k = R_k - \frac{1}{K-1}\sum_{j \neq k} R_j$（排己均值基线），数学上等价于上述单行实现
   - 直觉：你不需要 Critic 网络来"预测"你的价值——让同 prompt 的 $K$ 个并列采样互相当基线

2. **REINFORCE 损失** $-\log\pi_\theta(y_k \mid x) \cdot A_k$：
   - 第一步：使用 `torch.no_grad()` 下采样 $K$ 次，保存 `old_log_probs` 和 `rewards`（第171-179行）
   - 第二步：重新前向得到 `new_log_probs`（第197行）
   - 第三步：`-(new_log_probs * advantages).mean()` — **纯 REINFORCE，无重要性采样，无裁剪**（第200行）
   - **关键差异 vs GRPO**：RLOO 不使用 `ratio = exp(new - old)`，不需要 PPO 裁剪——直接对 log_prob 做加权
   - 直觉：这是策略梯度最原始的形态——"你做得好 → 增加这个动作的概率"，没有花哨的修正

3. **RLOO vs GRPO 裁剪策略的本质分歧**：
   ```
   RLOO:  policy_loss = -(new_log_probs * advantages).mean()    ← 纯 REINFORCE
   GRPO:  ratio = exp(new_log_probs - old_log_probs)            ← 重要性采样
          surr1 = ratio * advantages                            
          surr2 = clip(ratio, 1-ε, 1+ε) * advantages            ← PPO 裁剪
          loss = -min(surr1, surr2).mean()
   ```
   RLOO 的哲学：**数学上的无偏性比工程上的稳定性更重要**。LOO 基线已经足够降低方差，不需要额外的裁剪。

4. **LOO 基线无偏性验证**（代码中的数学背景，注释于第117-120行）：
   - 基线 $b_k = \frac{1}{K-1}\sum_{j \neq k} R_j$ 不依赖于 $y_k$（只依赖其他采样）
   - 因此 $\mathbb{E}[\nabla_\theta \log\pi_\theta(y_k) \cdot b_k] = \mathbb{E}[\nabla_\theta \log\pi_\theta] \cdot \mathbb{E}[b_k] = 0$
   - 减去基线不改变梯度期望 → **无偏**；同时因 $R_k$ 与 $b_k$ 正相关 → **降方差**
   - 对比 Critic：$V(s)$ 是近似器 → 有偏差，且偏差会累积

> 📐 **RLOO 的极简主义**：
> ```
> PPO/RLHF:  Policy + Ref + Critic + RM  →  4个模型，GAE + Clip + mini-batch
> GRPO:      Policy + Ref                →  2个模型，Clip + z-score
> RLOO:      Policy + Ref                →  2个模型，纯 REINFORCE + LOO 基线
> 
> RLOO 的核心赌注：在 LLM 对齐场景中，采样 K=4 个响应互相做基线
> 就足以替代 Critic 网络的"价值预测"——而且无偏、更简洁。
> ```

### 💻 关键代码速查

| # | 代码行 | 为什么关键 |
|---|--------|-----------|
| 1 | `(K / (K - 1)) * (rewards - mean_r)` | **RLOO 的灵魂**：一行代码实现 LOO 优势估计，替代了整个 Critic 网络 + GAE |
| 2 | `policy_loss = -(new_log_probs * advantages).mean()` | **纯 REINFORCE**：RLOO 最激进的设计选择——无重要性采样、无裁剪，回归最原始的梯度 |
| 3 | `total_loss = policy_loss + self.beta * kl` | **KL 锚定**：即使没有 PPO 裁剪，KL 惩罚依然不可少——它是最简 RLOO 中唯一防止发散的约束 |

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 RLOO |
|------|--------|-----------|
| 模型 | MLP 分类器 | Transformer（如 Llama/Command-R） |
| 序列建模 | 单 token 分类 | 自回归序列生成，Token-Level REINFORCE |
| LOO 优势 | 单步奖励 | 完整序列奖励的 LOO |
| KL 惩罚 | prompt 层面概率 KL | 逐 token KL 累加 |
| 奖励来源 | 模拟 RM / 规则 | 训练好的 RM 或规则奖励 |
| 训练规模 | 几百步 | 数万步，多 GPU 训练 |
| K 值 | 2~4（可调） | 通常 K=4，与论文一致 |

### RLOO 与 GRPO 对比

| 维度 | RLOO | GRPO |
|------|------|------|
| **提出时间** | 2024.02 | 2025.01 |
| **基线方式** | LOO（除自己外的均值） | 组内均值+标准差归一化 |
| **优势公式** | $A_k = \frac{K}{K-1}(R_k - \mu)$ | $A_i = (R_i - \mu) / \sigma$ |
| **无偏性** | ✅ 严格无偏 | ⚠️ 近似（除以σ引入轻微偏差） |
| **裁剪机制** | ❌ 无（纯 REINFORCE） | ✅ PPO-CLIP |
| **最小样本数** | K=2 即可 | 通常 G≥4（需要σ有效） |
| **损失粒度** | Token 级 | 样本级 |
| **适用场景** | 通用 RLHF | 推理任务 |
| **代表作品** | Command-R | DeepSeek-R1 |

---

## 6. DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization

### 论文信息
- **标题**: *DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization*
- **作者**: Yu Yang, Jingcheng Hu, Yixiao Li et al. (ByteDance Seed + Tsinghua AIR, 2025)
- **链接**: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- **发表时间**: 2025 年 3 月

### 💡 一句话讲透 DAPO

**"GRPO 在竞赛级数学推理上还有四个致命缺陷——DAPO 是给 GRPO 打了四个精确补丁的终极版本。"**

### 核心洞察：GRPO 的四大短板

GRPO 让 DeepSeek-R1 成功涌现了推理能力，但它并不是完美的。当 ByteDance Seed 团队试图在 **AIME 2024（美国数学邀请赛）** 级别的竞赛题上压榨极致性能时，他们发现 GRPO 存在四个关键问题：

| 问题 | 现象 | 根因 |
|------|------|------|
| **1. 裁剪不对称** | 高奖励样本的梯度被过早截断 | PPO 的对称裁剪不适合推理的"极致优化" |
| **2. 样本浪费** | 某些 prompt 的所有回答都对/都错 | 组内方差为 0 → GRPO 优势为 0 → 无学习信号 |
| **3. 长回答未惩罚** | 模型学会"绕圈子"凑长度 | 样本级损失对长回答更敏感，但对短回答不公正 |
| **4. 过长度奖励稀释** | 超长回答的 token 级贡献被稀释 | 没有长度惩罚 → 模型趋向生成冗长推理 |

DAPO 的解决方案非常工程化：**四个精确的补丁，逐一修复上述问题。**

### 🔬 创新一：Clip-Higher — 解耦裁剪

#### 问题

标准 PPO 裁剪使用对称区间 $[1-\epsilon, 1+\epsilon]$。但在推理任务中，我们希望**更大胆地提升好的推理方式**（允许更大的 ratio），而**保守地抑制差的推理**（避免过度惩罚）。

#### 解决方案

引入不对称裁剪：

$$\text{proposed\_clip}(r, \varepsilon_{\text{low}}, \varepsilon_{\text{high}}) = 
\begin{cases}
\max(r, 1-\varepsilon_{\text{low}}) & \text{如果优势 } A < 0 \text{（抑制坏行为）}\\
\min(r, 1+\varepsilon_{\text{high}}) & \text{如果优势 } A \geq 0 \text{（鼓励好行为）}
\end{cases}$$

但 DAPO 观察到：PPO 中的 $\min$ 操作已经隐含地处理了不同方向的裁剪。他们提出的 **Clip-Higher** 策略更为激进：

$$\text{Clip-Higher}(r, A) = 
\begin{cases}
\min\left(\max(r, 1-\varepsilon_{\text{low}}), \; 1+\varepsilon_{\text{high}}\right) \cdot A & \text{如果 } A > 0\\
0 & \text{如果 } A \leq 0
\end{cases}$$

**核心改动**：
- $\varepsilon_{\text{low}}$（下界）= 通常很小（如 0.2），允许快速抑制差行为
- $\varepsilon_{\text{high}}$（上界）= 设置得很大（如 0.6 甚至 1.0），允许好行为大幅提权
- 当 $A \leq 0$ 时直接设为 0（不更新），因为**只学对的，不强化学错的**

> 💡 **直觉**：标准 PPO 像一个"温和的老师"——你进步时拦住你、你退步时也拦住你。DAPO 的 Clip-Higher 像"虎爸狼妈"——你进步时全力助推（上限很高），你退步时直接忽略（不强化学坏习惯）。

### 🔬 创新二：Dynamic Sampling — 动态采样过滤

#### 问题

GRPO 对每个 prompt 采样 $G$ 个响应，用组内标准差归一化。但如果某道题所有 $G$ 个回答都正确（或都错误），组内标准差 $\sigma = 0$，导致优势全部为 0，**这批样本完全浪费了。**

#### 解决方案

在采样后动态过滤：

$$\text{过滤条件}: \text{std}(\{R_1, R_2, ..., R_G\}) > 0$$

即：只保留那些**至少有一个正确和一个错误回答**的 prompt 的采样组。其余 prompt 直接丢弃（不参与本次更新）。

**实现细节**：
- 对每个 prompt 的 $G$ 个响应，计算奖励的标准差
- 如果标准差 = 0 → 丢弃该 prompt 及所有 $G$ 个响应
- 被丢弃的 prompt 不会消失——下一轮采样时可能再次出现（因为模型在更新）

```python
def dynamic_sampling(rewards, threshold=1e-8):
    """过滤无效样本组：std(R) = 0 的prompt不参与更新"""
    # rewards: (batch, G)
    std_per_prompt = rewards.std(dim=-1)           # (batch,)
    valid_mask = std_per_prompt > threshold        # (batch,)
    return valid_mask
```

> 💡 **直觉**：如果一个班的所有学生都考了 100 分，这次考试就没有区分度——你不知道谁更优秀。DAPO 选择"跳过这次考试"，等下次模型更新后，区分度可能自然出现。

### 🔬 创新三：Token-Level Loss — Token 级损失

#### 问题

GRPO 使用**样本级损失**：对整个响应的所有 token 的对数概率求和后，乘上同一个优势。这导致：长回答中的每个 token 获得的总梯度与短回答不均衡。

#### 解决方案

改用 **Token 级损失**：对每个 token 分别计算损失，然后按序列长度归一化。

$$\mathcal{L}^{\text{token-level}} = -\frac{1}{\sum_{i=1}^{G} |a_i|} \sum_{i=1}^{G} \sum_{t=1}^{|a_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \; \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \hat{A}_i\right)$$

其中：
- $|a_i|$ 是第 $i$ 个回答的 token 数量
- $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_t)}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_t)}$ 是 token 级别的重要性采样比率
- $\sum_{i=1}^{G} |a_i|$ 是当前批次所有响应的总 token 数（归一化因子）

**关键差异**：
| 损失粒度 | GRPO（样本级） | DAPO（Token 级） |
|----------|---------------|-------------------|
| 归一化 | 除以 G（组大小） | 除以 $\sum|a_i|$（总 token 数） |
| 长短回答权重 | 长回答贡献更大 | 每个 token 贡献相等 |
| 对长回答的偏好 | 倾向于产生更长回答 | 无偏 |
| 训练稳定性 | 一般 | 更好（更多样本） |

### 🔬 创新四：Overlong Reward Shaping — 过长度奖励整形

#### 问题

模型可能生成超长推理链（如 >2048 tokens），这些回答的奖励信号被稀释——每个 token 的梯度贡献变得很微小。而且超长回答在实际部署中不可接受（推理成本高、延迟大）。

#### 解决方案

引入长度惩罚因子，重塑最终奖励：

$$R_{\text{shaped}} = R_{\text{original}} \times \exp\left(-\alpha \cdot \max\left(0, \frac{\text{len}(a) - L_{\text{max}}}{L_{\text{max}}}\right)\right)$$

其中：
- $R_{\text{original}}$：原始奖励（如准确率奖励 + 格式奖励）
- $\text{len}(a)$：响应的 token 长度
- $L_{\text{max}}$：允许的最大长度阈值
- $\alpha$：惩罚强度系数（通常 0.01 ~ 0.1）

惩罚曲线特点：
- 当 $\text{len}(a) \leq L_{\text{max}}$ 时：$R_{\text{shaped}} = R_{\text{original}}$（无惩罚）
- 当 $\text{len}(a) > L_{\text{max}}$ 时：指数衰减
- 超长越多，衰减越剧烈

```python
def overlong_reward_shaping(reward, response_len, L_max=2048, alpha=0.05):
    """过长度奖励整形"""
    excess = torch.clamp(response_len - L_max, min=0) / L_max
    penalty_factor = torch.exp(-alpha * excess)
    return reward * penalty_factor
```

### DAPO 完整目标函数

将四个创新组合，DAPO 的完整目标函数为：

$$\mathcal{L}_{\text{DAPO}} = -\frac{1}{\sum_{i \in \mathcal{V}} |a_i|} \sum_{i \in \mathcal{V}} \sum_{t=1}^{|a_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \; \text{proposed\_clip}(r_{i,t}, \varepsilon_{\text{low}}, \varepsilon_{\text{high}}) \hat{A}_i\right) + \beta \cdot D_{KL}\left(\pi_\theta \parallel \pi_{\text{ref}}\right)$$

其中：
- $\mathcal{V}$：经过 Dynamic Sampling 过滤后的有效 prompt 集合（$\text{std}(R) > 0$）
- $\text{proposed\_clip}$：Clip-Higher 解耦裁剪
- $\hat{A}_i$：组内标准化优势 $A_i = (R_{\text{shaped},i} - \mu)/\sigma$
- $R_{\text{shaped},i}$：经 Overlong Reward Shaping 处理后的奖励
- $\frac{1}{\sum |a_i|}$：Token-Level 归一化

### GRPO vs DAPO — 四大改进对比

| 改进维度 | GRPO | DAPO | 改进效果 |
|----------|------|------|----------|
| **裁剪策略** | 对称 PPO-CLIP：$[1-\varepsilon, 1+\varepsilon]$ | Clip-Higher：$\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$，且 $A \leq 0$ 直接置零 | 好的推理被更大胆强化，差的推理不被惩罚 |
| **样本利用** | 所有 prompt 参与的采样组都用于更新 | Dynamic Sampling：丢弃 $\text{std}(R)=0$ 的无效组 | 避免无效样本浪费计算、引入噪声 |
| **损失粒度** | 样本级损失：$\frac{1}{G}\sum_i$ | Token 级损失：$\frac{1}{\sum|a_i|}\sum_i\sum_t$ | 长回答不享特权，训练更稳定 |
| **长度控制** | 无长度惩罚 | Overlong Reward Shaping：长度指数惩罚 | 抑制过度冗长的推理链，保证推断效率 |

### AIME 2024 基准性能

DAPO 论文在 AIME 2024 竞赛题目上的核心结果（Qwen2.5-32B 基座）：

| 方法 | AIME 2024 Pass@1 | 相对 GRPO 提升 |
|------|-------------------|----------------|
| Qwen2.5-32B (Zero-shot) | 16.7% | — |
| + SFT | 23.3% | — |
| + GRPO (G=8) | 46.7% | 基线 |
| + GRPO (G=16) | 50.0% | +3.3% |
| + DAPO (G=8) | **53.3%** | **+6.6%** |
| + DAPO (G=16) | **56.7%** | **+10.0%** |

> 💡 AIME 2024 是高中数学竞赛级别的题目，普通高中生平均得分约 10-15%。DAPO 将开源模型的 Pass@1 从 16.7% 提升到 56.7%，这已经超越了大多数参赛学生。

### 超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $G$ | 8 ~ 16 | 每组采样响应数，与 GRPO 一致 |
| $\varepsilon_{\text{low}}$ | 0.2 | 下界裁剪（抑制坏行为） |
| $\varepsilon_{\text{high}}$ | 0.6 ~ 1.0 | 上界裁剪（鼓励好行为）——显著大于 GRPO |
| $\beta$ (KL) | 0.04 | KL 惩罚系数，与 GRPO 一致 |
| $L_{\text{max}}$ | 2048 ~ 4096 | 最大允许长度（超长惩罚启动阈值） |
| $\alpha$ (长度惩罚) | 0.02 ~ 0.1 | 长度惩罚强度 |
| 学习率 | $10^{-6} \sim 5 \times 10^{-6}$ | 标准 LLM 微调学习率 |

### 📝 代码走读（从外到内）

**第1层 — DAPO 四大创新组件初始化**（`dapo.py` DAPOTrainer 类）：
```python
class DAPOTrainer:
    def __init__(self, policy, ref_policy, group_size=8,
                 eps_low=0.2, eps_high=0.6, beta=0.04,
                 L_max=2048, alpha_len=0.05):
        self.policy = policy          # 可训练策略
        self.ref_policy = ref_policy  # 冻结参考
        self.G = group_size           # 组大小
        self.eps_low = eps_low        # Clip-Higher 下界
        self.eps_high = eps_high      # Clip-Higher 上界
        self.beta = beta              # KL 系数
        self.L_max = L_max            # 长度阈值
        self.alpha = alpha_len        # 长度惩罚强度
```

**第2层 — 四大创新在训练步中的组合**（`dapo.py` train_step 方法）：
```python
def train_step(self, prompts, targets):
    G = self.G

    # ---- Step 1: 采样G个响应 ----
    old_log_probs_list, responses_list, rewards_list = [], [], []
    self.policy.eval()
    with torch.no_grad():
        logits = self.policy(prompts)
        for _ in range(G):
            dist = torch.distributions.Categorical(logits=logits)
            response = dist.sample()
            log_prob = dist.log_prob(response)
            reward = self.compute_rule_reward(response, targets)
            # 创新四: Overlong Reward Shaping
            shaped_reward = self.overlong_shaping(reward, len(response))
            
            old_log_probs_list.append(log_prob)
            responses_list.append(response)
            rewards_list.append(shaped_reward)
    
    # (batch, G)
    old_log_probs = torch.stack(old_log_probs_list, dim=1)
    rewards = torch.stack(rewards_list, dim=1)
    responses = torch.stack(responses_list, dim=1)
    
    # ---- Step 2: 创新二 — Dynamic Sampling ----
    valid_mask = self.dynamic_sampling_filter(rewards)
    if valid_mask.sum() == 0:
        return {'loss': 0.0, 'filtered': True}  # 所有prompt都被过滤
    
    # 只保留有效的 prompt
    rewards = rewards[valid_mask]
    old_log_probs = old_log_probs[valid_mask]
    responses = responses[valid_mask]
    prompts = prompts[valid_mask]
    
    # ---- Step 3: GRPO 组内优势 ----
    advantages = self.group_relative_advantage(rewards)
    
    # ---- Step 4: 创新三 — Token-Level Loss + 创新一 Clip-Higher ----
    self.policy.train()
    self.optimizer.zero_grad()
    
    logits = self.policy(prompts)
    dist = torch.distributions.Categorical(logits=logits.unsqueeze(1))
    new_log_probs = dist.log_prob(responses)  # (valid_batch, G, seq_len)
    
    ratio = (new_log_probs - old_log_probs).exp()
    
    # Clip-Higher: 不对称裁剪
    # A > 0 → 用 eps_high; A <= 0 → 直接置零
    surr1 = ratio * advantages.unsqueeze(-1)
    
    # 分别裁剪上下界
    lower_clip = torch.clamp(ratio, min=1 - self.eps_low)
    upper_clip = torch.clamp(ratio, max=1 + self.eps_high)
    
    # A > 0 走 upper_clip; A <= 0 走 0
    clipped_adv = torch.where(
        advantages.unsqueeze(-1) > 0,
        upper_clip * advantages.unsqueeze(-1),
        torch.zeros_like(advantages.unsqueeze(-1))
    )
    
    surr2 = clipped_adv
    policy_loss = -torch.min(surr1, surr2)
    
    # Token-Level 归一化: 除以总token数
    total_tokens = responses.numel()
    policy_loss = policy_loss.sum() / total_tokens
    
    # ---- Step 5: KL 惩罚 ----
    kl = self.compute_kl(prompts, logits).mean()
    total_loss = policy_loss + self.beta * kl
    
    total_loss.backward()
    self.optimizer.step()
```

**第3层 — 四大创新的独立函数**（`dapo.py` 工具函数）：
```python
def overlong_shaping(self, reward, response_len):
    """创新四: 过长度奖励整形"""
    excess = max(0, response_len - self.L_max) / self.L_max
    return reward * math.exp(-self.alpha * excess)

def dynamic_sampling_filter(self, rewards):
    """创新二: 过滤std(R)=0的无效样本组"""
    # rewards: (batch, G)
    std_per_prompt = rewards.std(dim=-1)
    return std_per_prompt > 1e-8  # 只保留有区分度的prompt

def proposed_clip(self, ratio, advantage, eps_low, eps_high):
    """创新一: Clip-Higher 解耦裁剪"""
    lower_bounded = torch.clamp(ratio, min=1 - eps_low)
    upper_bounded = torch.clamp(ratio, max=1 + eps_high)
    # A > 0 时用 upper bound; A <= 0 时直接置零
    clipped = torch.where(advantage > 0, upper_bounded, 
                          torch.zeros_like(ratio))
    return clipped * advantage

def token_level_normalize(self, per_token_losses):
    """创新三: Token-Level 归一化"""
    return per_token_losses.sum() / per_token_losses.numel()
```

### 🔬 公式与代码一一对应

下表展示了 DAPO 中每个数学公式与 `dapo.py` 中代码的精确对应关系：

| 公式 | 数学表达 | 代码位置 | 代码片段 |
|------|----------|----------|----------|
| **组内优势** | $$A_i = \frac{R_i - \mu_G}{\sigma_G + \varepsilon}$$ | `DAPOTrainer.group_relative_advantage()` (第118-120行) | `(rewards - mean_r) / (std_r + 1e-8)` |
| **★ 解耦裁剪** | $$\text{clip}(r_i,\; 1-\varepsilon_{\text{low}},\; 1+\varepsilon_{\text{high}})$$ | `DAPOTrainer.train_step()` (第261-266行) | `torch.clamp(ratio, 1 - self.clip_epsilon_low, 1 + self.clip_epsilon_high)` |
| **★ 动态采样** | $$\mathbf{1}[\text{std}(R) > 0]$$ | `DAPOTrainer.dynamic_sampling_filter()` (第134-136行) | `valid_mask = reward_std > 1e-8` |
| **★ 过长惩罚** | $$R_{\text{shaped}} = R_{\text{original}} - \alpha \cdot |\log\pi_{\text{old}}|$$ | `DAPOTrainer.compute_reward()` (第158-159行) | `rewards = accuracy - self.overlong_alpha * torch.abs(old_log_probs)` |
| **重要性比率** | $$r_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$ | `DAPOTrainer.train_step()` (第258行) | `ratio = (new_log_probs - kept_old_log_probs).exp()` |
| **★ 总损失** | $$\mathcal{L}_{\text{DAPO}} = -\frac{1}{|V|}\sum_{i\in V} \min(r_i A_i, \text{decoupled\_clip}(r_i) A_i) + \beta D_{KL}$$ | `DAPOTrainer.train_step()` (第267, 272行) | `policy_loss = -torch.min(surr1, surr2).mean()` / `total_loss = policy_loss + self.beta * kl` |

**公式逐行解读:**

1. **★ 解耦裁剪** `clip(r, 1-ε_low, 1+ε_high)` — DAPO 第一大创新：
   - 第一步：`surr1 = ratio * advantages` — 未裁剪的原始优势加权比率（第261行）
   - 第二步：`torch.clamp(ratio, 1 - self.clip_epsilon_low, 1 + self.clip_epsilon_high)` — **不对称裁剪**（第262-266行）
   - **不对称的数学意义**：
     - 上界 `1+ε_high = 1.28`（默认 ε_high=0.28）：允许好行为的比率放大 **28%**（vs GRPO 的 20%）
     - 下界 `1-ε_low = 0.80`（默认 ε_low=0.20）：坏行为的比率最多压缩 **20%**（与 GRPO 一致）
   - **为什么上界更大？** DAPO 的核心洞察：推理任务中，**"正确的推理路径"应该被大胆放大**，而"错误的推理"只需保守抑制
   - 对比 GRPO 的对称裁剪 `clip(ratio, 1-0.2, 1+0.2)`：上下对称限制了探索空间

2. **★ 动态采样** `std(R) > 0` — DAPO 第二大创新：
   - 第一步：`reward_std = rewards.std(dim=-1)` — 计算每个 prompt 的 G 个奖励的标准差（第134行）
   - 第二步：`valid_mask = reward_std > 1e-8` — 仅保留标准差 > 0 的 prompt（第135行）
   - 第三步：`kept_prompts = prompts[valid_mask]` — 过滤后仅用有效样本更新（第242行）
   - **过滤逻辑**：
     - `std(R) = 0`：所有 G 个回答全对或全错 → 该 prompt 当前没有任何区分度 → 丢弃
     - `std(R) > 0`：至少有一个对一个错 → 有学习价值 → 保留
   - **关键**：被丢弃的 prompt 不会永久消失——等模型下一轮更新后，区分度可能自然出现
   - 直觉：考试如果全班都考了 100 分，这场考试就白考了——等下次更有区分度的考试

3. **★ 过长惩罚** $R - \alpha \cdot |\log\pi_{\text{old}}|$ — DAPO 第四大创新（本实现中与第三创新联动）：
   - 第一步：`accuracy = (responses == target.unsqueeze(1)).float()` — 原始准确率奖励（第155行）
   - 第二步：`penalty_factor = torch.abs(old_log_probs)` — 惩罚因子：旧策略的对数概率绝对值（第158行）
   - 第三步：`rewards = accuracy - self.overlong_alpha * penalty_factor` — 最终奖励（第159行）
   - **直觉**：`|log_prob|` 越大意味着模型对该采样**越不确定**（概率越低）→ 低质量/高不确定性的响应被惩罚
   - 在真实 DAPO 中，这是基于响应长度（token 数）而非概率——本实现用概率绝对值作为简化代理
   - 参数 α（`overlong_alpha`）= 0.05：控制惩罚强度

4. **★ Token-Level Loss（第三大创新）**：
   - 本简化实现中，`-torch.min(surr1, surr2).mean()` 对 (num_kept × G) 的所有元素取平均
   - `.mean()` 等价于对所有保留样本的贡献做均匀平均 → 避免了长回答在损失中占比过大
   - 真实 DAPO 中，分母为 $\sum |a_i|$（总 token 数），而非组大小 G——核心目的是**让长回答不享特权**

5. **DAPO 训练流程概览**（四大创新 ★ 标注）：
   ```
   ① 采样 G 个响应 + 存储 old_log_probs
   ② ★ 过长惩罚: R = accuracy - α·|log_prob|
   ③ ★ 动态采样: 过滤 std(R)=0 的 prompt
   ④ 组内优势: A_i = (R_i - μ) / σ
   ⑤ 计算 ratio = exp(new - old)
   ⑥ ★ 解耦裁剪: clip(ratio, 1-ε_low, 1+ε_high)
   ⑦ ★ Token-Level 归一化 + KL → 梯度更新
   ```

> 📐 **DAPO = GRPO + 四大精确补丁**：
> ```
> GRPO 基础:    组内优势 + PPO 对称裁剪 + 规则奖励
> DAPO 改进 ①:  Clip-Higher       裁剪不对称（上限更高）
> DAPO 改进 ②:  Dynamic Sampling   过滤无区分度样本
> DAPO 改进 ③:  Token-Level Loss   逐 token 归一化
> DAPO 改进 ④:  Overlong Shaping   惩罚低质量响应
> 
> 结果: AIME 2024 从 GRPO 的 46.7% → DAPO 的 53.3%（G=8）
> ```

### 💻 关键代码速查

| # | 代码行 | 为什么关键 |
|---|--------|-----------|
| 1 | `torch.clamp(ratio, 1 - self.clip_epsilon_low, 1 + self.clip_epsilon_high)` | **Clip-Higher**：DAPO 最大的创新——不对称裁剪允许好行为获得比 GRPO 更大的概率提升空间 |
| 2 | `valid_mask = reward_std > 1e-8` | **Dynamic Sampling**：一行过滤掉所有无区分度的训练样本，避免在无效数据上白费算力 |
| 3 | `rewards = accuracy - self.overlong_alpha * torch.abs(old_log_probs)` | **Overlong Shaping**：惩罚模型的高不确定性采样，间接抑制低质量/过长的推理链 |

#### ⚠️ 简化说明（理论与实践差距）

| 项目 | 本实现 | 真实 DAPO |
|------|--------|-----------|
| 模型 | MLP 分类器 | Transformer（如 Qwen2.5-32B） |
| 序列生成 | 单 token 分类 | 自回归生成，真实序列级 token |
| Token-Level Loss | 单 token 等同于样本级 | 真实多 token 序列的逐 token 计算 |
| Dynamic Sampling | 单步奖励 std 过滤 | 完整序列奖励 std 过滤 |
| Overlong Shaping | 简化长度计算 | 真实 tokenizer 级别的长度计数 |
| AIME 评测 | 不适用（任务不同） | 完整 AIME 2024 竞赛评估 |
| 训练规模 | 几百步 | 数万步，16-64 GPU |

---

## 7. 五大算法总结对比表

| 维度 | RLHF | DPO | RLOO | GRPO | DAPO |
|------|------|-----|------|------|------|
| **提出时间** | 2022 (OpenAI) | 2023 (Stanford) | 2024.02 (Cohere) | 2025.01 (DeepSeek) | 2025.03 (ByteDance) |
| **核心论文** | InstructGPT | DPO | Back to Basics | DeepSeek-R1 | DAPO |
| **典型代表** | ChatGPT, Claude, Gemini | Zephyr, Llama 3, Qwen 2 | Command-R | DeepSeek-R1 | 竞赛级推理模型 |
| **模型数量** | 4（Policy + Ref + Critic + RM） | 2（Policy + Ref） | 2（Policy + Ref） | 2（Policy + Ref） | 2（Policy + Ref） |
| **需要 Critic** | ✅ 需要 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |
| **需要奖励模型** | ✅ 需要（神经RM） | ❌ 不需要（隐式） | 可选（RM或规则） | ❌ 不需要（规则奖励） | ❌ 不需要（规则奖励） |
| **裁剪机制** | PPO-CLIP 对称 | 无 | 无 | PPO-CLIP 对称 | Decoupled CLIP 不对称 |
| **优势估计** | GAE + Critic | 隐式（对数比率差） | LOO 均值中心化（无偏） | 组内标准差归一化 | 组内标准差归一化 |
| **损失粒度** | Token 级 | 样本级 | Token 级 | 样本级 | Token 级 |
| **奖励来源** | 神经 RM | 隐式 | RM 或规则 | 可编程规则 | 可编程规则 |
| **训练方式** | 在线 RL | 离线监督 | 在线 RL | 在线 RL | 在线 RL |
| **稳定性** | 低 | 高 | 中高 | 中 | 中 |
| **计算开销** | 极高（~4×模型） | 低（~2×模型） | 低（~2×模型） | 中（~2×模型 + G次采样） | 中（~2×模型 + G次采样 + 4项改进） |
| **推理能力** | 一般 | 一般 | 一般 | 极强 | **极强（超越GRPO）** |
| **算法简洁度** | 复杂（三阶段） | 极简（单阶段） | 极简（单阶段） | 中等 | 较复杂（四项改进） |
| **适用场景** | 通用对话对齐 | 开源微调 | 通用RLHF | 推理任务 | **竞赛级推理** |
| **最小 GPU 需求** | 8×A100 | 2×A100 | 2×A100 | 4×A100 | 8×A100 |

---

## 8. 🧭 五算法快速决策流程图

```
你要对齐一个LLM，该选哪个算法？
                │
                ▼
    ┌─────────────────────────────┐
    │ 你的任务有可验证的正确答案吗？ │
    │   （数学题、编程、逻辑推理）   │
    └──────────────┬──────────────┘
                   │
         ┌─────────┴─────────┐
         │ YES               │ NO
         ▼                   ▼
  ┌──────────────────┐   ┌──────────────────┐
  │ 你追求竞赛级极致   │   │ 你有大规模人类    │
  │ 性能（>AIME 50%）?│   │ 偏好比较数据吗？  │
  └────────┬─────────┘   └────────┬─────────┘
           │                      │
     ┌─────┴─────┐          ┌─────┴─────┐
     │YES   │NO  │          │YES   │NO  │
     ▼      ▼    │          ▼      ▼    │
   DAPO   GRPO   │        RLHF   DPO    │
                 │                      │
    ┌────────────┴──────────────────────┘
    │
    ▼
  ┌──────────────────────────────────────────┐
  │ 补充决策：如果你要的是"最简洁的在线RL"：  │
  │                                           │
  │  ┌─────────────────────────────────┐      │
  │  │ 你能接受无裁剪 + K=4采样？       │      │
  │  └────────────┬────────────────────┘      │
  │               │                           │
  │         ┌─────┴─────┐                     │
  │         │YES   │NO  │                     │
  │         ▼      ▼    │                     │
  │       RLOO   按上面  │                     │
  │              的流程   │                     │
  └──────────────────────┴─────────────────────┘
    │
    ▼
  ┌──────────────────────────────────────────────────────┐
  │ 快速参考：                                            │
  │ • 推理任务 + 极致性能 → DAPO（AIME 50%+）              │
  │ • 推理任务 + 标准性能 → GRPO（DeepSeek-R1 方案）       │
  │ • 通用RLHF + 追求简单 → RLOO（无裁剪、无Critic）       │
  │ • 对话对齐 + 有GPU + 有标注 → RLHF（ChatGPT 方案）     │
  │ • 资源有限 / 开源微调 → DPO（离线、稳定）              │
  │ • 有偏好数据但无GPU → DPO                             │
  │ • 无偏好数据但想在线探索 → RLOO 或 RLHF                │
  └──────────────────────────────────────────────────────┘
```

---

## 9. ❓ 常见 FAQ

### Q1: RLHF 会奖励黑客（Reward Hacking）吗？怎么防止？

**简短回答**：会。没有 KL 惩罚的 RLHF 必然导致奖励黑客。

**具体机制**：奖励模型 $r_\phi$ 是训练出来的，不是完美的。语言模型会在训练中发现 RM 的"盲区"——比如发现重复特定关键词或使用特定句式就能拿到高分（但这不符合人类真实偏好）。

**防护手段**（按效果排序）：
1. **KL 惩罚**（最常用）：$\beta \cdot D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$，确保策略不偏离 SFT 模型太远
2. **RM 在线更新**：定期用新的人类反馈数据重新训练 RM，堵住被发现的漏洞
3. **多维度奖励**：不只用单一 RM，同时用有用性、安全性、真实性等多维评估
4. **预训练梯度**：在 RLHF 损失中加入语言建模损失，防止语言退化

### Q2: DPO 和 RLHF 该选哪个？

| 你的情况 | 推荐 | 原因 |
|----------|------|------|
| GPU 资源有限（<8 卡） | **DPO** | RLHF 需要 4 个模型常驻显存，DPO 只需 2 个 |
| 已有高质量偏好数据集 | **DPO** | DPO 离线训练，不需要在线采样 |
| 需要在线探索（没偏好数据） | **RLHF** | RLHF 可以自己生成样本，由 RM 在线评分 |
| 追求极致对齐质量 | **RLHF** | 在线 RL 可以探索出训练数据中不存在的好回答 |
| 开源模型微调（如 Llama） | **DPO** | 社区主流选择，工具链成熟（TRL 库） |
| 需要迭代提升（多轮训练） | **RLHF** | 可以多轮循环：收集新数据 → 更新 RM → 继续 PPO |

> 💡 **业界趋势**：RLHF 用于头部公司（OpenAI、Anthropic），DPO 用于开源社区。两者不是"谁替代谁"，而是**不同约束下的最优解**。

### Q3: GRPO 是不是只适用于数学/编程任务？

**目前主要适用于有可验证答案的任务**，这是由"规则奖励"决定的——你需要一个能自动判断答案对错的机制。

但 GRPO 的思想可以扩展到其他领域：
- **代码生成**：通过测试用例通过率作为规则奖励
- **翻译**：通过 BLEU/COMET 等自动指标作为规则奖励
- **安全对齐**：可以通过规则检测有害内容（但需要配合人工审核）
- **多模态**：图像生成的 aesthetic score 等可编程指标

对于纯粹的"对话质量"优化（无法用规则衡量），DPO 或 RLHF 仍然是更好的选择。

### Q4: SFT 已经让模型表现很好了，RLHF 的额外收益到底有多少？

根据 InstructGPT 论文的实验数据：

| 指标 | 仅 SFT | SFT + RLHF | 提升 |
|------|--------|-----------|------|
| 人类偏好胜率 vs SFT | 50%（基线） | **71%** | +21% |
| 真实性（Truthfulness） | 基线 | +10% | 显著 |
| 有害输出率 | 基线 | -25% | 显著 |
| 指令遵循能力 | 基线 | 明显提升 | ~15% |

> 💡 RLHF 不是"锦上添花"，而是**从"会说话"到"说人话"的质变**。SFT 教会模型格式，RLHF 教会模型判断"什么算好的回答"。

### Q5: 用 DPO 会不会"学不到新东西"（因为只用离线数据）？

这是一个非常敏锐的问题。DPO 确实存在**离线数据的分布限制**——它只能优化训练数据中见过的 chosen/rejected 对。如果最优回答不在当前策略的生成分布中，DPO 发现不了它。

这就是为什么：
- **RLHF 仍然被需要**：在线 PPO 可以**探索并发现**训练数据中不存在的好回答
- **迭代 DPO** 是折中方案：用当前模型生成新回答 → 人工标注偏好 → 再用 DPO 训练
- **在线 DPO** 是前沿方向：近年研究尝试在 DPO 框架中加入在线采样，兼顾效率和探索

### Q6: RLOO 为什么宣称"返璞归真"？REINFORCE 不是最原始的算法吗？

**简短回答**：正因为 REINFORCE 是最原始的，它在"简单的正确性"上反超了复杂的 PPO。

**具体原因**：
1. **RLHF 场景的特殊性**：LLM 对齐中的奖励信号是**稀疏的**（只在序列末尾给出），这意味着 GAE（广义优势估计）的多步奖励累积收益有限
2. **LOO 基线的无偏性**：Critic 网络是一个近似器，它的估计总是有偏的。而 LOO 基线是严格无偏的——在数学上更"干净"
3. **减少超参数**：PPO 有 $\epsilon$（裁剪）、$\lambda$（GAE）、多种学习率等超参数需要调优。RLOO 只需要 $K$ 和 $\beta$
4. **去工程化**：PPO 的 Critic 网络 + GAE + mini-batch + 多次更新是一个复杂的工程系统。RLOO 就是一个 for 循环——采样 K 次 → 算 LOO → 梯度更新

> 💡 RLOO 论文的核心发现：在 LLM RLHF 中，**PPO 的所有花哨组件（Critic, GAE, Clipping）加起来带来的收益，不如一个简洁的 LOO 基线来得稳定和高效。**

### Q7: RLOO 和 GRPO 都是无 Critic 的算法，有什么区别？

这是自然会产生的问题——二者确实"长得很像"。关键区别在于**基线/优势的计算方式**：

| 维度 | RLOO | GRPO |
|------|------|------|
| **基线** | Leave-One-Out：$b_k = \frac{1}{K-1}\sum_{j \neq k} R_j$ | 组内均值 + 标准差归一化 |
| **优势公式** | $A_k = \frac{K}{K-1}(R_k - \mu)$ | $A_i = (R_i - \mu) / \sigma$ |
| **无偏性** | ✅ 严格无偏 | ⚠️ 除以 $\sigma$ 有轻微偏差 |
| **裁剪** | ❌ 无（纯 REINFORCE） | ✅ PPO-CLIP |
| **适用场景** | 通用 RLHF | 推理任务（CoT 涌现） |
| **最小 K** | K=2 即可 | 通常 G≥4（需要稳定的 $\sigma$） |

> 💡 **简单记忆**：RLOO = "同学的绝对分数比较"，GRPO = "相对排名（z-score）"。前者无偏但保守，后者激进但可能引入偏差。

### Q8: DAPO 的四个改进是不是可以独立使用？

**是的，完全独立。** DAPO 的四个创新是**正交的**（互不依赖）：

1. **Clip-Higher**：纯修改裁剪策略，可以单独应用在任何使用 PPO-CLIP 的算法中
2. **Dynamic Sampling**：纯数据过滤，可以在采样后单独应用
3. **Token-Level Loss**：纯损失函数修改，可以应用到 GRPO、RLOO 甚至 PPO
4. **Overlong Reward Shaping**：纯奖励修改，可以应用到任何 RL 算法

你可以只选其中一个改进来用——比如在现有的 GRPO 实现中只加 Overlong Reward Shaping，而不改其他三项。

### Q9: DAPO 和 GRPO 性能差距到底有多大？值得额外的复杂度吗？

根据 DAPO 论文的实验（Qwen2.5-32B 基座，AIME 2024）：

| 方法 | AIME 2024 Pass@1 | 复杂度 |
|------|-------------------|--------|
| 基座模型 (Zero-shot) | 16.7% | — |
| + SFT | 23.3% | 低 |
| + GRPO (G=8) | 46.7% | 中 |
| + DAPO (G=8) | **53.3%** | 中高 |
| 差距 | **+6.6 个百分点** | — |

> 💡 6.6 个百分点的绝对提升是**非常显著**的。这意味着每 15 道题就能多解对 1 题。对于 AIME 级别的竞赛，这可能是"入围"和"出局"的差别。如果你的目标是**竞赛级推理**，DAPO 的额外复杂度是值得的。如果你的目标是**通用推理**，GRPO 已经足够好。

---

## 10. 简化项汇总

由于本课程使用 MLP 简化模型进行教学演示，与真实 LLM 训练存在以下重要差异：

| 简化项 | 本课程实现 | 真实 LLM 训练 |
|--------|-----------|---------------|
| **基础模型** | 小型 MLP（~100K 参数） | GPT/LLaMA 等 Transformer（1B-405B 参数） |
| **序列建模** | 单 token 分类 | 自回归序列生成（逐 token 预测下一个 token） |
| **对数概率** | $\log \pi(label \mid input)$ | $\sum_t \log \pi(token_t \mid input, tokens_{<t})$ |
| **PPO 实现** | 简化 REINFORCE 损失 | 完整 PPO-CLIP + Critic + GAE + mini-batch |
| **RLOO 实现** | 单步 LOO 基线，REINFORCE | 多步序列 LOO，Token-Level REINFORCE |
| **DAPO 实现** | 单步四改进，简化裁剪 | 完整序列 Clip-Higher + Dynamic Sampling + Token-Level + Overlong |
| **奖励模型输入** | 单 token one-hot 向量 | prompt + 完整生成序列的嵌入 |
| **KL 计算** | 单步概率分布 KL | 逐 token 累加 KL 散度 |
| **数据** | 随机生成的模拟数据 | 数万-数十万条真实人类标注数据 / 竞赛题目 |
| **训练规模** | 数百步，CPU 可运行 | 数十万-百万步，需要数百-数千 GPU |
| **组采样（GRPO/DAPO）** | 从同一分布重复采样 | 从同一 prompt 采样 G 个**不同**的响应序列 |
| **LOO 采样（RLOO）** | 从同一分布重复采样 | 从同一 prompt 采样 K 个**不同**的响应序列 |

这些简化是**有意为之**的——目标是让你在几分钟内就能理解和运行全部五种算法的核心逻辑，而不需要 A100 集群。

---

## 11. 运行方法

本章所有算法均提供完整的教学演示脚本：

```bash
# RLHF 三阶段完整演示
cd chapter_04_llm_rl/01_rlhf
python rlhf.py
# 或使用 train.py：
python train.py --demo

# DPO 直接偏好优化演示
cd chapter_04_llm_rl/02_dpo
python dpo.py
# 或使用 train.py：
python train.py --demo

# GRPO (DeepSeek-R1 核心) 演示
cd chapter_04_llm_rl/03_grpo
python grpo.py
# 或使用 train.py：
python train.py --demo

# RLOO (REINFORCE Leave-One-Out) 演示
cd chapter_04_llm_rl/04_rloo
python rloo.py

# DAPO (Decoupled Clip and Dynamic sAmpling) 演示
cd chapter_04_llm_rl/05_dapo
python dapo.py
```

每个脚本运行时间约 10-30 秒，在终端中实时输出训练指标。

---

## 12. 学习路线建议

1. **先读"为什么需要 RL"（第0节）**：理解 SFT 的局限性和 RL 的必要性，这是阅读后续所有内容的**认知前提**。很多 LLM 工程师卡在"为什么不直接用 SFT"这个问题上——这一节就是为此准备的。

2. **然后理解 RLHF 三阶段**：这是基石。搞清楚为什么需要 SFT → RM → PPO 这个链条，每个阶段在解决什么问题。对照代码走读的"三层结构"来理解。**重点使用 🔬 公式与代码一一对应 表格**：逐行对照数学公式与 `rlhf.py` 代码，确保你看到的每个公式都能在代码中找到精确映射。

3. **再学 DPO 为什么更简洁**：理解"拔河比喻"下的隐式奖励数学洞察——这是当前主流开源模型（如 Llama 3）对齐方案的基础。**打开 🔬 公式-代码映射表格**，跟踪 β·log(π/π_ref) 如何从数学变成代码。

4. **理解 RLOO 的"返璞归真"**：用 LOO 基线替代 Critic，用纯 REINFORCE 替代 PPO。这节让你明白——**复杂工程并不总是胜出，简单的数学正确性有时更重要。** 注意对比 RLOO 和 GRPO 在公式-代码映射表中无裁剪 vs 有裁剪的区别。

5. **然后看 GRPO 的推理革命**：理解组内相对优势和规则奖励如何让 LLM "学会思考"。注意：只有推理类任务适用。**对照 🔬 表格**理解 `(rewards - mean_r) / (std_r + 1e-8)` 这一行如何替代了整个 Critic 网络。

6. **最后深入 DAPO 的工程精进**：理解四项精确改进如何在 GRPO 之上进一步压榨性能。这节展示了一个重要原则——**在好的基础算法上做精确的工程改进，而不是发明全新范式。** **逐个追踪 ★ 标注的四大创新**在公式-代码表格中的映射位置。

7. **善用 💻 关键代码速查**：每个算法的公式-代码映射部分末尾都有一个三行速查表——它列出每个算法中**最关键的 3 行代码**及原因。当你需要在项目中快速回忆某算法的核心时，直接跳到这个速查表。

8. **参考决策流程图**：当你需要在真实项目中选择算法时，回到第8节的流程图。

9. **关注"理论与实践"差异**：每个算法后的 ⚠️ 简化说明表格帮你理解实现在哪里简化了，以及真实训练要面对什么。

10. **必读 FAQ**：第9节涵盖了 RLHF 入门的最高频疑问，建议在学完算法后通读。

---

## 13. 关键概念速查

### KL 惩罚的重要性
防止模型：
- 生成无意义但高奖励的文本（**奖励黑客**）
- 遗忘预训练知识（**灾难性遗忘**）
- 偏离自然语言分布（**语言退化**）

### 偏好数据格式
```json
{
  "prompt": "解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，通过数据训练模型...",
  "rejected": "机器学习就是让机器学习。"
}
```

### 推理强化学习的关键要素（GRPO/DAPO 启示）
- **Chain-of-Thought（思维链）**：模型在给出答案前先展示推理过程
- **Self-Reflection（自我反思）**：模型检查自己的推理是否有误
- **Self-Correction（自我修正）**：发现错误后自动修正并重新推导
- **Emergent Reasoning（涌现式推理）**：这些能力不是手动编程的，而是 GRPO/DAPO 训练中自发产生的

### RLOO 的核心不等式
LOO 基线无偏性的本质在于：对于独立同分布的 $K$ 个样本：
$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(y_k|x) \cdot \frac{1}{K-1}\sum_{j \neq k} R(y_j)\right] = 0$$
因此**减去 LOO 基线不改变梯度期望**，但大幅降低方差。

### DAPO 的四大支柱
1. **Clip-Higher**：好行为放大上限、坏行为直接归零
2. **Dynamic Sampling**：只学有区分度的样本
3. **Token-Level Loss**：让长回答不享特权
4. **Overlong Reward Shaping**：控制推理链长度
