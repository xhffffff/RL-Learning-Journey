# 第二章：深度强化学习 (2015–2018)

> **阅读前提**：本章假定你已经理解第一章中的 MDP、Q-Learning、SARSA、REINFORCE 以及 On-Policy/Off-Policy 的区别。如果你对概念还不熟悉，建议先回顾第一章。

本章介绍深度神经网络与强化学习的融合，涵盖从 **DQN (2015)** 到 **SAC (2018)** 六个里程碑式算法。当状态/动作空间从离散变为连续、维度从个位数暴涨到数千维时，第一章中的 **Q 表**已经无法胜任——深度神经网络从此登场。

> **本章学习目标**：读完本章后，你应该能够：(1) 解释经验回放和目标网络为什么是 DQN 成功的关键；(2) 说出 Double DQN、Dueling DQN 各自解决了 DQN 的什么问题；(3) 理解 A2C 到 PPO 的演进逻辑（裁剪目标的作用）；(4) 描述 SAC 的"软更新"和"双 Q"机制。

---

## 1. 从表格方法到深度神经网络

### 1.1 表格方法的局限性 —— 为什么 Q 表不够用了？

#### 直觉理解

第一章中，Q-Learning 和 SARSA 都用一个 **Q 表**（一张二维表格）来存储每个（状态, 动作）组合的价值。这在小规模问题上非常有效。但想象一下：如果状态是**一张图片**（比如 Atari 游戏的画面），有多少种可能的图片？

答案是：**天文数字**。一张 84x84 像素的灰度图像就有 $256^{7056}$ 种组合——远超宇宙中的原子总数。你不可能在内存中维护这么大一张表。

#### 直观类比：电话簿 vs. 搜索引擎

- **Q 表**就像一本电话簿：你只能查到"记录在案"的号码。如果遇到一个不在电话簿上的人，你就束手无策。
- **深度神经网络**就像一个搜索引擎：即使你输入一个从未见过的查询，它也能根据"相似性"给出合理的答案。

这引出了一个关键概念：**泛化（Generalization）**。神经网络可以在未见过的状态上做出合理的预测，因为相似的输入会产生相似的输出。

#### 三大问题总结

| 问题 | 说明 |
|------|------|
| **状态数量爆炸** | 自动驾驶的摄像头图像有 $256^{800\times600\times3}$ 种组合，远超宇宙原子总数 |
| **泛化缺失** | Q 表对未见过的状态无任何先验知识，每个格子独立学习 |
| **内存不可行** | 即便是简单游戏也经常有 $10^4$–$10^6$ 种状态 |

---

> **检查点 1**："泛化"在强化学习中意味着什么？为什么表格方法没有泛化能力，而神经网络有？

---

### 1.2 函数逼近（Function Approximation）—— 用神经网络替代 Q 表

核心思路：用一个带参数 $\theta$ 的函数 $f_\theta$ 来**近似** Q 值或策略，而非维护一张表。

$$Q(s, a) \approx Q_\theta(s, a) \quad \text{或} \quad \pi(a|s) \approx \pi_\theta(a|s)$$

**深度神经网络**是目前最强大的函数逼近器。然而直接将神经网络引入 RL 会面临两个额外挑战：

1. **样本相关性**：RL 产生的数据是时间序列，前后样本高度相关，违反 i.i.d.（独立同分布）假设。如果把连续帧直接喂给神经网络，网络会"记住"时间顺序而非学习真实规律。
2. **非平稳目标**：网络同时用于生成数据和计算更新目标。每次更新后，网络改变了，之前生成的数据所对应的"正确答案"也随之改变——相当于"追逐自己的尾巴"。

这两个问题分别由 **经验回放（Experience Replay）** 和 **目标网络（Target Network）** 来解决。

---

## 2. 两大核心技术：经验回放与目标网络

### 2.1 经验回放（Experience Replay）

#### 直觉理解与类比：复习错题本

想象一个学生在做练习题。如果她只是**按顺序做题、从不错题回顾**，那么：
- 前面章节的知识可能在做后面章节时已经遗忘
- 她做的题高度相关（同一章节的题都很类似），无法形成全面理解

更好的方法是什么？**把所有做过的题放进一个"错题本"（回放缓冲区），然后随机抽取题目来复习**。这样做的好处：
- 打破了章节顺序（**打破时间相关性**）
- 每道题可以被复习多次（**提高样本效率**）
- 不同章节的题目混合复习（**平滑学习分布**）

经验回放就是强化学习的"错题本复习法"！

#### 原理

将智能体与环境交互产生的转移数据 $(s, a, r, s', done)$ 存储到一个固定容量的**回放缓冲区（Replay Buffer）**中。训练时，从缓冲区随机采样一个 **mini-batch** 来计算梯度。

| 优点 | 说明 |
|------|------|
| **打破相关性** | 随机采样使训练数据近似 i.i.d. |
| **提高样本效率** | 每条经验可被多次使用（而非用一次就丢掉） |
| **平滑学习分布** | 缓冲区保留了多样化的历史经验，缓解灾难性遗忘 |

> 在我们的代码中，回放缓冲区由 `common/replay_buffer.py` 中的 `ReplayBuffer` 类实现。DQN 系列算法初始化时指定 `buffer_capacity`（如 `10000`），并设定 `batch_size`（如 `64`）来控制每次更新使用的样本数。

---

### 2.2 目标网络（Target Network）

#### 直觉理解与类比：射击移动靶 vs. 固定靶

射击训练的时候，你是愿意打**移动靶**还是**固定靶**？

DQN 的困境就像在打移动靶：你瞄准一个目标，正要扣动扳机，目标却移走了。具体来说：
- 每次更新 Q 网络，网络的参数变了
- 这导致 TD 目标 $r + \gamma \max_{a'} Q_\theta(s', a')$ 也立刻改变了
- 网络在追逐一个**不断移动的目标**，极易发散

**目标网络**的解决方案很简单：**锁定一个"固定靶"，每隔一段时间才更新靶的位置**。这就像射击训练中，先打固定靶，练好了再打移动靶。

#### 原理

维护两个结构相同但参数不同的 Q 网络：

- **在线网络** $Q_\theta$：每一步都更新，用于动作选择
- **目标网络** $Q_{\theta^-}$：定期（如每 $N$ 步）从在线网络复制参数，用于计算 TD 目标

若没有目标网络，TD 目标 $r + \gamma \max_{a'} Q_\theta(s', a')$ 中的 $Q_\theta$ 会随着梯度更新而立刻变化。这相当于更新时目标也在移动，极易导致训练发散。

目标网络的参数 $\theta^-$ 保持固定（或缓慢更新），使得 TD 目标在一段时间内稳定，相当于监督学习中的「固定标签」。

---

> **检查点 2**：如果把经验回放和目标网络比作"学骑自行车"，它们分别对应什么？提示：经验回放 = 反复练习各种路况，目标网络 = ?

---

## 3. 算法详解

### 架构演进路线图

在深入各算法之前，先看清整体演进路线：

```
Q-Learning (1989, 表格)
    │
    │  引入深度神经网络
    ▼
DQN (2015) ─── 核心：经验回放 + 目标网络
    │
    ├─→ Double DQN (2015) ─── 改进：解耦动作选择与价值评估
    │
    └─→ Dueling DQN (2016) ─── 改进：拆分 V(s) 和 A(s,a)
    
REINFORCE (1992, 策略梯度)
    │
    │  引入 Critic 基线
    ▼
A2C (2016) ─── 核心：Actor-Critic 架构
    │
    │  引入裁剪目标
    ▼
PPO (2017) ─── 核心：裁剪 + 多轮优化
    
SAC (2018) ─── 融合：Off-Policy + Actor-Critic + 最大熵
```

---

### 3.1 DQN (Deep Q-Network, 2015) -- 里程碑

**论文**：*Human-level control through deep reinforcement learning* -- Mnih et al., Nature 2015

#### 直觉理解：DQN 想解决什么问题？

Q-Learning 用一张表格存储 Q 值，当状态是图片时就彻底失效了。DQN 用一个**神经网络**替代 Q 表：输入是游戏画面（状态），输出是每个可能动作的 Q 值。

但是，直接用神经网络做 Q-Learning 会遇到两个致命问题：
1. 连续的游戏画面高度相关，神经网络会"过拟合"到最近的画面
2. 网络在追逐自己产生的目标，就像狗追自己的尾巴

DQN 用**经验回放**解决第 1 个问题，用**目标网络**解决第 2 个问题。这两个技术至今仍是深度 RL 的标配。

#### 直观类比：DQN = Q-Learning + 记忆系统 + 参考系

- **Q-Learning（大脑）**：负责学习"什么动作好"
- **经验回放（记忆）**：把过去的经验存起来，随机抽取回忆
- **目标网络（参考系）**：提供一个稳定的"评分标准"，不让标准随学习而漂移

#### 核心公式

**TD 目标**：

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

**损失函数**（均方误差）：

$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(y - Q_\theta(s, a)\right)^2\right]$$

**逐项解读**：

- $Q_{\theta^-}$：用**目标网络**（不是在线网络）评估下一状态的 Q 值 —— 这是关键
- $\mathcal{D}$：经验回放缓冲区，从中随机采样
- 损失就是让在线网络的预测 $Q_\theta(s,a)$ 逼近 TD 目标 $y$

#### 代码实现（带逐行注释）

DQN 更新逻辑在 `DQN.update()` 方法中（[`01_dqn/dqn.py`](01_dqn/dqn.py)，第 113–154 行）：

```python
# ====== 用目标网络计算 TD 目标 ======
with torch.no_grad():                                # 目标网络不计算梯度
    next_q_values = self.target_network(next_states)  # 目标网络评估 s' 的 Q 值
    max_next_q = next_q_values.max(1)[0]              # 取每个状态的最大 Q 值
    # TD 目标 = 当前奖励 + γ × 下一个状态的最大 Q 值
    # (1 - dones) 确保终止状态没有"未来"
    target_q = rewards + self.gamma * max_next_q * (1 - dones)

# ====== 用在线网络计算当前 Q 值 ======
q_values = self.q_network(states)                    # 在线网络评估当前状态
q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # 取出实际执行动作的 Q 值

# ====== 计算 MSE 损失并更新 ======
loss = F.mse_loss(q_value, target_q)                 # 让在线 Q 值逼近目标 Q 值
```

> **关键观察**：`self.target_network(next_states)` 使用**目标网络**来评估下一状态的 Q 值，而 `self.q_network(states)` 使用**在线网络**来评估当前状态的 Q 值。目标网络本身不接收梯度（`torch.no_grad()`）。

**目标网络同步**（第 143–146 行）：

```python
self.update_count += 1
if self.update_count % self.target_update == 0:
    # 每 target_update 步，将在线网络权重完整复制到目标网络
    self.target_network.load_state_dict(self.q_network.state_dict())
```

每 `target_update` 步将在线网络权重完整复制到目标网络（**硬更新，Hard Update**）。

---

> **常见误区 1：把 DQN 当成万能算法**

DQN 在 Atari 游戏上表现惊艳，但它只适用于**离散动作空间**（因为需要计算 $\max_a Q(s,a)$）。对于连续动作空间（如机器人控制），DQN 无法直接使用。这就是为什么后续需要 DDPG、SAC 等算法。

---

> **检查点 3**：如果去掉 DQN 中的经验回放（直接从环境中顺序采样），会发生什么问题？如果去掉目标网络呢？

---

### 3.2 Double DQN (2015) —— 解决 Q 值过高估计

**论文**：*Deep Reinforcement Learning with Double Q-learning* -- van Hasselt et al., 2015

#### 过渡：DQN 有什么问题？

DQN 在 Atari 游戏上表现很好，但它继承了 Q-Learning 的一个老问题：**Q 值过高估计（Overestimation）**。

回忆 DQN 的 TD 目标：$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$。问题出在 $\max$ 操作上：

- 同一个网络既选动作（"哪个动作最好？"），又评估该动作（"这个动作值多少？"）
- 网络对某些动作的 Q 值估计会有误差（噪声）。$\max$ 操作总是挑选**被高估**的动作
- 被高估的 Q 值又会通过 TD 更新传播到其他状态，造成系统性高估

#### 直观类比：裁判同时是选手

想象一场选秀比赛，评委给选手打分。但如果**评委自己也参赛**，而且评分最高的人自动获胜……评委很可能给自己打最高分。这就是 DQN 中 $\max$ 操作的本质问题——**"裁判兼选手"的利益冲突**。

#### 核心思想：动作选择与价值评估解耦

Double DQN 的方案很简单——**让两个不同的"人"分别负责选动作和评价值**：

$$y = r + \gamma\, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$$

- 用**在线网络** $Q_\theta$ 选择最优动作（动作选择——"谁最好？"）
- 用**目标网络** $Q_{\theta^-}$ 评估该动作的 Q 值（价值评估——"他值多少？"）

这就像让裁判 A 选出最佳选手，然后让裁判 B 给这个选手打分——裁判之间互相制衡。

#### 代码实现（带逐行注释）

Double DQN 的 TD 目标计算在 ([`02_double_dqn/double_dqn.py`](02_double_dqn/double_dqn.py)，第 107–115 行）：

```python
with torch.no_grad():
    # 第1步：用在线网络选出最佳动作（"谁是下一个状态中最好的动作？"）
    online_next_q = self.q_network(next_states)
    best_actions = online_next_q.argmax(dim=1, keepdim=True)

    # 第2步：用目标网络评估该动作的价值（"这个动作值多少？"）
    target_next_q = self.target_network(next_states)
    max_target_q = target_next_q.gather(1, best_actions).squeeze(1)

    # 第3步：构造 TD 目标
    target_q = rewards + self.gamma * max_target_q * (1 - dones)
```

> **对比标准 DQN**：标准 DQN 只用目标网络做 `max`（裁判兼选手），Double DQN 显式分离了「谁选动作」（在线网络）和「谁估值」（目标网络），从而有效抑制 Q 值过高估计。

---

> **常见误区 2：Double DQN 需要额外训练一个新网络？**

不需要！Double DQN **复用已有的在线网络和目标网络**，只是改变了 TD 目标的计算方式。它不需要额外的网络参数或训练开销——仅仅是用在线网络来选动作而已。

---

> **检查点 4**：如果在线网络和目标网络的参数完全相同，Double DQN 的公式会退化成什么？它还能解决过高估计问题吗？

---

### 3.3 Dueling DQN (2016) —— 拆分状态价值与动作优势

**论文**：*Dueling Network Architectures for Deep Reinforcement Learning* -- Wang et al., 2016

#### 过渡：为什么需要新的网络架构？

Double DQN 改进了**目标计算方式**，但没有改变**网络结构**。Dueling DQN 从一个不同角度切入：有些状态下，**动作的选择根本不重要**。

想象你在玩一个游戏，当前画面是一片空白（还没出敌人）。无论你按左还是按右，都不会影响后续发展。这种情况下，Q(s, 左) 和 Q(s, 右) 应该几乎相等——状态本身的价值 $V(s)$ 就已经足够描述情况了，每个动作的"独特贡献"接近于零。

Dueling DQN 正是利用了这个洞察。

#### 直观类比：房子的"地段价值" vs. "装修加分"

评价一套房子：
- **地段价值 $V(s)$**：房子所在的区域、交通、学区——这是房子的"基础价值"，与你怎么使用它无关
- **装修加分 $A(s,a)$**：精装修 vs. 毛坯房、带花园 vs. 没花园——这是你针对这套房子的"特定选择"带来的附加值

总价 = 地段价值 + 装修加分。Dueling DQN 把 Q 值做了同样的拆分！

#### 核心公式

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')$$

**逐项解读**：

- $V(s)$：**状态价值流** —— 状态本身的好坏，与动作无关
- $A(s, a)$：**动作优势流** —— 每个动作比"平均水平"好多少
- $-\frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')$：减去优势的均值（原因见下）

**为什么要减去平均值？** 因为给定 $Q = V + A$，有多种方式拆分 $V$ 和 $A$。例如：
- 方案 1：$V = 5, A = 2$  →  $Q = 7$
- 方案 2：$V = 6, A = 1$  →  $Q = 7$

两种方案给出相同的 Q，但 $V$ 和 $A$ 不同——这就是**可辨识性（Identifiability）问题**。减去均值强制 $A$ 的均值为零，从而唯一确定 $V$ 和 $A$。

#### 代码实现（带逐行注释）

Dueling 架构的网络前向传播（[`03_dueling_dqn/dueling_dqn.py`](03_dueling_dqn/dueling_dqn.py)，第 53–60 行）：

```python
def forward(self, x):
    # 共享特征提取层（卷积/全连接）
    features = self.feature_layer(x)

    # V(s)：状态价值流 —— 输出1个值（这个状态本身有多好）
    value = self.value_stream(features)

    # A(s,a)：动作优势流 —— 输出action_dim个值（每个动作有多好）
    advantage = self.advantage_stream(features)

    # Q(s,a) = V(s) + [A(s,a) - mean(A)]  —— 减去均值保证可辨识性
    q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
    return q_values
```

> 网络结构体现了分解：`value_stream` 输出 **1 维标量**（$V(s)$），`advantage_stream` 输出 **action_dim 维向量**（$A(s,a)$）。这种架构的优点是即使某些动作的 $A(s,a)$ 没有学到，$V(s)$ 仍然可以为所有动作提供一个良好的基础估计。

---

> **常见误区 3：Dueling DQN 和 Double DQN 互斥？**

不互斥！它们是正交的改进：
- Double DQN 改进的是 **TD 目标的计算方式**（如何构造 $y$）
- Dueling DQN 改进的是 **网络架构**（如何计算 $Q(s,a)$）

二者可以组合使用，事实上很多实现中同时采用 Double + Dueling。

---

> **检查点 5**：设想一个状态，无论采取什么动作奖励都一样。在 Dueling DQN 中，$V(s)$ 和 $A(s,a)$ 分别会学到什么值？

---

### 3.4 A2C (Advantage Actor-Critic, 2016) —— Value + Policy 的融合

**论文**：*Asynchronous Methods for Deep Reinforcement Learning* -- Mnih et al., 2016

#### 过渡：从 Value-Based 回到 Policy-Based

DQN 系列算法（DQN、Double DQN、Dueling DQN）都属于 **Value-Based** 方法——先学 Q 值，再通过 $\arg\max$ 选动作。但第一章的 REINFORCE 告诉我们，还有另一条路：直接学习策略。

但是 REINFORCE 有问题：**方差太大**（MC 方法的通病），而且学完一局才能更新一次。

**A2C 是 REINFORCE 和 TD 思想的首次正式融合**：保留 REINFORCE 的策略梯度框架，同时引入一个 Critic 网络来提供低方差的信号。

#### 直觉理解：Actor-Critic 就像导演和影评人

- **Actor（演员）** = 策略网络 $\pi_\theta(a|s)$：负责"表演"——决定采取什么动作
- **Critic（评论家）** = 价值网络 $V_\phi(s)$：负责"评论"——评价当前状态的好坏

Actor 根据 Critic 的评价来改进自己的表演：如果某段表演获得了高于预期的评价（优势为正），就多演这样的戏；如果评价低于预期（优势为负），就改正。

#### 直观类比：Actor-Critic = 学员 + 教练

想象一个驾校学员（Actor）在练车：
- 学员完成了一段驾驶（执行了一系列动作）
- 教练（Critic）在旁边观察，给出评价："这一段的平均水平应该是 70 分，你刚才的表现是 85 分"
- 学员心想：我在这个路口的操作（状态-动作对）比平均水平高 15 分，以后要多这样开

优势函数 $A_t = G_t - V(s_t)$ 就是"教练的评语"——它衡量实际表现超出预期多少。

#### 核心公式

**优势函数**（Advantage Function）：

$$A_t = G_t - V(s_t)$$

- $G_t$：实际获得的 n-step 回报（或 MC 回报）
- $V(s_t)$：Critic 对当前状态价值的估计
- $A_t > 0$：动作比期望好，概率应被提升
- $A_t < 0$：动作比期望差，概率应被降低
- $A_t \approx 0$：动作与期望持平，概率基本不变

**三部分损失函数**：

$$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_v \mathcal{L}_{\text{critic}} - c_H \mathcal{L}_{\text{entropy}}$$

| 损失 | 公式 | 作用 |
|------|------|------|
| **Actor Loss** | $-\log\pi_\theta(a_t\|s_t) \cdot A_t$ | 推动策略向高优势方向更新 |
| **Critic Loss** | $\text{MSE}(V_\phi(s_t), G_t)$ | 训练价值估计的准确性 |
| **Entropy Bonus** | $-\sum_a \pi(a\|s)\log\pi(a\|s)$ | 鼓励策略多样性（熵越大越随机），防止过早收敛 |

> **为什么需要熵奖金？** 如果没有熵奖金，Actor 可能过早地对某个动作赋予接近 100% 的概率——即使那个动作只是当前"相对最优"而非"真正最优"。熵奖金强制策略保持一定随机性，给探索留下空间。

#### 代码实现（带逐行注释）

A2C 的完整更新逻辑（[`04_a3c/a2c.py`](04_a3c/a2c.py)，第 111–156 行）：

```python
# ====== 第1步：计算优势 ======
advantages = returns - values                        # A_t = G_t - V(s_t)
# 标准化优势（减去均值，除以标准差）—— 减小方差，加速收敛
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# ====== 第2步：Actor 损失 ======
# .detach() 很关键：优势不向 Critic 反向传播
actor_loss = (-log_probs * advantages.detach()).mean()

# ====== 第3步：Critic 损失 ======
# 让 V(s) 尽可能逼近实际回报 G
critic_loss = F.mse_loss(values, returns)

# ====== 第4步：熵损失 ======
probs = F.softmax(logits, dim=-1)
# 熵 = -Σ p * log(p)，衡量分布的不确定性
entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

# ====== 第5步：总损失 ======
# 注意：负号 — 熵越大越好（我们希望最大化熵），所以在损失中减去
loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
```

> **关键细节**：`advantages.detach()` 确保优势不向 Critic 反向传播——Critic 只通过 `critic_loss` 学习。而 `-self.entropy_coef * entropy` 是加熵奖金（减去负的熵 = 最大化熵）。

---

> **常见误区 4：混淆 Actor Loss 和 Critic Loss 的流向**

Actor 通过优势来学习（优势来自 Critic），但 Critic **不**通过优势来学习。`.detach()` 切断了这个梯度流。Critic 只通过自己的 MSE 损失来优化。如果忘记 `.detach()`，会导致两个网络纠缠在一起，训练不稳定。

---

> **检查点 6**：A2C 相比 REINFORCE 有哪两个关键改进？A2C 为什么比 REINFORCE 方差更低？

---

### 3.5 PPO (Proximal Policy Optimization, 2017) -- 最常用的 RL 算法

**论文**：*Proximal Policy Optimization Algorithms* -- Schulman et al., OpenAI, 2017

#### 过渡：A2C 有什么不足？

A2C 使用的是 **On-Policy** 数据：策略更新后，所有旧数据就失效了，必须用新策略重新采样。这导致**样本效率极低**——每条数据只用一次就被丢弃。

你可能会想："那能不能对同一批数据做多轮训练？" 理论上可以，但风险很大：如果一次更新太大，策略可能"崩溃"——从一个有效的策略跳到一个完全无效的策略。

**PPO 的核心创新**：通过**裁剪（Clipping）**限制策略更新幅度，让你可以安全地对同一批数据做多轮训练，大幅提高样本效率。

#### 直观类比：给方向盘装限位器

想象你在学开车，教练告诉你"方向盘打得太猛会翻车"。普通的策略更新就像猛打方向盘——可能直接冲出赛道。PPO 的做法是给方向盘装一个**限位器**：最多只能打 30 度，无论如何不会超出这个范围。这样你就可以在安全范围内反复练习同一个弯道。

#### 重要性采样比率

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

- $r_t > 1$：新策略提升了该动作的概率（"我更倾向于做这个动作了"）
- $r_t < 1$：新策略降低了该动作的概率（"我不太想做这个动作了"）
- $r_t = 1$：新旧策略对这个动作的态度一样

#### 裁剪目标函数（渐进式理解）

**第 1 步：朴素的想法 —— 直接用比率乘优势**

$$L^{\text{naive}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \hat{A}_t\right]$$

问题是：当 $r_t$ 很大时，一次更新可能把策略推向极端。

**第 2 步：加一个裁剪（Clipping）—— PPO 的核心创新**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t\right)\right]$$

**逐项解读**：

- $r_t(\theta) \hat{A}_t$：朴素的想法（无限制更新）
- $\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t$：被限制的更新
- $\min$：取两者中更保守（更接近零）的一项

**分情况理解**：

| 情况 | 优势 $\hat{A}_t$ | 比率 $r_t$ | 裁剪后效果 |
|------|-----------------|-----------|-----------|
| 好动作，想提升 | $> 0$ | $> 1$ | 被限制在 $1+\epsilon$（别太夸张） |
| 好动作，想降低 | $> 0$ | $< 1$ | 不限制（降低好动作概率本身就是保守的） |
| 坏动作，想降低 | $< 0$ | $< 1$ | 被限制在 $1-\epsilon$（别打压太狠） |
| 坏动作，想提升 | $< 0$ | $> 1$ | 不限制（提升坏动作概率本身就是保守的） |

#### GAE (Generalized Advantage Estimation)

PPO 使用 **GAE** 来平滑地权衡偏差与方差（[`05_ppo/ppo.py`](05_ppo/ppo.py)，第 147–162 行）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

**直觉理解**：GAE 是一个"时间加权平均"——离当前步越远的 TD 误差，其权重越低。参数 $\lambda$ 控制衰减速度：

- $\lambda = 0$：退化为一步 TD 误差（低方差、高偏差 —— "只看眼前"）
- $\lambda = 1$：退化为 MC 优势估计（高方差、低偏差 —— "看整局"）
- 常用 $\lambda = 0.95$：折中方案

#### 代码实现（带逐行注释）

**GAE 计算**（[`05_ppo/ppo.py`](05_ppo/ppo.py)，第 147–162 行）：

```python
def _compute_gae(self, last_value=0.0):
    values = np.array(self.values + [last_value])
    dones = np.array(self.dones + [0])
    rewards = np.array(self.rewards)

    advantages = np.zeros(len(rewards))
    gae = 0

    # 从后向前递推计算 GAE
    for t in reversed(range(len(rewards))):
        # TD 误差：r + γV(s') - V(s)
        delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
        # GAE 递推：当前 δ + γλ × 上一步的 GAE
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # 回报 = 优势 + 价值估计（用于 Critic 训练）
    returns = advantages + values[:-1]
    return advantages, returns
```

> **关键细节**：反向遍历（`reversed`）+ 递推公式。每一步的 GAE 同时包含了当前 TD 误差和以 $\gamma\lambda$ 为折扣的未来误差。

**裁剪目标实现**（[`05_ppo/ppo.py`](05_ppo/ppo.py)，第 207–213 行）：

```python
# 重要性采样比率：新策略概率 / 旧策略概率
ratio = (new_log_probs - batch_log_probs).exp()
# 为什么用 exp(差值)？因为 log(a/b) = log(a)-log(b)，取 exp 后得到 a/b

# 裁剪目标：取"无限制版本"和"裁剪版本"的较小值
surr1 = ratio * batch_adv                                 # 朴素目标
surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv  # 裁剪目标
policy_loss = -torch.min(surr1, surr2).mean()             # 取 min 后再取平均，加负号转为最小化
```

> **为什么是 $\min$？** 当 $\hat{A} > 0$，$\min$ 限制过度乐观；当 $\hat{A} < 0$，$\min$ 限制过度悲观。本质上是在"别太高兴"和"别太沮丧"之间取得平衡。

> PPO 最终的总损失还包含价值损失和熵奖金（第 219 行），与 A2C 类似但增加了裁剪机制。本实现支持多轮 PPO 更新（`ppo_epochs` 参数，第 183 行），这是 PPO 相比 A2C 提高样本效率的关键。

---

> **常见误区 5：PPO 的裁剪参数 $\epsilon$ 设太大或太小**

- $\epsilon$ 太大（如 0.5）：裁剪几乎无效，PPO 退化为无限制的 A2C 多轮更新，可能导致策略崩溃
- $\epsilon$ 太小（如 0.01）：策略更新过于保守，学习速度极慢
- 论文推荐 $\epsilon = 0.2$，这是一个经过大量实验验证的好起点

---

> **常见误区 6：混淆 PPO Epochs 和更新次数**

初学者常问："PPO 不是多轮更新吗，为什么还要收集新数据？" PPO 的 `ppo_epochs`（如 10）指的是**对同一批数据**重复训练 10 轮。10 轮之后，必须用更新后的策略重新收集一批新数据，因为 PPO 本质上仍是 On-Policy（虽然它比 A2C 更接近 Off-Policy）。

---

> **检查点 7**：PPO 的裁剪目标中 $\min$ 操作的作用是什么？如果去掉 $\min$（只保留 $r_t \hat{A}_t$），PPO 会退化成什么？

---

### 3.6 SAC (Soft Actor-Critic, 2018) —— 最大熵强化学习

**论文**：*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* -- Haarnoja et al., 2018

#### 过渡：把前面学到的全部融合

如果你已经理解了 DQN（Off-Policy + 经验回放）、Double DQN（双 Q）、A2C（Actor-Critic）、PPO（稳定更新），那么 SAC 就是这些思想的"终极融合"：

| SAC 的特性 | 来源 |
|-----------|------|
| Off-Policy + 经验回放 | DQN |
| Actor-Critic 架构 | A2C / PPO |
| 双 Q 网络 | Double DQN |
| 熵最大化 | SAC 独创 |
| 自动温度调节 | SAC 独创 |
| 软目标更新 | SAC 的改进 |

#### 直觉理解：SAC 想解决什么问题？

**核心矛盾**：RL 智能体需要平衡"做已知的好事（利用）"和"尝试未知的可能（探索）"。传统方法用 epsilon-greedy 或熵奖金来做探索，但这些都是"外加"的机制。

**SAC 的哲学**：把"探索"写入算法的 DNA 中。不是"回报最大化 + 偶尔探索"，而是"回报 + 随机性 一起最大化"。

#### 直观类比：SAC = 带着好奇心去旅行

- 传统 RL：去一个城市旅游，只去攻略上评分最高的景点（**纯利用**）
- SAC：去旅游时，既去评分高的景点（**高回报**），也保持好奇心去探索未知小巷（**高熵**），而且有一个自动调节的"好奇心强度"（**自动温度调节**）

#### 最大熵目标

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha\, \mathcal{H}\big(\pi(\cdot \mid s_t)\big)\right]$$

其中 $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$ 是策略在状态 $s_t$ 处的**熵**。$\alpha$ 是**温度参数**，控制熵项的重要性。

**解读**：目标 = 最大化回报 + $\alpha$ x 最大化策略的随机性。$\alpha$ 大 → 好奇心强（多探索）；$\alpha$ 小 → 目标导向（多利用）。

#### SAC 的三大关键组件

**组件 1：软 Q 函数（双 Q + 熵奖励）**

使用两个独立的 Q 网络（$Q_{\theta_1}$ 和 $Q_{\theta_2}$），取最小值来抑制过高估计：

$$y = r + \gamma \left(\min_{i=1,2} Q_{\theta_i^-}(s', a') - \alpha \log \pi_\phi(a' \mid s')\right), \quad a' \sim \pi_\phi(\cdot \mid s')$$

注意新加的 $-\alpha \log \pi_\phi(a'|s')$ 项——这是 SAC 的"软"之处：**在 Q 目标中直接加入熵奖励**。如果策略在 $s'$ 处很确定（低熵），这一项就是惩罚。

**组件 2：软策略更新**

最小化 KL 散度，等价于最大化 Q 值同时保持高熵：

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{s\sim\mathcal{D}}\left[\alpha \log \pi_\phi(a \mid s) - \min_{i=1,2} Q_{\theta_i}(s, a)\right], \quad a \sim \pi_\phi(\cdot \mid s)$$

**组件 3：自动温度调节**

将 $\alpha$ 也作为可学习参数，在训练中动态调整以匹配目标熵：

$$\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi}\left[-\alpha \left(\log \pi(a \mid s) + \bar{\mathcal{H}}\right)\right]$$

其中 $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ 是目标熵（通常设为动作空间维度的负数）。

**直觉**：如果探索不够（当前熵 < 目标熵），自动增大 $\alpha$ 来鼓励更多探索；如果探索太多，自动减小 $\alpha$。

#### 代码实现（带逐行注释）

SAC 的完整更新在 `SAC.update()` 方法中（[`06_sac/sac.py`](06_sac/sac.py)，第 159–218 行），分为四个子步骤：

**Step 1 -- Critic 更新（第 171–184 行）**：

```python
with torch.no_grad():
    # 从当前策略采样下一个动作（注意：不是 argmax，而是采样）
    next_actions, next_log_probs, _ = self.policy.sample(next_states)
    # 两个目标 Q 网络各自评估
    target_q1 = self.q1_target(next_states, next_actions)
    target_q2 = self.q2_target(next_states, next_actions)
    # 双Q取最小 + 熵奖励（-alpha * log_prob 即加熵）
    target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
    # TD 目标
    target_q = rewards + self.gamma * (1 - dones) * target_q

# 两个 Q 网络都向同一个 TD 目标回归
q1_loss = F.mse_loss(self.q1(states, actions), target_q)
q2_loss = F.mse_loss(self.q2(states, actions), target_q)
```

> `torch.min(target_q1, target_q2)` 体现了双 Q 网络的核心：取两个 Q 网络的最小值来抑制过高估计。

**Step 2 -- Actor 更新（第 187–196 行）**：

```python
# 从当前策略重新采样动作（重参数化技巧）
sampled_actions, log_probs, _ = self.policy.sample(states)
q1_new = self.q1(states, sampled_actions)
q2_new = self.q2(states, sampled_actions)
q_new = torch.min(q1_new, q2_new)

# 策略损失：最大化 Q 值（-q_new）同时惩罚低熵（+alpha*log_probs）
policy_loss = (self.alpha * log_probs - q_new).mean()
```

> `self.alpha * log_probs - q_new`：$q$ 越大越好（所以取负号），$\log \pi$ 越小（低熵）越不好（所以取正号）。

**Step 3 -- Alpha 自动调节（第 199–205 行）**：

```python
# 当策略熵低于目标熵时（探索不足），降低 alpha_loss 以增大 alpha
alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
self.alpha_optimizer.zero_grad()
alpha_loss.backward()
self.alpha_optimizer.step()
# α 必须为正，通过对数参数化保证：α = exp(log_alpha) > 0
self.alpha = self.log_alpha.exp().item()
```

> 使用 $\log\alpha$（而非 $\alpha$）作为可学习参数：保证 $\alpha = e^{\log\alpha} > 0$，同时让梯度在 $\alpha$ 接近零时仍然有效。

**Step 4 -- 软更新目标网络（第 208–211 行）**：

```python
# 软更新：θ^- ← τ*θ + (1-τ)*θ^-
# τ 很小（如 0.005），目标网络平滑漂移而非硬拷贝
for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

> SAC 使用**软更新（Soft Update）**而非硬拷贝：$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$，其中 $\tau = 0.005$。这使目标网络平滑漂移，比 DQN 的定期复制更加稳定。

---

> **常见误区 7：SAC 只能用于连续动作空间？**

SAC 的原始设计和最广为人知的应用确实在连续控制领域，但 SAC 也有离散动作的变体。关键区别在于：连续动作的 SAC 通过重参数化技巧（Reparameterization Trick）采样动作并计算梯度；离散动作的 SAC 则需要对所有动作计算期望。初学者在换环境时应注意这个区别。

---

> **常见误区 8：把 SAC 的"软更新"理解为"比硬更新差"**

"软更新"（用 $\tau = 0.005$）看起来非常慢，但它在实践中比硬更新（如 DQN 那样每 N 步完整复制）**更稳定**。软更新相当于目标网络以指数移动平均的方式跟踪在线网络，避免了硬拷贝带来的目标突变。

---

> **检查点 8**：SAC 中的熵奖励项出现在哪里？（提示：至少有两个地方）如果温度 $\alpha$ 一直很大，SAC 最终会学到什么样的策略？

---

## 4. A2C 与 PPO 的关系 —— 一张图看懂

```
REINFORCE (1992)
    │
    │  加入 Critic 基线减少方差
    ▼
A2C (2016)
    │                ┌──────────────────────────────────┐
    │                │  PPO 在 A2C 基础上新增：          │
    │                │  * Clipped Surrogate Objective   │
    │                │  * 多轮 PPO Epochs               │
    │                │  * GAE 优势估计                   │
    ▼                │  * 梯度裁剪（Gradient Clipping）  │
PPO (2017) ─────────┘                                   │
                       └──────────────────────────────────┘
```

| 特性 | A2C | PPO |
|------|-----|-----|
| 策略更新次数 | 每条数据只用一次 | 同批数据可迭代多次（PPO Epochs） |
| 更新限制 | 无显式限制 | 裁剪比率 / KL 约束 |
| 优势估计 | 简单 n-step 或 MC 优势 | GAE ($\lambda$-加权) |
| 稳定性 | 对超参数敏感 | 超参数鲁棒 |
| 样本效率 | 较低 | 中等 |
| 训练速度 | 较快（单轮更新） | 较慢（多轮更新） |

**核心区别**：A2C 是 PPO 的基础框架，PPO 在 A2C 的 Actor-Critic 架构上增加了**裁剪目标**和**多轮优化**，从而在稳定性和样本效率之间达到了极佳的平衡。如果把 A2C 比作"一笔一画地写字"，PPO 就是"先写完，再反复擦拭打磨"。

---

## 5. SAC 的双 Q 网络与温度自动调节

### 5.1 为什么需要双 Q 网络？

Q-Learning 及其深度版本都面临 Q 值过高估计的问题。SAC 沿用 Double DQN 的思路，但做了一步改进：

| 方法 | 选动作 | 估值 | 问题 |
|------|--------|------|------|
| DQN | 目标网络（用 max） | 目标网络 | 裁判兼选手，过高估计 |
| Double DQN | 在线网络 | 目标网络 | 仅解决了一部分 |
| SAC 双 Q | 不选动作（采样） | 两个目标网络取 min | 最稳定 |

取最小值的思想源自 Clipped Double Q-learning：即使其中一个网络对某个动作的 Q 值估计偏高，取 min 可以保证目标值不会被过度乐观的估计带偏。实验证明这比 Double DQN 的解耦策略在连续控制任务中更稳定。

### 5.2 温度参数 $\alpha$ 的自动调节

温度 $\alpha$ 控制"回报最大化"与"熵最大化"之间的平衡：

- $\alpha$ 大：鼓励高熵 → 策略更随机 → 更多探索 → 但可能以牺牲回报为代价
- $\alpha$ 小：接近标准 RL → 策略更确定 → 可能过早收敛到次优策略

手动调 $\alpha$ 很麻烦（不同环境最佳值不同，训练过程中最优值也可能变化）。SAC 将 $\alpha$ 也作为一个可学习参数，通过最小化以下损失来自动调节：

$$\mathcal{L}(\alpha) = -\alpha \big(\log\pi(a|s) + \bar{\mathcal{H}}\big)$$

目标熵 $\bar{\mathcal{H}}$ 通常设为 $-\dim(\mathcal{A})$（离散动作空间）。

在代码中（[`06_sac/sac.py`](06_sac/sac.py)，第 127–139 行）：

```python
if target_entropy is None:
    self.target_entropy = -action_dim    # 默认目标熵 = -动作空间维度

# 使用 log(α) 作为可学习参数（保证 α > 0）
self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
```

使用 $\log\alpha$（而非 $\alpha$）作为可学习参数可以保证 $\alpha = e^{\log\alpha} > 0$，同时让梯度在 $\alpha$ 接近零时仍然有效。

---

> **常见误区 9：忘记软更新中对第二个 Q 网络也做更新**

SAC 有两个 Q 网络，它们的**目标网络**都需要软更新。初学者常只更新其中一个目标网络，导致另一个目标网络参数陈旧。

---

## 6. 运行方法

每个子目录含 `train.py` 训练脚本。DQN 系列算法默认使用 CartPole 环境，SAC 默认使用 Pendulum 环境：

```bash
# DQN
cd chapter_02_deep_rl/01_dqn
python train.py

# Double DQN
cd chapter_02_deep_rl/02_double_dqn
python train.py

# Dueling DQN
cd chapter_02_deep_rl/03_dueling_dqn
python train.py

# A2C
cd chapter_02_deep_rl/04_a3c
python train.py

# PPO
cd chapter_02_deep_rl/05_ppo
python train.py

# SAC
cd chapter_02_deep_rl/06_sac
python train.py
```

运行结果（奖励曲线、训练时长等）将打印到控制台，推荐截图保存于各子目录下作为学习记录。

---

> **终极检查点**：
> 1. 从 DQN → Double DQN → Dueling DQN，每一步改进分别解决了什么问题？
> 2. A2C 到 PPO 的核心改进是什么？为什么这个改进能提高样本效率？
> 3. SAC 融合了前面哪些算法的思想？"Soft"的核心含义是什么？

---

## 7. 学习路线建议

1. **先彻底理解 DQN 的三大组件**：经验回放、目标网络、神经网络函数逼近。这是整个深度 RL 的基石。问自己：如果把这三个组件各去掉一个，分别会发生什么？
2. **逐个对比 DQN → Double → Dueling**：后两者只是在前者基础上改动了网络结构或目标计算方式，代码差异极小，对比阅读十分高效。
3. **从 A2C 过渡到 PPO**：理解了 A2C 的 Actor-Critic 框架和损失构成后，PPO 只是在外面包了一层裁剪机制。建议先跑通 A2C，再对比 PPO 的训练曲线。
4. **最后学习 SAC**：SAC 融合了 Off-Policy、Actor-Critic、最大熵等概念，是最复杂的算法之一。确保理解前面的所有算法再来看 SAC，会发现它其实是"集大成者"。
5. **动手调参**：尝试修改 `gamma`、`batch_size`、`clip_epsilon`（PPO）、`alpha`（SAC）等参数，观察对训练的影响。调参是培养算法直觉的最好方式。
6. **看训练曲线，不要只看最终分数**：训练过程中的曲线（奖励的震荡、收敛速度、是否发散）比最终分数更能反映算法的特性。
