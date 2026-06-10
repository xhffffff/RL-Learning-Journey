# 第一章：强化学习基础算法 (1989–1994)

本章涵盖强化学习（Reinforcement Learning, RL）的数学基础与三个里程碑式的基础算法：**Q-Learning (1989)**、**SARSA (1994)**、**REINFORCE (1992)**。它们分别代表了基于价值（Value-Based）与基于策略（Policy-Based）的两大范式，是理解后续所有深度强化学习方法的基石。

> **本章学习目标**：读完本章后，你应该能够：(1) 独立推导 Bellman 方程；(2) 手写 Q-Learning 和 SARSA 的更新公式并说出它们的区别；(3) 理解策略梯度定理的直观含义。

---

## 1. 强化学习基础理论

### 1.1 马尔可夫决策过程（Markov Decision Process, MDP）

#### 直觉理解

想象你正在玩一个迷宫游戏。你站在某个位置（这就是**状态**），你可以选择向上、下、左、右走（这就是**动作**）。每走一步，你可能：
- 走近出口（得到**正奖励**）
- 撞到墙上（得到**负奖励**）
- 什么都没发生（奖励为零）

走完之后，你来到了新的位置（**状态转移**）。你的目标是：从起点出发，选择一系列动作，使得一路上获得的总奖励最大。

这个"迷宫游戏"的数学模型，就是**马尔可夫决策过程（MDP）**。它是所有强化学习问题的通用语言。

#### 直观类比：MDP 就像一份"游戏规则说明书"

| MDP 元素 | 棋盘游戏类比 | 迷宫类比 |
|----------|------------|---------|
| **状态** (State) | 棋盘上棋子的位置 | 你在迷宫中的坐标 |
| **动作** (Action) | 可以走的棋步 | 上/下/左/右 |
| **状态转移** (Transition) | 走完一步后棋子的新位置 | 移动后的新坐标 |
| **奖励** (Reward) | 吃掉对方棋子得分 | 走近出口 +1，撞墙 -1 |
| **折扣因子** (Discount) | 你是否更看重眼前的优势 | 远期的奖励打几折 |

#### 形式化定义

RL 问题通常被形式化为 **MDP**，由一个五元组定义：

$$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 状态空间（State Space）— 所有可能状态的集合 |
| $\mathcal{A}$ | 动作空间（Action Space）— 所有可能动作的集合 |
| $\mathcal{P}$ | 状态转移概率 $p(s'\|s, a)$ — 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率 |
| $\mathcal{R}$ | 奖励函数 $r(s, a)$ 或 $r(s, a, s')$ — 立即反馈信号 |
| $\gamma$ | 折扣因子（Discount Factor）$\in [0, 1]$ — 权衡当前奖励与未来奖励的重要性 |

智能体（Agent）在每一步与环境交互：观察状态 $s_t$ → 选择动作 $a_t$ → 接收奖励 $r_{t+1}$ → 转移到新状态 $s_{t+1}$。这个过程产生一条轨迹（Trajectory）：

$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)$$

> **关键理解**："马尔可夫"意味着**未来只取决于现在，与过去无关**。给定当前状态 $s_t$，之前的历史 $s_0, a_0, \dots, s_{t-1}$ 不会提供任何额外信息。这就像下象棋——你只需要看当前的棋盘，不需要知道之前每一步是怎么走的。

---

> **检查点 1**：MDP 五元组的每个元素分别代表什么？如果你要为一个"自动售货机"建立 MDP 模型，状态、动作和奖励分别是什么？

---

### 1.2 回报（Return）

#### 直觉理解

在迷宫中，你每走一步可能得到不同的奖励。"回报"就是把从当前时刻开始，未来所有奖励加起来的总和。但这里有个关键问题：**明天的 1 块钱和今天的 1 块钱，你认为价值一样吗？**

大多数人都会认为"今天的 1 块钱更值钱"——这就是**折扣因子** $\gamma$ 的作用。

#### 形式化定义

**回报** $G_t$ 是从时刻 $t$ 开始的所有未来**折扣奖励**之和：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

#### 渐进式理解折扣因子

- 当 $\gamma = 0$ 时：$G_t = R_{t+1}$，智能体**只看眼前一步**（极度短视/"今朝有酒今朝醉"）
- 当 $\gamma = 0.5$ 时：$G_t = R_{t+1} + 0.5 R_{t+2} + 0.25 R_{t+3} + \dots$，越远的奖励衰减越快（"有点远见"）
- 当 $\gamma = 0.99$ 时：$G_t \approx R_{t+1} + 0.99 R_{t+2} + 0.98 R_{t+3} + \dots$，远期的奖励也很重要（"深谋远虑"）
- 当 $\gamma = 1$ 时：所有奖励同等重要，仅适用于有限步长的任务（"全部打包带走"）

对于有限步长（episodic）任务，不需要折扣时可以直接使用 $\gamma = 1$，此时 $G_t = \sum_{k=0}^{T-t} R_{t+k+1}$。

> **检查点 2**：如果 $\gamma = 0.9$，第 10 步之后的奖励被打了多少"折扣"？为什么在无限步长的任务中不能设 $\gamma = 1$？

---

### 1.3 策略（Policy）

#### 直觉理解

**策略**就是智能体的"行为准则"。在迷宫的每个位置，策略告诉智能体应该往哪走。策略可以很确定（"在 A 点一定向右走"），也可以有概率（"在 A 点有 70% 概率向右，30% 概率向下"）。

#### 形式化定义

**策略** $\pi$ 定义了智能体在给定状态下选择动作的方式：

- **确定性策略**：$\pi(s) = a$ —— 给定状态，输出唯一的动作
- **随机策略**：$\pi(a|s) = \mathbb{P}[A_t = a \mid S_t = s]$ —— 给定状态，输出动作的概率分布

> **为什么需要随机策略？** 在石头剪刀布这样的博弈中，确定性策略（总是出石头）会被对手轻易利用。随机策略则可以避免被预测，同时在探索时也至关重要。

---

### 1.4 价值函数（Value Function）与动作价值函数（Action-Value Function）

#### 直觉理解

如果你是一个房产估价师，你需要回答两个问题：

1. "这套房子值多少钱？" —— 这就是**状态价值函数** $V(s)$：从当前状态出发，按某个策略走，能获得多少期望回报。
2. "如果我买了这套房子然后出租，值多少钱？" —— 这就是**动作价值函数** $Q(s, a)$：在某个状态下执行特定动作，然后按策略走，能获得多少期望回报。

#### 形式化定义

**状态价值函数** $V^\pi(s)$：从状态 $s$ 出发，遵循策略 $\pi$ 的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\middle|\, S_t = s\right]$$

**动作价值函数** $Q^\pi(s, a)$：从状态 $s$ 出发，执行动作 $a$，之后遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]$$

#### V 和 Q 的关系

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s)\, Q^\pi(s, a)$$

**解读**：状态价值 = 所有可能动作的 Q 值的加权平均（权重是策略给出的概率）。就像"这套房子的价值 = 自住的价值 x 自住的概率 + 出租的价值 x 出租的概率"。

> **检查点 3**：$V(s)$ 和 $Q(s, a)$ 的区别是什么？如果策略是确定性的（$\pi(s) = a_0$），$V(s)$ 和 $Q(s, a_0)$ 的关系是什么？

---

### 1.5 Bellman 方程 —— 强化学习的"牛顿定律"

#### 直觉理解

Bellman 方程是强化学习中最重要的公式。它的核心思想可以用一句话概括：

> **"当前的价值 = 当前这一步的奖励 + 未来的价值"**

想象你站在迷宫的一个岔路口：
- 往左走：立刻得到一个金币（+1），然后进入一个价值为 5 的区域
- 往右走：立刻得到两个金币（+2），然后进入一个价值为 3 的区域

如果 $\gamma = 0.9$：
- 往左的总价值 = 1 + 0.9 x 5 = 5.5
- 往右的总价值 = 2 + 0.9 x 3 = 4.7

所以你选择往左走！Bellman 方程就是把这种"一步奖励 + 未来价值"的直觉用数学精确表达出来。

#### 渐进式推导（从简单到复杂）

**第 1 步：从回报的定义出发**

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$

**第 2 步：提出一个 $\gamma$**

$$G_t = R_{t+1} + \gamma \big(R_{t+2} + \gamma R_{t+3} + \dots\big)$$

**第 3 步：发现括号里正是 $G_{t+1}$**

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**第 4 步：对两边取条件期望（在状态 $s$ 下）**

$$V^\pi(s) = \mathbb{E}_\pi\big[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\big]$$

这就是 **Bellman 期望方程**的核心思想！价值函数的当前值，等于"当前奖励 + 折扣后的下一个状态价值"的期望。

#### Bellman 期望方程（完整形式）

描述了价值函数在给定策略下的递归关系：

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma\, V^\pi(s')\right]$$

**逐项解读**：
- $\pi(a|s)$：在状态 $s$ 下选择动作 $a$ 的概率
- $p(s', r \mid s, a)$：在状态 $s$ 执行动作 $a$ 后，得到奖励 $r$ 并转移到 $s'$ 的概率
- $r + \gamma V^\pi(s')$：当前奖励 + 折扣后的未来价值
- 最外层的 $\sum_a \sum_{s',r}$：对所有可能的动作和结果取期望

对应的 **Q 函数版本**：

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s')\, Q^\pi(s', a')\right]$$

#### Bellman 最优方程

如果说 Bellman 期望方程回答的是"给定策略下，价值是多少"，那么 **Bellman 最优方程**回答的是"最优策略下，价值是多少"。

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma\, V^*(s')\right]$$

$$Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma\, \max_{a' \in \mathcal{A}} Q^*(s', a')\right]$$

**关键变化**：$\sum_a \pi(a|s)$ 变成了 $\max_a$ —— "按策略走"变成了"选最好的走"。

一旦求得 $Q^*$，最优策略即为 $\pi^*(s) = \arg\max_a Q^*(s, a)$，这也是 Q-Learning 的核心目标。

---

> **检查点 4**：Bellman 期望方程和 Bellman 最优方程的关键区别是什么？为什么最优方程中 $\max$ 操作可以替代策略的加权和？

---

### 1.6 时序差分（Temporal Difference, TD）与蒙特卡洛（Monte Carlo, MC）

#### 直觉理解

假设你在学习一个游戏的得分规律。有两种方法：

- **蒙特卡洛方法**：完整玩一局游戏，记录每一步的最终得分，然后回头分析哪一步走得好。优点：结果精确。缺点：必须等游戏结束才能学习。
- **时序差分方法**：每走一步就估计"这一步走得好不好"，不等游戏结束就更新。优点：实时学习。缺点：估计可能不准。

这就像：
- MC 像是考完试后对答案，知道自己每道题得了多少分
- TD 像是做题过程中根据"感觉"随时调整，不等到最后

| 特性 | 蒙特卡洛（MC） | 时序差分（TD） |
|------|---------------|---------------|
| 更新时机 | Episode 结束后 | 每一步 |
| 目标值 | 完整回报 $G_t$ | $r + \gamma V(s')$（有偏但低方差） |
| 偏差 | 无偏 | 有偏（Bootstrap） |
| 方差 | 高方差 | 低方差 |
| 需要 Episode 结束 | 是 | 否 |
| 代表算法 | REINFORCE | Q-Learning, SARSA |

**Bootstrap（自举）是什么意思？** TD 方法中，我们用 $V(s')$ 来估计 $V(s)$，即"用一个估计值来更新另一个估计值"，这就像"抓着鞋带把自己举起来"。这在数学上看似循环，但实际上 $r$（真实奖励）提供了真实信号，使得整个系统可以稳定收敛。

TD 方法因为每一步都可以更新，在持续任务和在线学习中更具优势；MC 方法无偏但方差高，适合简单环境下的策略梯度估计。

---

> **检查点 5**：为什么 MC 方法方差高但无偏，而 TD 方法方差低但有偏？"有偏估计"在什么情况下反而可能是优势？

---

## 2. 算法详解

### 概述：Value-Based 与 Policy-Based 两大范式

在进入具体算法之前，先建立整体框架。RL 基础算法可以分为两大类：

- **基于价值（Value-Based）**：学习 Q 函数 $Q(s, a)$，然后通过 $\arg\max_a Q(s, a)$ 间接得到策略。代表：Q-Learning、SARSA。
- **基于策略（Policy-Based）**：直接学习策略函数 $\pi(a|s)$，跳过 Q 函数的中间步骤。代表：REINFORCE。

Q-Learning 和 SARSA 都属于 Value-Based，但它们对"用什么策略来做探索"的处理方式截然不同——这就是 On-Policy 和 Off-Policy 的区别。

---

### 2.1 Q-Learning (1989) — Off-Policy 时序差分控制

**论文**：*Learning from Delayed Rewards* — Chris Watkins, 1989

#### 直觉理解：Q-Learning 想解决什么问题？

想象你是一个美食评论家，你的目标是给城市里的每家餐厅打分（这就是 Q 值）。每次你实际去一家餐厅吃饭，你会获得一个体验评分（奖励）。吃完后，你更新对这家餐厅的评价：

Q-Learning 的做法是：**不管你今天实际选择了哪家餐厅，你总是假设下次会去"评分最高的那家餐厅"**。这就是 Off-Policy 的核心——学习和行动可以分开。

#### 直观类比：试吃餐厅并更新评分手册

你有一本"餐厅评分手册"（这就是 **Q 表**），记录每家餐厅的评分。

1. **选餐厅**：你以 90% 的概率去评分最高的餐厅（**利用**），以 10% 的概率随机尝试一家新餐厅（**探索**）
2. **吃饭体验**：吃到美味 → 奖励 +1；吃到难吃 → 奖励 -1
3. **更新评分**：不管刚才怎么选餐厅的，你更新时总是假设："吃完这家餐厅后，我下一顿会去全城评分最高的那家" —— 这就是公式中的 $\max_{a'} Q(s', a')$
4. **重复**：日复一日，你的评分手册越来越准确

#### 核心公式

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

**逐项解读**（从直觉到数学）：

| 部分 | 数学符号 | 直觉含义 |
|------|---------|---------|
| 当前估计 | $Q(s, a)$ | 你认为这家餐厅当前有多好 |
| 学习目标 | $r + \gamma \max_{a'} Q(s', a')$ | 实际体验 + 假设下次去最好的餐厅 |
| 预测误差 | $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ | "现实"和"预期"之间的差距（**TD 误差**） |
| 学习率 | $\alpha$ | 你有多愿意根据新经验改变旧看法 |

- $\alpha = 1$：完全相信最新经验，抛弃旧估计（"一朝被蛇咬，十年不想吃"）
- $\alpha = 0$：完全不相信新经验，固守旧估计（"固执己见"）
- $\alpha = 0.1$：新旧兼顾，缓慢更新（通常的做法）

#### 代码实现（带逐行注释）

更新公式在 `QLearning.update()` 方法中实现（[`01_q_learning/q_learning.py`](01_q_learning/q_learning.py)，第 84–108 行）：

```python
def update(self, state, action, reward, next_state, done):
    # 情况1：如果已经到达终止状态（游戏结束）
    if done:
        target = reward  # 没有"未来"了，目标就是当前奖励本身
    # 情况2：正常状态转移
    else:
        # 核心：取下一个状态中所有动作的 Q 值中的最大值
        # 这就是 Off-Policy 的关键——假设接下来采取最优动作
        target = reward + self.gamma * np.max(self.q_table[next_state])

    # 计算 TD 误差："现实"减去"预期"
    td_error = target - self.q_table[state, action]

    # 用 TD 误差和学习率更新 Q 值
    # 如果 td_error > 0，说明实际比预期好，Q 值上调
    # 如果 td_error < 0，说明实际比预期差，Q 值下调
    self.q_table[state, action] += self.lr * td_error

    return abs(td_error)  # 返回误差大小，可用于监控训练
```

> **关键点**：`np.max(self.q_table[next_state])` —— 直接取下一个状态下所有动作的最大 Q 值，而非实际执行的动作。这正是 **Off-Policy** 的本质：更新时总是假设接下来采取最优动作。

#### Off-Policy（离策略）本质

Q-Learning 是 **Off-Policy** 的，意味着学习的目标策略和实际执行的策略可以不同：

- **行为策略（Behavior Policy）**：使用 epsilon-greedy 与环境交互（可能采取非最优动作以探索）
- **目标策略（Target Policy）**：总是选择 $\arg\max_a Q(s, a)$（完全贪婪）

这种"学一套、做一套"的分离有什么好处？你可以用别人（甚至过去的自己）收集的数据来学习，样本效率更高。

> **一句话总结**：Q-Learning 学习的是**最优策略的价值**，而不是当前执行策略的价值。

#### Epsilon-Greedy 探索策略

为了平衡 **探索（Exploration）** 与 **利用（Exploitation）**，Q-Learning 使用 epsilon-greedy 策略（[`01_q_learning/q_learning.py`](01_q_learning/q_learning.py)，第 68–82 行）：

```python
def select_action(self, state, eval_mode=False):
    # 评估模式：始终选择最优动作（不探索）
    if eval_mode:
        return int(np.argmax(self.q_table[state]))

    # 训练模式：epsilon-greedy
    if random.random() < self.epsilon:
        # 以概率 ε 随机选一个动作（探索）
        return random.randint(0, self.n_actions - 1)
    else:
        # 以概率 1-ε 选择当前 Q 值最高的动作（利用）
        return int(np.argmax(self.q_table[state]))
```

- **探索**（$\epsilon$ 概率）：随机尝试，发现更好的动作
- **利用**（$1-\epsilon$ 概率）：选择当前已知最好的动作
- $\epsilon$ 通常随时间衰减：`self.epsilon = max(epsilon_end, epsilon - epsilon_decay)` —— 初期多探索，后期多利用

---

> **常见误区 1：Q-Learning 学到的不一定是安全策略**

初学者常犯的错误：以为 Q-Learning 学出的策略在任何环境下都是最优的。实际上：
- Q-Learning 学到的策略可能**过于激进**。因为 max 操作会放大正误差（对某些动作的 Q 值估计偏高），可能导致智能体在"看起来好但实际上危险"的状态下选择有害动作。
- 在安全关键场景（如自动驾驶），SARSA 的保守策略可能更可取。

---

> **常见误区 2：学习率 $\alpha$ 设为 1 或 0 都不行**

- $\alpha = 1$ 意味着完全丢掉旧信息，导致 Q 值剧烈震荡，无法收敛
- $\alpha = 0$ 意味着完全不学习
- 实践中 $\alpha = 0.1$ 或 $0.01$ 是好的起点

---

> **检查点 6**：Q-Learning 为什么被称为 Off-Policy？如果我们在训练时把 epsilon 设为 0（纯贪婪），Q-Learning 还能学到新知识吗？为什么？

---

### 2.2 SARSA (1994) — On-Policy 时序差分控制

**论文**：*On-Line Q-Learning Using Connectionist Systems* — Rummery & Niranjan, 1994

#### 过渡：从 Q-Learning 到 SARSA

Q-Learning 总是乐观地假设"下一步会走最优的路"。但在很多现实场景中，这种乐观假设是危险的。

想象你走在悬崖边上。Q-Learning 会想："虽然现在离悬崖很近，但我相信下一步我会走最优的路（远离悬崖），所以现在这个位置没问题。" ——这可能导致你在悬崖边做出危险动作。

SARSA 则更谨慎："我现在用的是什么策略？我用的是 epsilon-greedy，意味着我有 ε 的概率随机走一步。如果我不小心随机走向了悬崖……那这个位置其实很危险。" ——SARSA 会把这个风险计入 Q 值。

#### 直觉理解：SARSA 想解决什么问题？

SARSA 学习的是**智能体实际执行的策略**的价值，而非最优策略的价值。名称 "SARSA" 来自其使用的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$——这五个字母正好描述了一次完整的交互循环。

#### 直观类比：带着实际决策习惯去评价餐厅

继续餐厅评分的类比。Q-Learning 的做法是：不管我今天怎么选餐厅的，更新评分时总是假设"下一顿我会去最好的餐厅"。

SARSA 的做法是：我今天实际是用 epsilon-greedy 选餐厅的（有时候故意尝试新餐厅），那更新评分时也要考虑这个事实："下一顿我可能还是会随机尝试，所以如果下一顿随机选到了一家难吃的餐厅怎么办？"

这就像一个谨慎的人做决策：**不光考虑"最优情况"，也考虑"万一我不小心选错了"的风险**。

#### 核心公式

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma\, Q(s', a') - Q(s, a)\right]$$

与 Q-Learning 的唯一区别：将 $\max_{a'} Q(s', a')$ 替换为 $Q(s', a')$，其中 $a'$ 是智能体在 $s'$ 处**实际执行**的动作。

**对比**：

| | Q-Learning | SARSA |
|------|------------|-------|
| TD 目标 | $r + \gamma \max_{a'} Q(s', a')$ | $r + \gamma Q(s', a')$ |
| 假设 | 下一步选最优动作 | 下一步按实际策略选动作 |
| 策略类型 | Off-Policy | On-Policy |
| 探索风险 | 被忽略 | 被计入 Q 值 |

#### 代码实现（带逐行注释）

SARSA 更新在 `SARSA.update()` 方法中（[`02_sarsa/sarsa.py`](02_sarsa/sarsa.py)，第 59–82 行）：

```python
def update(self, state, action, reward, next_state, next_action, done):
    # 注意：相比 Q-Learning，这里多了一个参数 next_action
    if done:
        target = reward  # 终止状态，没有下一步
    else:
        # 关键区别：使用实际执行的下一个动作的 Q 值
        # 而不是 max Q 值
        target = reward + self.gamma * self.q_table[next_state, next_action]

    td_error = target - self.q_table[state, action]
    self.q_table[state, action] += self.lr * td_error
    return abs(td_error)
```

> **关键区别**：`self.q_table[next_state, next_action]` —— 使用**实际执行的下一个动作**的 Q 值，而非最大值。这意味着如果探索策略偶尔走出一个"坏"动作，这个"坏"的影响会被计入 Q 值中。

#### On-Policy（同策略）本质

SARSA 是 **On-Policy** 的：行为策略与目标策略是**同一个策略**。智能体学习的是自己正在执行的策略的 $Q^\pi$，而非最优 $Q^*$。

**一句话总结**：SARSA 学习的是**"带探索的策略"的价值**，所以学出的策略更保守、更安全。

> **什么时候 SARSA 比 Q-Learning 更好？** 当环境中存在"走错一步就会受到严重惩罚"的风险时（例如悬崖行走、自动驾驶），SARSA 的保守性反而是优势。

#### Expected SARSA（期望 SARSA）—— 折中方案

Expected SARSA 是一种介于 Q-Learning 和 SARSA 之间的变体（[`02_sarsa/sarsa.py`](02_sarsa/sarsa.py)，第 94–156 行）：

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \sum_{a'} \pi(a'|s')\, Q(s', a') - Q(s, a)\right]$$

直觉：不用 $\max$（太乐观），也不用单个采样（太随机），而是用**所有可能动作的加权平均**。

```python
# 计算期望Q值 —— 用epsilon-greedy策略的概率分布做加权平均
best_action = np.argmax(self.q_table[next_state])
prob_best = 1 - self.epsilon + self.epsilon / self.n_actions
prob_other = self.epsilon / self.n_actions

expected_q = 0.0
for a in range(self.n_actions):
    if a == best_action:
        expected_q += prob_best * self.q_table[next_state, a]
    else:
        expected_q += prob_other * self.q_table[next_state, a]

target = reward + self.gamma * expected_q  # 期望 Q 值作为目标
```

Expected SARSA 通过对所有动作的 Q 值做加权平均，**消除了 SARSA 中因动作随机采样引入的方差**，同时保持了对探索动作风险的考虑。它仍然属于 On-Policy 方法。

---

> **常见误区 3：SARSA 和 Q-Learning 选哪个？**

初学者常问"哪个算法更好"，但答案是**取决于任务**。在安全关键场景（悬崖行走），SARSA 更好；在不需要担心探索风险的场景，Q-Learning 因为直接学习最优策略，通常收敛更快。**不存在绝对更好，只有更适合**。

---

> **常见误区 4：SARSA 的 next_action 忘了传参数**

SARSA 的 `update()` 比 Q-Learning 多一个 `next_action` 参数。初学者在写训练循环时容易忘记在调用 `update()` 之前先选择 `next_action`，导致代码报错或行为错误。

---

> **检查点 7**：如果 SARSA 训练时将 epsilon 逐渐衰减到 0，最终 SARSA 和 Q-Learning 学到的策略会趋同吗？为什么？

---

### 2.3 REINFORCE (1992) — 蒙特卡洛策略梯度

**论文**：*Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* — Ronald J. Williams, 1992

#### 过渡：从 Value-Based 到 Policy-Based

Q-Learning 和 SARSA 都是先学 Q 表，再通过查表选动作。但这种方法有一个根本局限：**当动作空间是连续的（如机器人关节的角度），你没法建一张无限大的表**。

**REINFORCE 走了一条完全不同的路**：直接学习一个策略函数 $\pi_\theta(a|s)$（比如一个神经网络），输入状态，输出每个动作的概率。不再需要 Q 表这个"中间商"。

这也意味着 REINFORCE 属于 **Policy-Based（基于策略）** 方法，与 Q-Learning/SARSA 的 **Value-Based（基于价值）** 方法形成对比。

#### 直觉理解：REINFORCE 想解决什么问题？

核心问题：**如何直接优化一个策略函数，使得"好"的动作被选中的概率更高，"坏"的动作的概率更低？**

直觉解法：
1. 让智能体完整跑一局游戏，记录每一步的 (状态, 动作, 最终总回报)
2. 如果某个动作导致了高回报 → 提升这个动作的概率
3. 如果某个动作导致了低回报 → 降低这个动作的概率

这就是 REINFORCE 的全部思想！

#### 直观类比：篮球教练的赛后复盘

想象你是一个篮球教练，你在赛后回看比赛录像：

- 球员在某个位置投了一个三分球（**状态-动作对**）
- 最终球队赢了 20 分（**高回报**）
- 结论：在这个位置投三分是个好选择，以后应该多鼓励（**提高概率**）

- 另一个球员在关键时刻传球失误（**状态-动作对**）
- 最终球队输了 5 分（**低回报**）
- 结论：在类似情况下不要那样传球（**降低概率**）

REINFORCE 做的事情和这个篮球教练一模一样——只不过它是自动的、数学化的。

#### 策略梯度定理（渐进式推导）

**第 1 步：定义优化目标**

我们希望最大化期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

其中 $\tau$ 是轨迹，$G(\tau)$ 是该轨迹的总回报。

**第 2 步：我们需要梯度**

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

但"期望的梯度"不好直接计算，因为期望依赖于策略参数 $\theta$。

**第 3 步：使用 log-derivative 技巧**

关键数学技巧：$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$

利用这个技巧和期望的线性性质，可以得到：

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

**第 4 步：直觉解读**

- $G_t$（回报）：动作的"好/坏"程度 —— 提示更新的**方向**
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$（对数概率梯度）：如何改变参数来提升该动作的概率 —— 提示更新的**方式**
- 乘积：好动作（$G_t$ 大）→ 大幅提升概率；坏动作（$G_t$ 小/负）→ 降低概率

**一句话**：**REINFORCE = 用最终回报来加权每个动作的"概率提升方向"**。

#### 代码实现（带逐行注释）

策略梯度在 `REINFORCE.update()` 中计算（[`03_reinforce/reinforce.py`](03_reinforce/reinforce.py)，第 98–137 行）：

```python
# ====== 第1步：计算每个时间步的折扣回报 ======
returns = []
G = 0
# 从最后一步向前遍历（因为回报是"从当前往后"的累加和）
for r in reversed(self.rewards):
    G = r + self.gamma * G   # 递推公式：G_t = r_t + γ * G_{t+1}
    returns.insert(0, G)     # 插入到列表开头，保持时间顺序

# ====== 第2步：标准化回报（减少方差，加速收敛）======
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
# 1e-8 防止除以零

# ====== 第3步：计算策略梯度 ======
for state, action, G_val in zip(self.states, self.actions, returns):
    probs = self.policy_network(state)             # 策略网络输出动作概率
    dist = torch.distributions.Categorical(probs)   # 构造分类分布
    log_prob = dist.log_prob(action)                # 计算 log π(a|s)
    # 注意负号！PyTorch做梯度下降，策略梯度需要梯度上升
    # 因此取负号将"最大化回报"转化为"最小化负回报"
    policy_loss.append(-log_prob * G_val)
```

> **关键细节**：`-log_prob * G_val` 的负号 —— PyTorch 的优化器执行的是**梯度下降**（最小化损失），但策略梯度算法需要**梯度上升**（最大化回报）。因此我们构造负损失，最小化它等价于最大化期望回报。

#### 带基线的 REINFORCE（REINFORCE with Baseline）

**问题**：原始 REINFORCE 的梯度估计方差很大。比如所有回报都 +100 和所有回报都 +101，区别很小，但原始方法无法感知这种细微差别。

**解决方案**：引入基线（Baseline）——一个价值网络 $V(s_t)$，用它来"归零"回报：

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[(G_t - V(s_t))\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

$(G_t - V(s_t))$ 称为**优势估计（Advantage Estimate）**：

- $G_t > V(s_t)$：动作比预期好 → 提升概率
- $G_t < V(s_t)$：动作比预期差 → 降低概率
- $G_t \approx V(s_t)$：动作和预期差不多 → 概率几乎不变

**直观类比**：基线就像"平均分"。考了 80 分在满分 100 的考试中还不错，但在满分 150 的考试中就不行了。只有知道"平均分"（基线），才能正确评价一个分数。

在代码中（[`03_reinforce/reinforce.py`](03_reinforce/reinforce.py)，`REINFORCEWithBaseline` 类，第 140–238 行）：

```python
# 计算价值估计（用价值网络预测每个状态的 V 值）
states_tensor = torch.cat(self.states)
values = self.value_network(states_tensor)

# 优势估计 = 实际回报 - 价值网络预测（.detach() 防止梯度传到价值网络）
advantages = returns - values.detach()

# 用优势替代原始回报来计算策略损失
for state, action, adv in zip(self.states, self.actions, advantages):
    probs = self.policy_network(state)
    dist = torch.distributions.Categorical(probs)
    log_prob = dist.log_prob(action)
    policy_loss.append(-log_prob * adv)  # 优势替代了原始回报

# 价值网络的损失：让 V(s) 尽可能接近实际回报 G
value_loss = F.mse_loss(values, returns)
```

---

> **常见误区 5：把 log_prob 当成普通的概率**

初学者容易混淆 $\log \pi_\theta(a|s)$ 和 $\pi_\theta(a|s)$。在策略梯度定理中，梯度使用的是 **对数概率** 而非概率本身。为什么？因为 $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$ —— 除以概率起到了"归一化"的作用：对于已经很频繁的动作，即使梯度相同，更新幅度也会因除以概率而减小。

---

> **常见误区 6：忘记标准化回报**

原始 REINFORCE 的梯度方差极大，往往无法收敛。对回报做标准化（减去均值，除以标准差）是几乎所有实现中的标配操作。初学者如果跳过这一步，训练结果通常会很差。

---

> **检查点 8**：REINFORCE 属于 MC 方法还是 TD 方法？为什么它必须等到 episode 结束才能更新？

---

### 2.4 On-Policy 与 Off-Policy 对比总结

| 维度 | On-Policy (SARSA) | Off-Policy (Q-Learning) |
|------|------------------|------------------------|
| 学习目标 | 学习当前执行策略的 $Q^\pi$ | 直接学习最优策略的 $Q^*$ |
| 行为策略 = 目标策略？ | 是 | 否 |
| 更新中使用 | 实际下一个动作 $a'$ | $\max_{a'} Q(s', a')$ |
| 探索策略的影响 | 探索动作的风险被计入 Q 值 | 不受探索动作影响 |
| 策略保守性 | 更保守，更安全 | 更激进，追求最优 |
| 样本效率 | 较低（需与环境交互遵循当前策略） | 较高（可复用历史数据） |
| 典型问题 | 策略可能过早收敛到次优 | Q 值可能被过高估计 |

> **一句话记住**：Q-Learning 问"最优是什么"，SARSA 问"我在做什么"。

---

### 2.5 总结对照

| 算法 | 年份 | 类型 | 学习方式 | 策略关系 | 输出 |
|------|------|------|---------|---------|------|
| Q-Learning | 1989 | Value-Based | TD | Off-Policy | Q 表 → 隐式策略 |
| SARSA | 1994 | Value-Based | TD | On-Policy | Q 表 → 隐式策略 |
| REINFORCE | 1992 | Policy-Based | MC | On-Policy | 显式策略网络 |

---

> **第一章终极检查点**：
> 1. 如果状态空间有 1000 个状态，动作空间有 10 个动作，Q 表有多大？
> 2. 如果状态空间从 1000 变成一百万，Q 表方法还可行吗？如果不可以，下一章会给出什么解决方案？

---

## 3. 运行方法

每个算法子目录下包含 `train.py` 训练脚本，直接运行即可：

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
```

运行结果（奖励曲线、收敛速度对比等）将在训练过程中打印到控制台，推荐自行运行后截图保存于各算法子目录下。

---

## 4. 学习建议

1. **先理解 MDP 框架**：RL 所有理论的起点是 MDP 的五个元素，搞清楚它们再去读算法会事半功倍。
2. **手推 Bellman 方程**：拿出一张纸，从 $G_t = r + \gamma G_{t+1}$ 出发，一步步推到 $Q^*(s,a)$ 的最优方程。亲手推一遍胜过读十遍。
3. **对比 Q-Learning 和 SARSA 的代码**：二者的 `update()` 方法只有一行不同，但这「一行之差」体现了 On-Policy 和 Off-Policy 的分野。对比阅读，体会设计哲学的区别。
4. **REINFORCE 是策略梯度的起点**：理解了 $\nabla_\theta \log \pi_\theta(a|s) \cdot G_t$ 这个式子，后续的 A2C、PPO 就只是在此基础上的变体。如果这里不理解，不要急着往后走。
5. **动手运行**：尝试修改探索率 $\epsilon$、学习率 $\alpha$、折扣因子 $\gamma$ 等超参数，观察对训练速度和最终性能的影响。调参是培养直觉的最好方式。
