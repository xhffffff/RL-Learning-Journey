# 第三章：高级强化学习算法 (2015-2023)

## 章节介绍：从 Model-Free 到 Model-Based 的过渡

本章是强化学习算法进阶的核心章节。在前两章中，我们学习了基础的 Policy Gradient、DQN 及其变体，它们都属于 **Model-Free** 范式——智能体直接与环境交互，不尝试理解环境的内在规律。然而，当面对样本效率要求高、需要长期规划的任务时，Model-Free 方法的局限性逐渐显现。

本章的四篇文章代表了 RL 从 Model-Free 向更复杂范式演进的关键节点：

| 算法 | 年份 | 范式 | 核心贡献 |
|------|------|------|----------|
| **TRPO** | 2015 | Model-Free | 信任区域约束，保证策略单调提升 |
| **TD3** | 2018 | Model-Free | 解决 Q 值过估计，DDPG 的三大改进 |
| **MuZero** | 2019 | Model-Based + Planning | 无需环境规则就能学习世界模型并规划 |
| **Dreamer** | 2020-2023 | Model-Based | 在隐空间"做梦"学习，极致样本效率 |

学习路线建议：按时间顺序学习，先理解 TRPO 的信任区域思想（它是 PPO 的前身），再掌握 TD3 如何工程化地修复 DDPG，最后进入 Model-Based 的世界——MuZero 教你如何学会一个可用于规划的世界模型，Dreamer 则展示了如何在想象中直接训练策略。

---

## 01. TRPO — Trust Region Policy Optimization (2015)

### 论文信息
- **标题**: *Trust Region Policy Optimization*
- **作者**: Schulman et al. (UC Berkeley)
- **链接**: [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)

### 核心思想

TRPO 的核心问题是：**如何在保证策略不"变坏"的前提下，尽可能大地更新策略？**

在普通 Policy Gradient 中，如果学习率太大，策略可能"一脚踩空"导致性能崩溃。TRPO 引入了**信任区域（Trust Region）**的概念：限制新旧策略之间的 KL 散度，确保每次更新都在一个"安全的半径"内。

### 💡 直觉解释：学骑自行车的类比

想象你正在学骑自行车。你当前的"策略"是某种身体倾斜+踏板节奏的组合。现在你想改进策略——但你怎么知道改多大步子不会摔倒？

- **普通 Policy Gradient**：每次大幅度调整身体姿势（大学习率）→ 很容易摔倒（性能崩溃）
- **TRPO**：每次只允许自己在一个"安全的晃动范围"内调整（KL 散度约束）→ 虽然进步慢，但确保你不会比之前更差

TRPO 的核心哲学是：**宁可小步稳进，不可大步倒退**。这种思想后来被 PPO 用更简单的方式（裁剪目标函数）实现了，但 TRPO 是第一个提出并严格证明"单调整"理论的算法。

### 优化目标

TRPO 求解如下带约束的优化问题：

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a)\right]$$

$$\text{s.t.} \quad \mathbb{E}\left[D_{KL}\left(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)\right)\right] \leq \delta$$

其中：
- $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 是重要性采样比率（Importance Sampling Ratio）
- $\hat{A}(s,a)$ 是优势函数估计（本实现使用 GAE）
- $D_{KL}(\pi_{\theta_{old}} \parallel \pi_\theta)$ 衡量新旧策略的差异
- $\delta$ 是信任区域半径（超参数）

目标直观理解：**在策略变化不超过 $\delta$ 的前提下，最大化期望优势**。

### 本实现的简化说明

> ⚠️ **重要**：本实现采用 **简化版 TRPO**，使用 KL 散度惩罚系数来近似信任区域约束，而非原始论文中的共轭梯度 + 回溯线搜索方法。

完整的 TRPO 使用二阶优化：用共轭梯度求解自然梯度方向，再用回溯线搜索选择合适的步长。这涉及 Hessian-vector product 等复杂计算。本简化版本将 KL 约束转化为惩罚项直接加入损失函数，在概念上更接近 PPO 的 KL 惩罚变体。

代码中简化版 KL 惩罚实现（`trpo.py` 第 170-177 行）：

```python
# KL惩罚项 (TRPO简化为KL惩罚)
with torch.no_grad():
    old_dist = torch.distributions.Categorical(
        logits=self.network.actor(states_t).detach())
kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

# KL惩罚损失
actor_loss = surr_loss + self.damping * kl
```

重要性采样比率计算（`trpo.py` 第 165 行）：

```python
ratio = (new_log_probs - old_log_probs_t).exp()
```

GAE 优势估计（`trpo.py` 第 124-138 行）：

```python
def _compute_gae(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    values = np.array(self.values + [last_value])
    dones = np.array(self.dones + [0])
    rewards = np.array(self.rewards)

    advantages = np.zeros(len(rewards))
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns
```

### 关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑参数 |
| `max_kl` | 0.01 | 信任区域半径 |
| `damping` | 0.1 | KL 惩罚系数 |

### 完整训练

✅ 提供完整 `train.py`，可在 Gym 环境中运行。

---

## 02. TD3 — Twin Delayed DDPG (2018)

### 论文信息
- **标题**: *Addressing Function Approximation Error in Actor-Critic Methods*
- **作者**: Fujimoto et al. (McGill University)
- **链接**: [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

### 核心思想

DDPG（Deep Deterministic Policy Gradient）是连续动作空间的经典 Actor-Critic 算法，但它有一个致命缺陷：**Q 值过估计（Overestimation Bias）**。由于 max 操作和函数近似误差，Critic 倾向于系统性高估动作价值，导致 Actor 向错误方向学习。

TD3 用三个针对性改进彻底修复了这个问题。

### 💡 直觉解释：三个评委打分

想象你在参加体操比赛，评分机制有如下问题：

- **问题**：只有一个评委（单个 Critic），但他有点近视（函数近似误差）——他可能因为没看清而给不完美的动作打高分
- **你（Actor）的困境**：你看到高分就以为那个动作好，实际上只是评委看错了

TD3 的三招修复：

1. **双评委制（Clipped Double Q-Learning）**：请两个独立的评委，取他们中的**最低分**。即使一个评委误判了高分，取 min 后分数也不会虚高
2. **动作加噪（Target Policy Smoothing）**：评分时，在你做的动作上加点随机扰动再打分——确保你不是"恰好蒙对了"某个特定动作
3. **评委先打分，选手再调整（Delayed Policy Updates）**：等评委多打几次分（Critic 更新），让分数稳定了，你再调整动作（Actor 更新）

> 💡 **一句话总结**：TD3 = DDPG + 双 Q 取 min + 目标平滑 + 延迟更新。它本质上是一套"防止被不准确的 Q 值误导"的工程补丁集合。

### 改进一：Clipped Double Q-Learning（裁剪双 Q 学习）

TD3 使用两个独立的 Critic 网络 $Q_{\theta_1}$ 和 $Q_{\theta_2}$，计算目标时取两者的**最小值**：

$$y = r + \gamma \min_{i=1,2} Q_{\theta_i^-}(s', \tilde{a})$$

这有效抑制了过估计——即使一个 Q 网络给出了偏高的估计，取 min 后也会被另一个约束。

代码实现（`td3.py` 第 177-180 行）：

```python
# ---- TD3改进2: Clipped Double Q-Learning ----
target_q1, target_q2 = self.critic_target(next_states, next_actions)
target_q = torch.min(target_q1, target_q2)
target_q = rewards + self.gamma * (1 - dones) * target_q
```

双 Critic 网络架构（`td3.py` 第 54-61 行）——在一个模块内实现两个独立的 Q 网络：

```python
# Q1架构
self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
self.fc2 = nn.Linear(hidden_dim, hidden_dim)
self.fc3 = nn.Linear(hidden_dim, 1)

# Q2架构 (双Q网络)
self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
self.fc5 = nn.Linear(hidden_dim, hidden_dim)
self.fc6 = nn.Linear(hidden_dim, 1)
```

### 改进二：Target Policy Smoothing（目标策略平滑）

给目标动作添加裁剪过的高斯噪声，使 Q 函数在动作周围更平滑，防止策略利用 Q 函数的"尖峰"：

$$\tilde{a} = \pi_{\phi^-}(s') + \epsilon, \quad \epsilon \sim \text{clip}\left(\mathcal{N}(0, \sigma), -c, c\right)$$

其中 $\sigma$ 是噪声标准差（`policy_noise=0.2`），$c$ 是噪声裁剪范围（`noise_clip=0.5`）。

代码实现（`td3.py` 第 169-175 行）：

```python
# ---- TD3改进1: 目标策略平滑 ----
with torch.no_grad():
    noise = (torch.randn_like(actions) * self.policy_noise).clamp(
        -self.noise_clip, self.noise_clip
    )
    next_actions = (self.actor_target(next_states) + noise).clamp(
        -self.max_action, self.max_action
    )
```

### 改进三：Delayed Policy Updates（延迟策略更新）

Actor（策略网络）的更新频率低于 Critic（Q 网络）——默认每更新 2 次 Critic 才更新 1 次 Actor。这确保了 Actor 总是基于相对准确的 Q 值来学习，避免"垃圾进，垃圾出"。

代码实现（`td3.py` 第 190-202 行）：

```python
# ---- TD3改进3: 延迟策略更新 ----
if self.total_it % self.policy_freq == 0:
    actor_loss = -self.critic.q1(states, self.actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # 软更新目标网络
    for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
        target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
    for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
        target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
```

### 关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `policy_noise` | 0.2 | 目标策略平滑噪声标准差 $\sigma$ |
| `noise_clip` | 0.5 | 噪声裁剪范围 $c$ |
| `policy_freq` | 2 | 策略延迟更新频率 |
| `tau` | 0.005 | 目标网络软更新系数 |
| `gamma` | 0.99 | 折扣因子 |

### 完整训练

✅ 提供完整 `train.py`，适用于 Pendulum 等连续控制环境。

---

## 03. MuZero — 学习式规划 (2019)

### 论文信息
- **标题**: *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*
- **作者**: DeepMind
- **链接**: [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)

### 核心思想

MuZero 是 Model-Based RL 的一个里程碑。与传统 Model-Based 方法不同，MuZero **不需要知道环境的规则**——它从零开始学习一个抽象的隐藏状态空间，并在这个空间中做规划（MCTS）。

### 🔥 为什么 MuZero 重要？

MuZero 解决了一个困扰 RL 多年的根本问题：**如何在不知道环境规则的情况下做规划？**

- **AlphaGo** 需要知道围棋的完整规则（落子、提子、胜负判定）
- **AlphaZero** 同样需要环境规则来执行 MCTS 搜索
- **MuZero** 完全从像素/观测中学习世界模型——给它看 Atari 像素，它就能在脑内"想象"未来的游戏画面并做出规划

这意味着 MuZero 是第一个**真正通用的基于模型的 RL**——它可以应用于任何环境，无论你能否写出环境的数学规则。这对于现实世界应用（机器人、自动驾驶、推荐系统）意义深远：现实世界没有完美的物理方程，但 MuZero 可以**从经验中学会**环境的规律。

> 💡 与 LLM 时代的关联：MuZero 的"学一个隐式世界模型来做规划"的思想，与 GPT 系列"学一个隐式语言模型来预测下一个 token"在哲学上高度一致。

它的三个核心网络函数构成了一个"内部世界模拟器"：

| 函数 | 数学符号 | 输入 → 输出 | 作用 |
|------|----------|-------------|------|
| **Representation** | $h_\theta(o_t) \to s^0$ | 原始观察 $o_t$ → 隐藏状态 $s^0$ | 将观察编码到隐空间 |
| **Dynamics** | $g_\theta(s^{k-1}, a^k) \to r^k, s^k$ | 隐藏状态 + 动作 → 奖励 + 下一状态 | 在隐空间中"模拟" |
| **Prediction** | $f_\theta(s^k) \to p^k, v^k$ | 隐藏状态 → 策略 + 价值 | 从隐状态预测策略和价值 |

### Representation 函数（表示网络）

$h_\theta(o_t) \to s^0$：将像素或向量观测映射为隐藏状态。原始论文使用 ResNet，本实现提供简化版 MLP/CNN。

代码实现（`muzero.py` 第 23-67 行）：

```python
class MuZeroRepresentation(nn.Module):
    """表示函数 h: o_t → s^0"""
    def __init__(self, obs_shape, hidden_dim: int = 256):
        # ...自适应处理向量观测和图像观测...
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
```

### Dynamics 函数（动态网络）

$g_\theta(s^{k-1}, a^k) \to r^k, s^k$：给定隐状态和动作，预测即时奖励和下一隐状态。

代码实现（`muzero.py` 第 70-101 行）：

```python
class MuZeroDynamics(nn.Module):
    """动态函数 g: (s^{k-1}, a^k) → (r^k, s^k)"""
    def forward(self, hidden_state, action):
        x = torch.cat([hidden_state, action], dim=-1)
        features = self.network(x)
        reward_logits = self.reward_head(features)
        next_hidden = self.state_head(features)
        return reward_logits, next_hidden
```

### Prediction 函数（预测网络）

$f_\theta(s^k) \to p^k, v^k$：从隐状态预测策略分布和价值。

代码实现（`muzero.py` 第 104-133 行）：

```python
class MuZeroPrediction(nn.Module):
    """预测函数 f: s^k → (p^k, v^k)"""
    def forward(self, hidden_state):
        features = self.network(hidden_state)
        policy_logits = self.policy_head(features)
        value_logits = self.value_head(features)
        return policy_logits, value_logits
```

### MCTS 搜索与 PUCT 公式

有了上述三个函数，MuZero 使用 **蒙特卡洛树搜索（MCTS）** 进行规划。在每个节点，使用 **PUCT（Predictor + UCT）** 公式选择子节点：

$$a^k = \arg\max_a \left[Q(s,a) + c_1 \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}\right]$$

其中：
- $Q(s,a)$：该动作的估计价值
- $P(s,a)$：策略网络给出的先验概率
- $N(s,a)$：该动作的访问次数
- $c_1$、$c_2$：探索常数

代码实现（`muzero.py` 第 181-196 行）：

```python
def select_child(self, c1: float = 1.25, c2: float = 19652.0):
    """PUCT选择公式"""
    for action, child in self.children.items():
        ucb_score = child.value() + \
            c1 * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count) * \
            (c2 + math.log((self.visit_count + c2 + 1) / c2))

        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action
            best_child = child
```

### 当前实现的简化说明

> ⚠️ 由于完整 MuZero 实现极其复杂（包含并行 MCTS、reanalyse buffer、value/reward 的 categorical 表示等），本文件仅演示 **三个核心网络组件 + 简化 MCTS 节点**。这不构成端到端可训练的 MuZero 系统。

### 完整训练

❌ 本章仅提供核心网络组件的教学演示（运行 `python muzero.py` 可观察网络结构），无完整训练脚本。

---

## 04. Dreamer — 在隐空间"做梦"学习 (2020-2023)

### 论文信息
- **标题**: *Mastering Diverse Domains through World Models* (DreamerV3, 2023)
- **作者**: Hafner et al. (DeepMind & TU/e)
- **链接**: [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

### 核心思想

Dreamer 使 Model-Based RL 走向实用化的代表作。它的核心理念是：**先学一个世界模型，然后在隐空间中"想象"（imagine）来训练策略，只用少量真实交互。**

```
真实环境交互 → 学习世界模型 → 隐空间中想象展开 → 策略学习
```

### 🔥 为什么 Dreamer 重要？

Dreamer 解决了 RL 最痛的一个问题：**样本效率**。

- **Model-Free RL**（PPO, DQN）：需要与环境交互数百万次才能学会一个任务——在真实机器人上，这意味着无数次试错和物理损坏
- **Dreamer**：只用几千次真实交互学习世界模型，然后在"梦中"（隐空间想象）做无限次虚拟试错

DreamerV3 更进一步：它是第一个**无需任何超参数调优**就能在 Atari、DeepMind Control Suite、甚至 Minecraft 上都能工作的算法。这种"通用性"非常罕见——大多数 RL 算法在不同任务上需要重新调参。

> 💡 与 LLM 时代的关联：Dreamer 的"先建模、再在模型中规划"的思路，和当前 LLM Agent 用世界模型做推理规划的范式（如 LLM-Planner）一脉相承。

### RSSM 架构：确定性状态 + 随机状态

Dreamer 的核心是世界模型中的 **RSSM（Recurrent State-Space Model，循环状态空间模型）**，它将隐状态分解为两种互补的表示：

```
         ┌──────────────────────────┐
         │     RSSM 隐状态分解       │
         ├──────────┬───────────────┤
         │    h_t   │      z_t      │
         │ 确定性状态 │   随机状态     │
         │ (GRU建模) │ (Categorical) │
         │  存储长期  │   建模不确定性  │
         │  上下文   │   和随机性     │
         └──────────┴───────────────┘
```

- **确定性状态 $h_t$**：通过 GRU 循环网络更新，$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$，负责存储长期上下文
- **随机状态 $z_t$**：通过离散 Categorical 分布建模，捕捉环境的随机性和多模态性

状态转移：

$$p(z_t \mid h_t) \quad \text{（先验：仅从确定性状态预测）}$$
$$q(z_t \mid h_t, o_t) \quad \text{（后验：融合观察信息后更准确的估计）}$$

RSSM 一步前向代码（`dreamer.py` 第 87-102 行）：

```python
def forward(self, action, h, z):
    """一步前向: (h_t, z_t, a_t) → (h_{t+1}, z_{t+1})"""
    rnn_input = torch.cat([z, action], dim=-1)
    h_new = self.rnn(rnn_input, h)
    prior_dist = self.prior(h_new)
    z_new = prior_dist.sample()
    return h_new, z_new
```

Prior 分布 $p(z|h)$（`dreamer.py` 第 74-78 行）：

```python
def prior(self, h: torch.Tensor):
    """先验分布 p(z|h)"""
    logits = self.prior_net(h)
    logits = logits.view(-1, self.stochastic_dim, self.class_dim)
    return torch.distributions.Categorical(logits=logits)
```

Posterior 分布 $q(z|h,o)$（`dreamer.py` 第 80-85 行）：

```python
def posterior(self, h, obs_embed):
    """后验分布 q(z|h,o)"""
    inputs = torch.cat([h, obs_embed], dim=-1)
    logits = self.posterior_net(inputs)
    logits = logits.view(-1, self.stochastic_dim, self.class_dim)
    return torch.distributions.Categorical(logits=logits)
```

### 三个损失函数

Dreamer 的世界模型通过最小化以下联合损失来训练：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{reward}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

| 损失项 | 含义 | 作用 |
|--------|------|------|
| $\mathcal{L}_{\text{recon}}$ | 重构损失 | 从隐状态 $z_t$ 解码回原始观察 $\hat{o}_t$，确保隐空间保留足够信息 |
| $\mathcal{L}_{\text{reward}}$ | 奖励预测损失 | 从隐状态预测奖励 $\hat{r}_t$，确保世界模型理解任务目标 |
| $\beta \cdot \mathcal{L}_{\text{KL}}$ | KL 正则化 | 约束后验 $q(z_t|h_t,o_t)$ 不要偏离先验 $p(z_t|h_t)$ 太远 |
| $\beta$ | KL 平衡系数 | DreamerV3 中自动调节，无需手动调参 |

Encoder（观察→嵌入）和 Decoder（隐变量→观察重构）代码（`dreamer.py` 第 105-151 行）：

```python
class DreamerEncoder(nn.Module):
    """编码器: 观察 → 嵌入"""
    def forward(self, obs):
        return self.network(obs)

class DreamerDecoder(nn.Module):
    """解码器: 隐变量 → 观察"""
    def forward(self, z):
        z_flat = z.view(batch_size, -1)
        return self.network(z_flat)
```

Reward Predictor 代码（`dreamer.py` 第 154-171 行）：

```python
class RewardPredictor(nn.Module):
    """奖励预测器: 隐变量 → 奖励"""
    def forward(self, z, h):
        return self.network(torch.cat([z_flat, h], dim=-1))
```

### Imagine（想象展开）

在学好的世界模型中，Dreamer 可以不与环境交互，从任意初始隐状态出发，仅使用世界模型展开多步"想象"，并基于想象的奖励训练 Actor-Critic。这是 Dreamer 样本效率的来源。

代码实现（`dreamer.py` 第 196-215 行）：

```python
def imagine(self, h, z, actions):
    """世界模型中想象 (Rollout)"""
    for a in actions:
        h, z = self.rssm(a, h, z)
        reward = self.reward_predictor(z, h)
        horizons.append(h); zs.append(z); rewards.append(reward)
    return torch.stack(horizons), torch.stack(zs), torch.stack(rewards)
```

### 版本演进

| 版本 | 年份 | 核心改进 |
|------|------|----------|
| **DreamerV1** | 2020 | 基础世界模型 + Actor-Critic in latent space |
| **DreamerV2** | 2021 | 离散隐变量（Categorical latents）+ KL 平衡 |
| **DreamerV3** | 2023 | 无需超参数调节 + 跨领域通用（Atari/DeepMind Control/Minecraft） |

### 当前实现的简化说明

> ⚠️ 本实现聚焦于 **RSSM 核心组件（Encoder / RSSM / Decoder / Reward Predictor / Imagine）** 的教学演示，不含完整的联合训练循环和 Actor-Critic 在隐空间中的策略优化。

### 完整训练

❌ 本章仅提供核心组件的教学演示（运行 `python dreamer.py` 可观察各模块的形状和数据流），无完整训练脚本。

---

## ⚠️ 常见误区

本章涵盖的算法概念较深，以下是初学者的高频误区：

### 误区 1：TRPO = PPO 的"弱化版"

**纠正**：TRPO 是 PPO 的**前身**，不是弱化版。TRPO 提供了严格的理论保证（策略单调提升），但实现复杂（需要共轭梯度）；PPO 用简单的裁剪操作近似了 TRPO 的约束，牺牲了部分理论优雅性换取了实现简单性。两者各有优劣。

### 误区 2：TD3 只是 DDPG 加了点 trick

**纠正**：TD3 的三个"trick"不是随便加的——每个都有明确的动机：
- 双 Q 解决过估计偏误（被广泛验证的严重问题）
- 目标平滑解决策略对 Q 函数"尖峰"的利用
- 延迟更新解决了"用错误的 Q 值引导策略"的问题

它们一起让 DDPG 从"经常不收敛"变成"几乎总是收敛"。这是"工程修正"的经典范例。

### 误区 3：MuZero 知道环境规则

**纠正**：MuZero 的重要卖点恰恰是**不知道环境规则**。它从像素级观测中隐式学习世界模型。区别于 AlphaZero（需要围棋规则来执行合法动作和判断终局），MuZero 可以学习任何环境。

### 误区 4：Model-Based RL 比 Model-Free 好

**纠正**：各有利弊：
- Model-Based（MuZero、Dreamer）：样本效率极高，但模型误差会累积（想象偏差）
- Model-Free（PPO、SAC）：直接优化真实奖励，但需要大量交互
- 实际中，DreamerV3 在某些任务上确实超越了 Model-Free，但没有一种方法在所有任务上都最好

### 误区 5：在学第四章之前必须先完全掌握本章

**纠正**：本章的 RLHF 直接依赖是 PPO（第二章），而非本章的算法。如果你时间有限，可以**跳读本章**，直接进入第四章。但理解本章的"信任区域思想"（TRPO）和"隐空间规划思想"（MuZero/Dreamer），会让你对 RLHF/GRPO 的理解更加深刻。

---

## 运行方法

### 有完整训练脚本的算法

```bash
# TRPO (CartPole 离散环境)
cd chapter_03_advanced/01_trpo
python train.py --env CartPole-v1

# TD3 (Pendulum 连续控制)
cd chapter_03_advanced/02_td3
python train.py --env Pendulum-v1
```

### 核心组件演示（无完整训练）

```bash
# MuZero 网络组件演示
cd chapter_03_advanced/03_muzero
python muzero.py

# Dreamer RSSM 组件演示
cd chapter_03_advanced/04_dreamer
python dreamer.py
```

---

## 学习建议

1. **先跑通 TRPO 和 TD3**：这两个是 Model-Free 强化学习的"工业标准"算法，有完整的训练脚本，可以在几分钟内看到效果。
2. **TRPO 重在理解"信任区域"思想**：不必纠结共轭梯度的实现细节，理解"为什么要限制策略变化"就足够了——这正是 PPO 裁剪机制的前身。
3. **TD3 重在理解"防御性设计"**：三个改进都是为了防御 Q 值过估计——这是一种典型的通过观察失败模式来改进算法的工程思维。
4. **MuZero 重在理解架构**：三大网络函数如何配合 MCTS 进行规划，这是学会"思考"的 RL 的雏形。
5. **Dreamer 理解隐空间学习**：RSSM 如何将观察空间的问题转化为隐空间的问题，以及"想象"的意义。
6. **从本章自然过渡到第四章**：Model-Based 的思想在 LLM 时代以另一种形式复兴——我们同样在"学习人的偏好模型"和"在隐空间规划 token 序列"。
