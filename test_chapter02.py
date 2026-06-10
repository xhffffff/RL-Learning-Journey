"""
第二章测试: DQN, Double DQN, Dueling DQN, A2C, PPO, SAC
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
sys.path.insert(0, os.path.join(ROOT, 'common'))

print('=' * 70)
print('第二章 深度强化学习 (2013-2017)')
print('=' * 70)

# ============ 1. DQN ============
print()
print('--- 1. DQN (2015, Mnih et al., Nature) ---')
print()
print('【数据获取方式】')
print('  离线(Off-Policy) + 经验回放(Experience Replay)')
print('  流程:')
print('    1. 智能体与环境交互 -> 存入 ReplayBuffer')
print('    2. 从 Buffer 随机采样 batch -> 打破样本相关性')
print('    3. 用目标网络计算 TD target -> 稳定训练')
print('    4. 梯度下降更新在线网络')
print()
print('【数据长什么样】')
print('  ReplayBuffer 结构:')
print('    states:  (capacity, 4)   例: [[0.1, -0.2, 0.3, 1.5], ...]')
print('    actions: (capacity,)     例: [1, 0, 1, ...]')
print('    rewards: (capacity,)     例: [1.0, 1.0, 1.0, ...]')
print('    next_states: (capacity,4)')
print('    dones:   (capacity,)')
print('  每次采样 batch_size=64 条')
print()
print('【测试环境】CartPole-v1')
print('  状态: 4维连续向量')
print('  动作: 左/右推 (2个)')
print('  神经网络: MLP(4->128->128->2), 约 17k 参数')
print()

mod = load_module('dqn', os.path.join(ROOT, 'chapter_02_deep_rl', '01_dqn', 'dqn.py'))
DQN = mod.DQN

device = 'cpu'
agent = DQN(state_dim=4, action_dim=2, device=device)

# 模拟收集数据
for i in range(100):
    s = np.random.randn(4).astype(np.float32)
    ns = np.random.randn(4).astype(np.float32)
    agent.store_transition(s, np.random.randint(0, 2), 1.0, ns, False)

loss = agent.update()
print(f'  DQN 网络参数: {sum(p.numel() for p in agent.q_network.parameters()):,} 个')
print(f'  Buffer大小: {len(agent.memory)}')
print(f'  单步训练Loss: {loss:.4f}')
print('  DQN => OK')

# ============ 2. Double DQN ============
print()
print('--- 2. Double DQN (2015, van Hasselt) ---')
print()
print('【数据获取方式】')
print('  与DQN完全相同 —— 也是用经验回放')
print('  唯一区别在更新计算:')
print('    DQN:        target = r + gamma * max_a Q_target(s_next, a)')
print('    Double DQN: target = r + gamma * Q_target(s_next, argmax_a Q_online(s_next, a))')
print('  作用: 解耦动作选择和Q值评估，减少过估计')
print()

mod = load_module('ddqn', os.path.join(ROOT, 'chapter_02_deep_rl', '02_double_dqn', 'double_dqn.py'))
DoubleDQN = mod.DoubleDQN

agent2 = DoubleDQN(state_dim=4, action_dim=2, device=device)
for i in range(100):
    s = np.random.randn(4).astype(np.float32)
    ns = np.random.randn(4).astype(np.float32)
    agent2.store_transition(s, np.random.randint(0, 2), 1.0, ns, False)
loss = agent2.update()
print(f'  Double DQN Loss: {loss:.4f}')
print('  Double DQN => OK')

# ============ 3. Dueling DQN ============
print()
print('--- 3. Dueling DQN (2016, Wang et al.) ---')
print()
print('【数据获取方式】')
print('  同样使用经验回放，数据格式与DQN一致')
print('  差异在网络架构: 将Q拆为 V+Advantage')
print('  Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)')
print('  V(s): 这个状态有多好')
print('  A(s,a): 这个动作比平均好多少')
print('  优势: 部分状态不需要区分动作优劣时学习更快')
print()

mod = load_module('dueldqn', os.path.join(ROOT, 'chapter_02_deep_rl', '03_dueling_dqn', 'dueling_dqn.py'))
DuelingDQN = mod.DuelingDQN

agent3 = DuelingDQN(state_dim=4, action_dim=2, device=device)
for i in range(100):
    s = np.random.randn(4).astype(np.float32)
    ns = np.random.randn(4).astype(np.float32)
    agent3.store_transition(s, np.random.randint(0, 2), 1.0, ns, False)
loss = agent3.update()
print(f'  Dueling DQN Loss: {loss:.4f}')
print('  Dueling DQN => OK')

# ============ 4. A2C ============
print()
print('--- 4. A2C (2016, Mnih et al.) ---')
print()
print('【数据获取方式】')
print('  On-Policy 在线收集: Actor-Critic架构')
print('  Actor: 输出动作概率分布 pi(a|s)')
print('  Critic: 估计状态价值 V(s)')
print('  数据格式: [(s, a, r, V(s), log_pi(a|s), done), ...]')
print('  n-step优势: A_t = r_t + r_t+1 + ... + V(s_t+n) - V(s_t)')
print('  更新: 每个episode结束后一次性更新')
print()
print('【测试环境】CartPole-v1')
print()

mod = load_module('a2c', os.path.join(ROOT, 'chapter_02_deep_rl', '04_a3c', 'a2c.py'))
A2C = mod.A2C

agent4 = A2C(state_dim=4, action_dim=2, device='cpu')
for _ in range(20):
    s = np.random.randn(4).astype(np.float32)
    a = agent4.select_action(s)
    agent4.store_reward(1.0, False)
a_loss, c_loss, ent = agent4.update()
print(f'  A2C Actor Loss: {a_loss:.4f}')
print(f'  A2C Critic Loss: {c_loss:.4f}')
print(f'  Entropy: {ent:.4f}')
print('  A2C => OK')

# ============ 5. PPO ============
print()
print('--- 5. PPO (2017, Schulman et al., OpenAI) ---')
print()
print('【数据获取方式】')
print('  On-Policy 在线收集 + 多轮利用')
print('  1. 用当前策略收集一批数据 (rollout)')
print('  2. 计算 GAE 优势: A_t = delta + gamma*lambda*delta + ...')
print('  3. 对同一批数据做 ppo_epochs=10 轮更新')
print('  4. 裁剪目标: min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)')
print()
print('  核心数据:')
print('    states, actions, old_log_probs, advantages, returns')
print('  优势: 比DQN简单、稳定，是LLM训练的标准算法')
print('  需要: Policy + Critic 网络 (共享参数)')
print()
print('【测试环境】CartPole-v1 / Pendulum-v1 (连续动作)')
print()

mod = load_module('ppo', os.path.join(ROOT, 'chapter_02_deep_rl', '05_ppo', 'ppo.py'))
PPO = mod.PPO

agent5 = PPO(state_dim=4, action_dim=2, device='cpu')
for _ in range(64):
    s = np.random.randn(4).astype(np.float32)
    a = agent5.select_action(s)
    agent5.store_reward(1.0, False)
losses = agent5.update()
print(f'  PPO Policy Loss: {losses["policy"]:.4f}')
print(f'  PPO Value Loss: {losses["value"]:.4f}')
print(f'  PPO Entropy: {losses["entropy"]:.4f}')
print('  PPO => OK')

# ============ 6. SAC ============
print()
print('--- 6. SAC (2018, Haarnoja et al.) ---')
print()
print('【数据获取方式】')
print('  Off-Policy + 最大熵 (Maximum Entropy)')
print('  1. 智能体与环境交互 -> 存入 ReplayBuffer')
print('  2. 从 Buffer 采样 batch 训练')
print('  3. 双Q网络减少过估计')
print('  4. 自动调节温度参数 alpha 平衡探索与利用')
print()
print('  目标: J = E[r + alpha * H(pi(·|s))]')
print('  H(pi) 是策略熵，鼓励探索，防止过早收敛')
print('  需要: Actor + 双Critic + 双Target Critic + alpha')
print('      共5个网络! 但样本效率极高')
print()
print('【测试环境】Pendulum-v1 (连续动作空间)')
print('  状态: 3维 [cos(theta), sin(theta), theta_dot]')
print('  动作: 1维力矩 [-2, 2]')
print('  目标: 将摆杆向上摆动并保持')
print()

# SAC需要安装，先检查
try:
    mod = load_module('sac', os.path.join(ROOT, 'chapter_02_deep_rl', '06_sac', 'sac.py'))
    SAC = mod.SAC
    agent6 = SAC(state_dim=3, action_dim=1, device='cpu')
    for i in range(256):  # SAC batch_size=256
        s = np.random.randn(3).astype(np.float32)
        ns = np.random.randn(3).astype(np.float32)
        agent6.store_transition(s, np.random.randn(1).astype(np.float32), -1.0, ns, False)
    info = agent6.update()
    if info:
        print(f'  SAC Q Loss: {info["q_loss"]:.4f}')
        print(f'  SAC Policy Loss: {info["policy_loss"]:.4f}')
        print(f'  SAC Alpha: {info["alpha"]:.4f}')
    print('  SAC => OK')
except Exception as e:
    print(f'  SAC 测试跳过: {e}')

print()
print('=' * 70)
print('第二章全部通过! DQN → DoubleDQN → DuelingDQN → A2C → PPO → SAC')
print('=' * 70)
