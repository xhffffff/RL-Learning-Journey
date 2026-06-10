"""
第一章测试: Q-Learning, SARSA, REINFORCE
"""
import sys, os
import importlib.util
import numpy as np

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = os.path.dirname(__file__)

print('=' * 70)
print('第一章 基础算法 (1980s-2013)')
print('=' * 70)

# ============ 1. Q-Learning ============
print()
print('--- 1. Q-Learning (1989, Watkins) ---')
print()
print('【数据获取方式】')
print('  在线交互 —— 智能体与环境实时交互产生数据')
print('  每步交互产生 1 条样本: (s, a, r, s_next, done)')
print('  存储结构: Q表 np.ndarray(n_states, n_actions)')
print()
print('【数据长什么样】')
print('  state = 0 (初始位置)')
print('  action = 1 (向下走)')
print('  reward = 0.0 (没有奖励)')
print('  next_state = 4 (到新格子)')
print('  done = False (没到终点)')
print()
print('【测试环境】FrozenLake-v1 (4x4冰湖网格)')
print('  状态: 16个离散格子 {0...15}')
print('  动作: {0=左, 1=下, 2=右, 3=上}')
print('  目标: 从S走到G，避开冰洞H')
print('  S F F F')
print('  F H F H')
print('  F F F H')
print('  H F F G')
print()

mod = load_module('ql', os.path.join(ROOT, 'chapter_01_foundation', '01_q_learning', 'q_learning.py'))
QLearning = mod.QLearning

agent = QLearning(n_states=16, n_actions=4, learning_rate=0.1, gamma=0.99)
print(f'  Q表创建: shape={agent.q_table.shape}, 初始全0')
print(f'  探索率 epsilon={agent.epsilon:.1f}')

td = agent.update(state=0, action=1, reward=0.0, next_state=4, done=False)
action = agent.select_action(0)
print(f'  更新后 Q[0,1]={agent.q_table[0,1]:.3f}')
print(f'  选择动作: {action}')
print('  Q-Learning => OK')

# ============ 2. SARSA ============
print()
print('--- 2. SARSA (1994, Rummery) ---')
print()
print('【数据获取方式】')
print('  与Q-Learning相同: 在线交互产生 (s,a,r,s_next,a_next,done)')
print('  关键区别: SARSA用实际执行的a_next更新，不是max')
print('  Q-Learning(Off-Policy): 用 max Q(s_next, ·)')
print('  SARSA(On-Policy):        用 Q(s_next, a_next) —— 实际采取的动作!')
print()
print('【数据样本对比】')
print('  Q-Learning target = r + gamma * max_a Q(s_next, a)')
print('  SARSA      target = r + gamma * Q(s_next, a_next)')
print('  => a_next 必须是策略实际选出的那个动作')
print()

mod = load_module('sarsa', os.path.join(ROOT, 'chapter_01_foundation', '02_sarsa', 'sarsa.py'))
SARSA = mod.SARSA
ExpectedSARSA = mod.ExpectedSARSA

agent_s = SARSA(n_states=16, n_actions=4, learning_rate=0.1, gamma=0.99)
td = agent_s.update(state=0, action=1, reward=0.0, next_state=4, next_action=2, done=False)
print(f'  SARSA更新 Q[0,1]={agent_s.q_table[0,1]:.3f}')

agent_es = ExpectedSARSA(n_states=16, n_actions=4, learning_rate=0.1, gamma=0.99)
td = agent_es.update(state=0, action=1, reward=0.0, next_state=4, done=False)
print(f'  Expected-SARSA更新 Q[0,1]={agent_es.q_table[0,1]:.3f}')
print('  SARSA + Expected-SARSA => OK')

# ============ 3. REINFORCE ============
print()
print('--- 3. REINFORCE (1992, Williams) ---')
print()
print('【数据获取方式】')
print('  蒙特卡洛(MC)方式 —— 必须跑完整个episode才更新一次')
print('  步骤:')
print('    1. 用当前策略跑完 1 个 episode')
print('    2. 收集轨迹 [(s0,a0,r0)...(sT,aT,rT)]')
print('    3. 计算折扣回报 Gt = rt + gamma*rt1 + ...')
print('    4. 一次梯度上升: theta += alpha * grad log pi(a|s) * Gt')
print()
print('  与Q-Learning的区别:')
print('    Q-Learning: 每步用TD更新 (低方差,有偏)')
print('    REINFORCE:  整个episode用MC更新 (高方差,无偏)')
print()
print('【测试环境】CartPole-v1')
print('  状态: 连续4维 [车位置, 车速, 杆角度, 杆角速度]')
print('  动作: {0=左推, 1=右推}')
print('  目标: 保持竖杆不倒 (每步奖励+1)')
print('  需要神经网络 —— 状态连续，无法用Q表')
print()

import torch
mod = load_module('reinforce', os.path.join(ROOT, 'chapter_01_foundation', '03_reinforce', 'reinforce.py'))
REINFORCE = mod.REINFORCE
REINFORCEWithBaseline = mod.REINFORCEWithBaseline

np.random.seed(42)
torch.manual_seed(42)

agent_r = REINFORCE(state_dim=4, action_dim=2)
for step in range(10):
    s = np.random.randn(4).astype(np.float32)
    a = agent_r.select_action(s)
    agent_r.store_reward(1.0)
loss = agent_r.update()
print(f'  REINFORCE策略损失: {loss:.4f}')

agent_rb = REINFORCEWithBaseline(state_dim=4, action_dim=2)
for step in range(10):
    s = np.random.randn(4).astype(np.float32)
    a, ent = agent_rb.select_action(s)
    agent_rb.store_reward(1.0)
p_loss, v_loss = agent_rb.update()
print(f'  REINFORCE+Baseline: policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}')
print('  REINFORCE => OK')

print()
print('=' * 70)
print('第一章全部通过! Q-Learning + SARSA + REINFORCE')
print('=' * 70)
