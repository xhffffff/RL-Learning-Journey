"""
第三章测试: TRPO, TD3, MuZero, Dreamer
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
print('第三章 高级算法 (2015-2023)')
print('=' * 70)

# ============ 1. TRPO ============
print()
print('--- 1. TRPO (2015, Schulman et al.) ---')
print()
print('【数据获取方式】')
print('  On-Policy 在线收集 (与PPO相同的基础)')
print('  收集 rollout -> 计算GAE优势 -> 优化')
print('  关键区别: TRPO用KL散度约束 + 共轭梯度(CG)二阶优化')
print()
print('  TRPO保证: 策略单调提升')
print('  PPO简化为: 裁剪目标函数 (更简单实用)')
print('  TRPO = PPO的理论前身')
print()
print('【测试环境】CartPole-v1')
print()

mod = load_module('trpo', os.path.join(ROOT, 'chapter_03_advanced', '01_trpo', 'trpo.py'))
TRPO = mod.TRPO

agent1 = TRPO(state_dim=4, action_dim=2, device='cpu')
for _ in range(64):
    s = np.random.randn(4).astype(np.float32)
    a = agent1.select_action(s)
    agent1.store_reward(1.0, False)
info = agent1.update()
print(f'  TRPO Actor Loss: {info["actor_loss"]:.4f}')
print(f'  TRPO Critic Loss: {info["critic_loss"]:.4f}')
print(f'  TRPO KL Divergence: {info["kl"]:.4f}')
print('  TRPO => OK')

# ============ 2. TD3 ============
print()
print('--- 2. TD3 (2018, Fujimoto et al.) ---')
print()
print('【数据获取方式】')
print('  Off-Policy + ReplayBuffer (与DDPG/SAC相同)')
print('  数据格式: (s, a, r, s_next, done) 存入Buffer')
print()
print('  TD3三大改进 (相对DDPG):')
print('    1. Clipped Double Q: 用2个Q网络的最小值 (减少过估计)')
print('    2. Delayed Policy: 策略更新慢于Critic (policy_freq=2)')
print('    3. Target Smoothing: 目标动作加噪声 (平滑Q函数)')
print()
print('【测试环境】Pendulum-v1 (连续动作)')
print()

mod = load_module('td3', os.path.join(ROOT, 'chapter_03_advanced', '02_td3', 'td3.py'))
TD3 = mod.TD3

agent2 = TD3(state_dim=3, action_dim=1, device='cpu')
for i in range(256):
    s = np.random.randn(3).astype(np.float32)
    ns = np.random.randn(3).astype(np.float32)
    agent2.store_transition(s, np.random.randn(1).astype(np.float32), -1.0, ns, False)
info = agent2.update()
if info:
    print(f'  TD3 Critic Loss: {info["critic_loss"]:.4f}')
print('  TD3 => OK')

# ============ 3. MuZero ============
print()
print('--- 3. MuZero (2019, DeepMind) ---')
print()
print('【数据获取方式 —— 与前面的完全不同!】')
print('  MuZero自身学习世界模型，不完全依赖环境交互!')
print('  三大组件:')
print('    表示函数 h(o) -> s: 原始观测 -> 隐藏状态')
print('    动态函数 g(s,a) -> (r,s_next): 内部世界模型')
print('    预测函数 f(s) -> (p,v): 策略+价值')
print()
print('  训练数据来源:')
print('    1. 真实环境交互 (少量)')
print('    2. MCTS搜索: 在学到的世界模型中展开搜索')
print('    3. 自对弈: 自我博弈产生训练数据')
print()
print('  这是AlphaZero的通用版本!')
print('  不需要知道环境规则，自己学世界模型')
print()
print('【测试环境】Atari / 围棋 / 国际象棋 / 将棋')
print('  (本实现演示核心网络组件)')
print()

mod = load_module('muzero', os.path.join(ROOT, 'chapter_03_advanced', '03_muzero', 'muzero.py'))
MuZeroNetwork = mod.MuZeroNetwork

obs_shape = (3, 96, 96)
action_dim = 4
model = MuZeroNetwork(obs_shape, action_dim)
obs = torch.randn(1, *obs_shape)

hidden, policy, value = model.initial_inference(obs)
print(f'  MuZero 输入: 图像 {obs_shape}')
print(f'  隐藏状态: {hidden.shape}')
print(f'  策略输出: {policy.shape}')
print(f'  价值输出: {value.shape}')

action = torch.eye(action_dim)[0:1]
reward, next_h, next_p, next_v = model.recurrent_inference(hidden, action)
print(f'  展开后奖励: {reward.shape}')
print(f'  展开后隐藏: {next_h.shape}')
print('  MuZero => OK')

# ============ 4. Dreamer ============
print()
print('--- 4. Dreamer (2020-2023, Hafner et al.) ---')
print()
print('【数据获取方式 —— 在"梦境"中学习!】')
print('  Dreamer = 世界模型 + 隐空间想象')
print('  核心流程:')
print('    1. 用少量真实交互学习世界模型 (RSSM)')
print('    2. 在世界模型的隐空间中"想象" rollout')
print('    3. 从想象的轨迹中学习策略 (Actor/Critic)')
print()
print('  RSSM状态分解:')
print('    h(t) 确定性状态 (通过GRU)')
print('    z(t) 随机状态   (通过离散/连续分布)')
print()
print('  数据流: s -> h -> z -> 解码(recon) -> 奖励预测')
print('  优势: 样本效率极高! 少量真实交互即可')
print()
print('【测试环境】DeepMind Control Suite / Atari')
print('  (本实现演示RSSM核心组件)')
print()

mod = load_module('dreamer', os.path.join(ROOT, 'chapter_03_advanced', '04_dreamer', 'dreamer.py'))
DreamerWorldModel = mod.DreamerWorldModel

obs_dim = 64
action_dim = 4
model2 = DreamerWorldModel(obs_dim, action_dim)
batch_size = 2

obs = torch.randn(batch_size, obs_dim)
embed = model2.encoder(obs)
print(f'  Dreamer 编码: {obs.shape} -> {embed.shape}')

h = torch.zeros(batch_size, 200)
prior = model2.rssm.prior(h)
z = prior.sample()
print(f'  RSSM: z ~ p(z|h), z shape: {z.shape}')

actions = torch.randn(10, batch_size, action_dim)
h_im, z_im, r_im = model2.imagine(h, z, actions)
print(f'  想象10步: h={h_im.shape}, z={z_im.shape}, r={r_im.shape}')
print('  Dreamer => OK')

print()
print('=' * 70)
print('第三章全部通过! TRPO -> TD3 -> MuZero -> Dreamer')
print('=' * 70)
