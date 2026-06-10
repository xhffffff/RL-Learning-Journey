"""
Q-Learning 可视化脚本
包含: 学习曲线、Q-Table 热力图、策略可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from q_learning import QLearning
import gymnasium as gym


def visualize_q_table(q_table, env_shape=(4, 4), save_path=None):
    """
    可视化 Q-Table 为热力图
    每个格子显示4个动作的颜色深浅
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Q-Table Visualization (FrozenLake 4x4)', fontsize=16, fontweight='bold')
    
    action_names = ['Up (0)', 'Right (1)', 'Down (2)', 'Left (3)']
    
    for action_idx, (ax, action_name) in enumerate(zip(axes.flat, action_names)):
        # 提取该动作的Q值并 reshape 为 4x4
        q_values = q_table[:, action_idx].reshape(env_shape)
        
        im = ax.imshow(q_values, cmap='RdYlGn', aspect='auto', vmin=q_values.min(), vmax=q_values.max())
        ax.set_title(action_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(env_shape[1]))
        ax.set_yticks(range(env_shape[0]))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # 在每个格子里显示数值
        for i in range(env_shape[0]):
            for j in range(env_shape[1]):
                state_idx = i * env_shape[1] + j
                text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Q-Value')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Q-Table 热力图已保存: {save_path}")
    plt.show()


def visualize_policy(q_table, env_shape=(4, 4), save_path=None):
    """
    可视化最优策略（箭头图）
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 获取最优策略
    policy = np.argmax(q_table, axis=1)
    
    # 定义障碍物和目标（FrozenLake 4x4）
    holes = [5, 7, 11, 12]  # H 的位置
    goal = 15  # G 的位置
    start = 0  # S 的位置
    
    # 绘制网格
    for i in range(env_shape[0] + 1):
        ax.axhline(i, color='black', linewidth=2)
    for j in range(env_shape[1] + 1):
        ax.axvline(j, color='black', linewidth=2)
    
    # 绘制每个格子
    for state in range(env_shape[0] * env_shape[1]):
        row = state // env_shape[1]
        col = state % env_shape[1]
        
        # 设置背景色
        if state == start:
            color = 'lightgreen'
            label = 'S'
        elif state == goal:
            color = 'gold'
            label = 'G'
        elif state in holes:
            color = 'lightcoral'
            label = 'H'
        else:
            color = 'white'
            label = ''
        
        rect = plt.Rectangle((col, env_shape[0] - row - 1), 1, 1, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # 添加标签
        if label:
            ax.text(col + 0.5, env_shape[0] - row - 0.5, label, 
                   ha='center', va='center', fontsize=20, fontweight='bold')
        
        # 如果不是终止状态，绘制箭头
        if state not in holes and state != goal:
            action = policy[state]
            dx, dy = 0, 0
            if action == 0:    # 上
                dy = 0.3
            elif action == 1:  # 右
                dx = 0.3
            elif action == 2:  # 下
                dy = -0.3
            elif action == 3:  # 左
                dx = -0.3
            
            ax.arrow(col + 0.5, env_shape[0] - row - 0.5, dx, dy,
                    head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    
    ax.set_xlim(0, env_shape[1])
    ax.set_ylim(0, env_shape[0])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Learned Policy Visualization\n(Blue arrows show the best action for each state)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Start (S)'),
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Goal (G)'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Hole (H)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Safe'),
        plt.Line2D([0], [0], color='blue', linewidth=2, marker='>', label='Best Action')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"策略可视化图已保存: {save_path}")
    plt.show()


def plot_learning_curve(rewards, window=100, save_path=None):
    """
    绘制学习曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Q-Learning Training Progress', fontsize=14, fontweight='bold')
    
    # 原始奖励
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                color='red', linewidth=2, label=f'Smoothed ({window} ep)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成功率
    ax2 = axes[1]
    success_rate = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        recent = rewards[start:i+1]
        success_rate.append(np.mean([r > 0 for r in recent]) * 100)
    
    ax2.plot(success_rate, color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate (Last {window} Episodes)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"学习曲线已保存: {save_path}")
    plt.show()


def train_and_visualize(episodes=5000, visualize_interval=1000):
    """
    训练并实时可视化
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)
    
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1.0 / (episodes * 0.5)
    )
    
    rewards_history = []
    success_rate_history = []
    
    print("=" * 60)
    print("Q-Learning Training on FrozenLake-v1")
    print("=" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # 计算最近100个episode的成功率
        recent_success = np.mean([r > 0 for r in rewards_history[-100:]])
        success_rate_history.append(recent_success)
        
        # 定期输出进度
        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode:5d} | Avg Reward: {avg_reward:.3f} | "
                  f"Success Rate: {recent_success:.1%} | Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Final Success Rate: {success_rate_history[-1]:.1%}")
    print(f"Average Reward (last 100): {np.mean(rewards_history[-100:]):.3f}")
    
    # 生成可视化
    os.makedirs('results', exist_ok=True)
    
    print("\n生成可视化...")
    plot_learning_curve(rewards_history, save_path='results/learning_curve.png')
    visualize_q_table(agent.q_table, save_path='results/q_table_heatmap.png')
    visualize_policy(agent.q_table, save_path='results/policy_visualization.png')
    
    return agent, rewards_history


if __name__ == "__main__":
    # 训练并可视化
    agent, rewards = train_and_visualize(episodes=3000)
