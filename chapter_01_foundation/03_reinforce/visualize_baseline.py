"""
REINFORCE with Baseline 可视化
对比展示 Baseline 如何改善学习
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from reinforce import REINFORCE, REINFORCEWithBaseline


def train_both(episodes=2000):
    """对比训练两种算法"""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 训练 REINFORCE
    print("Training REINFORCE...")
    agent1 = REINFORCE(state_dim, action_dim, learning_rate=0.01)
    rewards1 = train_agent(agent1, env, episodes)
    
    # 训练 REINFORCE with Baseline
    print("\nTraining REINFORCE with Baseline...")
    agent2 = REINFORCEWithBaseline(state_dim, action_dim, learning_rate=0.01)
    rewards2 = train_agent_baseline(agent2, env, episodes)
    
    # 对比可视化
    plot_comparison(rewards1, rewards2)
    
    return agent1, agent2, rewards1, rewards2


def train_agent(agent, env, episodes):
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent.policy_network(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            agent.states.append(state_tensor)
            agent.actions.append(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            agent.rewards.append(reward)
            episode_reward += reward
            state = next_state
        
        agent.update()
        rewards_history.append(episode_reward)
        
        if episode % 200 == 0:
            avg = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"  Episode {episode}: Avg Reward = {avg:.1f}")
    
    return rewards_history


def train_agent_baseline(agent, env, episodes):
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = agent.policy_network(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            agent.states.append(state_tensor)
            agent.actions.append(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            agent.rewards.append(reward)
            episode_reward += reward
            state = next_state
        
        agent.update()
        rewards_history.append(episode_reward)
        
        if episode % 200 == 0:
            avg = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"  Episode {episode}: Avg Reward = {avg:.1f}")
    
    return rewards_history


def plot_comparison(rewards1, rewards2):
    plt.figure(figsize=(12, 5))
    
    # 原始奖励
    plt.subplot(1, 2, 1)
    plt.plot(rewards1, alpha=0.2, color='blue', label='REINFORCE')
    plt.plot(rewards2, alpha=0.2, color='red', label='REINFORCE+Baseline')
    
    # 平滑曲线
    window = 50
    if len(rewards1) >= window:
        smooth1 = np.convolve(rewards1, np.ones(window)/window, mode='valid')
        smooth2 = np.convolve(rewards2, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards1)), smooth1, color='blue', linewidth=2, label='REINFORCE (smoothed)')
        plt.plot(range(window-1, len(rewards2)), smooth2, color='red', linewidth=2, label='REINFORCE+Baseline (smoothed)')
    
    plt.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max (500)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE vs REINFORCE+Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 累积平均
    plt.subplot(1, 2, 2)
    cumavg1 = np.cumsum(rewards1) / np.arange(1, len(rewards1) + 1)
    cumavg2 = np.cumsum(rewards2) / np.arange(1, len(rewards2) + 1)
    plt.plot(cumavg1, color='blue', label='REINFORCE')
    plt.plot(cumavg2, color='red', label='REINFORCE+Baseline')
    plt.axhline(y=500, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Average Reward')
    plt.title('Learning Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存: results/comparison.png")
    plt.show()


if __name__ == "__main__":
    train_both(episodes=2000)
