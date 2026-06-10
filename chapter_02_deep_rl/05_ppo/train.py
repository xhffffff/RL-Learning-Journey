"""
PPO 训练脚本
运行命令:
    python train.py --env CartPole-v1 --episodes 1000
    python train.py --env Pendulum-v1 --continuous --episodes 500
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

SEED = 42
from ppo import PPO


def train(env_id="CartPole-v1", episodes=1000, lr=3e-4, gamma=0.99, continuous=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(env_id)
    env.reset(seed=SEED)
    
    state_dim = env.observation_space.shape[0]
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        continuous=continuous,
        device=device
    )
    
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action if not continuous else [action])
            done = terminated or truncated
            agent.store_reward(reward, done)
            state = next_state
            episode_reward += reward
        
        losses = agent.update()
        rewards.append(episode_reward)
        
        if ep % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print(
                f"Episode {ep:5d} | "
                f"Avg: {avg_reward:8.2f} | "
                f"Policy: {losses['policy']:.4f} | "
                f"Value: {losses['value']:.4f}"
            )
    
    env.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(rewards, alpha=0.3)
    smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
    plt.plot(range(19, len(rewards)), smoothed, color='red')
    plt.title(f"PPO on {env_id}")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"ppo_{env_id}.png"), dpi=150)
    plt.show()
    print(f"\nFinal 50-episode average: {np.mean(rewards[-50:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"{'='*60}")
    print(f"PPO Training on {args.env}")
    print(f"{'='*60}\n")
    
    train(args.env, args.episodes, args.lr, args.gamma, args.continuous)
