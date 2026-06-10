"""
SAC 训练脚本
运行命令:
    python train.py --env Pendulum-v1 --episodes 200
    python train.py --env MountainCarContinuous-v1 --episodes 300
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

SEED = 42
from sac import SAC


def train(env_id="Pendulum-v1", episodes=200, lr=3e-4, gamma=0.99):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(env_id)
    env.reset(seed=SEED)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device
    )
    
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            info = agent.update()
        
        rewards.append(episode_reward)
        
        if ep % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(
                f"Episode {ep:5d} | "
                f"Avg: {avg:8.2f} | "
                f"Alpha: {agent.alpha:.3f}"
            )
    
    env.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(rewards, alpha=0.3)
    smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(rewards)), smoothed, color='red')
    plt.title(f"SAC on {env_id}")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"sac_{env_id}.png"), dpi=150)
    plt.show()
    print(f"\nFinal 10-episode average: {np.mean(rewards[-10:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"{'='*60}")
    print(f"SAC Training on {args.env}")
    print(f"{'='*60}\n")
    
    train(args.env, args.episodes, args.lr, args.gamma)
