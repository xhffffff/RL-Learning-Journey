"""
TRPO 训练脚本
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

SEED = 42
from trpo import TRPO


def train(env_id="CartPole-v1", episodes=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(env_id)
    env.reset(seed=SEED)
    
    agent = TRPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
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
            agent.store_reward(reward, done)
            state = next_state
            episode_reward += reward
        
        losses = agent.update()
        rewards.append(episode_reward)
        
        if ep % 50 == 0:
            print(
                f"Episode {ep:5d} | "
                f"Avg: {np.mean(rewards[-50:]):8.2f} | "
                f"KL: {losses['kl']:.4f}"
            )
    
    env.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(rewards, alpha=0.3)
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(rewards)), smoothed, color='red')
    plt.title("TRPO Learning Curve")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"trpo_{env_id}.png"), dpi=150)
    plt.show()
    print(f"\nFinal 50-episode average: {np.mean(rewards[-50:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"TRPO Training on {args.env}\n")
    train(args.env, args.episodes)
