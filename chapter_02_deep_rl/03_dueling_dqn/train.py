"""
Dueling DQN 训练脚本
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

SEED = 42
from dueling_dqn import DuelingDQN


def train(env_id="CartPole-v1", episodes=500, lr=1e-3, gamma=0.99):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(env_id)
    env.reset(seed=SEED)
    
    agent = DuelingDQN(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=lr,
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
            agent.update()
        
        rewards.append(episode_reward)
        
        if ep % 50 == 0:
            print(f"Episode {ep:5d} | Avg Reward: {np.mean(rewards[-50:]):8.2f}")
    
    env.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), color='red')
    plt.title("Dueling DQN Learning Curve")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"dueling_dqn_{env_id}.png"), dpi=150)
    plt.show()
    print(f"\nFinal 50-episode average: {np.mean(rewards[-50:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train(args.env, args.episodes, args.lr, args.gamma)
