"""
DQN 训练脚本
运行命令:
    python train.py --env CartPole-v1 --episodes 500
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

SEED = 42
from dqn import DQN


def train_dqn(
    env_id: str = "CartPole-v1",
    episodes: int = 500,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    hidden_dim: int = 128,
    batch_size: int = 64,
    target_update: int = 10,
    render_eval: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    env = gym.make(env_id)
    env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update=target_update,
        batch_size=batch_size,
        device=device
    )
    
    rewards_history = []
    best_avg_reward = float('-inf')
    
    for episode in range(episodes):
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
            
            loss = agent.update()
        
        rewards_history.append(episode_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Best: {best_avg_reward:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.memory)}"
            )
    
    env.close()
    return rewards_history


def evaluate(agent, env_id: str, episodes: int = 10):
    """评估训练好的智能体"""
    env = gym.make(env_id, render_mode="human")
    env.reset(seed=SEED)
    total_rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Eval Episode {ep+1}: Reward = {episode_reward}")
    
    env.close()
    print(f"\nEvaluation: Mean={np.mean(total_rewards):.2f}, Std={np.std(total_rewards):.2f}")


def plot_results(rewards, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(rewards)), smoothed, color='red', label='Smoothed (50)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1))
    plt.xlabel('Episode')
    plt.ylabel('Running Average')
    plt.title('Running Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"{'='*60}")
    print(f"DQN Training on {args.env}")
    print(f"{'='*60}\n")
    
    rewards = train_dqn(
        env_id=args.env,
        episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        batch_size=args.batch_size,
        target_update=args.target_update,
        render_eval=args.eval
    )
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    save_path = os.path.join(results_dir, f"dqn_{args.env}.png")
    plot_results(rewards, save_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final 50-episode average reward: {np.mean(rewards[-50:]):.2f}")
    print(f"{'='*60}")
