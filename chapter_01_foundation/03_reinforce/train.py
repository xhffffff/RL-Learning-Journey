"""
REINFORCE 训练脚本
运行命令:
    python train.py --env CartPole-v1 --episodes 2000
    python train.py --env CartPole-v1 --baseline --episodes 2000
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

SEED = 42
from reinforce import REINFORCE, REINFORCEWithBaseline


def train_reinforce(
    env_id: str = "CartPole-v1",
    episodes: int = 2000,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    use_baseline: bool = True
):
    env = gym.make(env_id)
    env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if use_baseline:
        agent = REINFORCEWithBaseline(
            state_dim, action_dim,
            learning_rate=learning_rate,
            gamma=gamma
        )
        algo_name = "REINFORCE with Baseline"
    else:
        agent = REINFORCE(
            state_dim, action_dim,
            learning_rate=learning_rate,
            gamma=gamma
        )
        algo_name = "REINFORCE"
    
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if use_baseline:
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            
            state = next_state
            episode_reward += reward
        
        if use_baseline:
            p_loss, v_loss = agent.update()
        else:
            agent.update()
        
        rewards_history.append(episode_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"[{algo_name}] Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:6.1f}"
            )
    
    env.close()
    return rewards_history, algo_name


def plot_results(rewards, algo_name, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='purple', label='Raw')
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(range(49, len(rewards)), smoothed, color='red', label='Smoothed (50)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"{algo_name} Learning Curve")
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
    parser = argparse.ArgumentParser(description="REINFORCE Training")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--baseline", action="store_true",
                       help="Use REINFORCE with baseline")
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"{'='*60}")
    print(f"REINFORCE{' with Baseline' if args.baseline else ''} Training on {args.env}")
    print(f"{'='*60}\n")
    
    rewards, algo_name = train_reinforce(
        env_id=args.env,
        episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        use_baseline=args.baseline
    )
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "baseline" if args.baseline else "vanilla"
    save_path = os.path.join(results_dir, f"reinforce_{suffix}_{args.env}.png")
    plot_results(rewards, algo_name, save_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    final_avg = np.mean(rewards[-100:])
    print(f"Final 100-episode average reward: {final_avg:.2f}")
    print(f"{'='*60}")
