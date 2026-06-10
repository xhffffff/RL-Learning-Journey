"""
SARSA 训练脚本
运行命令:
    python train.py --env FrozenLake-v1 --episodes 20000
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

SEED = 42
from sarsa import SARSA, ExpectedSARSA


def train_sarsa(
    env_id: str = "FrozenLake-v1",
    episodes: int = 20000,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    use_expected: bool = False
):
    """
    训练SARSA / Expected SARSA
    """
    env = gym.make(env_id, is_slippery=True)
    env.reset(seed=SEED)
    
    if use_expected:
        agent = ExpectedSARSA(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=0.1
        )
        algo_name = "Expected SARSA"
    else:
        agent = SARSA(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=1.0 / (episodes * 0.5)
        )
        algo_name = "SARSA"
    
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # SARSA需要在开始时选择第一个动作
        if use_expected:
            action = agent.select_action(state)
        else:
            action = agent.select_action(state)
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 先选择下一个动作
            if use_expected:
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, done)
            else:
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action if next_action is not None else 0
            total_reward += reward
        
        if not use_expected:
            agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        
        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = np.mean([r > 0 for r in rewards_history[-100:]])
            print(
                f"[{algo_name}] Episode {episode:6d} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Success Rate: {success_rate:.2%}"
            )
    
    env.close()
    return rewards_history, algo_name


def plot_results(rewards, algo_name, save_path=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='green', label='Raw')
    smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(range(99, len(rewards)), smoothed, color='red', label='Smoothed (100)')
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
    parser = argparse.ArgumentParser(description="SARSA Training")
    parser.add_argument("--env", type=str, default="FrozenLake-v1",
                       help="Environment to train on")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expected", action="store_true",
                       help="Use Expected SARSA instead")
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"{'='*60}")
    print(f"{'Expected ' if args.expected else ''}SARSA Training on {args.env}")
    print(f"{'='*60}\n")
    
    rewards, algo_name = train_sarsa(
        env_id=args.env,
        episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        use_expected=args.expected
    )
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "expected_sarsa" if args.expected else "sarsa"
    save_path = os.path.join(results_dir, f"{suffix}_{args.env}.png")
    plot_results(rewards, algo_name, save_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final 100-episode average: {np.mean(rewards[-100:]):.2f}")
    print(f"{'='*60}")
