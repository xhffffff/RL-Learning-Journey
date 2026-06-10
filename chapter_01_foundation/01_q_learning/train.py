"""
Q-Learning 训练脚本
运行命令:
    python train.py --env CartPole-v1 --episodes 5000
    python train.py --env FrozenLake-v1 --episodes 20000
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

SEED = 42
# 导入本地模块
from q_learning import QLearning, DiscretizedQLearning


def train_q_learning_discrete(
    env_id: str = "FrozenLake-v1",
    episodes: int = 20000,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    render: bool = False,
    save_path: str = None
):
    """
    训练离散状态空间的Q-Learning
    """
    env = gym.make(env_id, is_slippery=True)
    env.reset(seed=SEED)
    
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1.0 / (episodes * 0.5)
    )
    
    rewards_history = []
    success_rate_history = []
    
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
        
        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = np.mean([r > 0 for r in rewards_history[-100:]])
            print(
                f"Episode {episode:6d} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Success Rate: {success_rate:.2%} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )
    
    env.close()
    return rewards_history


def train_q_learning_continuous(
    episodes: int = 5000,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    render: bool = False
):
    """
    训练连续状态空间的Q-Learning（CartPole）
    """
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    
    # 观测: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    state_bounds = np.array([
        [-2.4, 2.4],    # cart position
        [-3.0, 3.0],    # cart velocity
        [-0.5, 0.5],    # pole angle (radians, ~±30°)
        [-3.0, 3.0]     # pole angular velocity
    ])
    n_buckets = (8, 8, 8, 8)
    
    agent = DiscretizedQLearning(
        state_bounds=state_bounds,
        n_buckets=n_buckets,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1.0 / (episodes * 0.3)
    )
    
    rewards_history = []
    
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
            total_reward += 1
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:6.1f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )
    
    env.close()
    return rewards_history


def plot_results(rewards, title="Q-Learning Learning Curve", save_path=None):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(range(99, len(rewards)), smoothed, color='red', label='Smoothed (100)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(rewards)
    plt.plot(cumulative / np.arange(1, len(rewards)+1))
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Running Average')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning Training")
    parser.add_argument("--env", type=str, default="FrozenLake-v1",
                       choices=["CartPole-v1", "FrozenLake-v1"],
                       help="Environment to train on")
    parser.add_argument("--episodes", type=int, default=50000,
                       help="Number of episodes")
    parser.add_argument("--lr", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--save", type=str, default=None,
                       help="Save plot path")
    args = parser.parse_args()
    
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"{'='*60}")
    print(f"Q-Learning Training on {args.env}")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"{'='*60}\n")
    
    if args.env == "CartPole-v1":
        rewards = train_q_learning_continuous(
            episodes=args.episodes,
            learning_rate=args.lr,
            gamma=args.gamma
        )
    else:
        rewards = train_q_learning_discrete(
            env_id=args.env,
            episodes=args.episodes,
            learning_rate=args.lr,
            gamma=args.gamma
        )
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"q_learning_{args.env}.png")
    plot_results(rewards, f"Q-Learning on {args.env}", save_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final 100-episode average reward: {np.mean(rewards[-100:]):.2f}")
    print(f"{'='*60}")
