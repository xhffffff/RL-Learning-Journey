"""
RLHF 演示脚本
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from rlhf import demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF Demo")
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()
    
    stats = demo()
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].plot(stats['reward_rm'])
    axes[0].set_title('RM Reward')
    axes[0].set_xlabel('Step')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stats['kl_penalty'])
    axes[1].set_title('KL Penalty')
    axes[1].set_xlabel('Step')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(stats['total_reward'])
    axes[2].set_title('Total Reward')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('RLHF Training Dynamics')
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "rlhf_demo.png"), dpi=150)
    plt.show()
