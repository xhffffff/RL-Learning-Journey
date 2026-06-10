"""
DAPO 演示脚本
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dapo import demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPO Demo")
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()
    
    stats = demo()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(stats['loss'], color='blue')
    axes[0, 0].set_title('DAPO Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(stats['reward'], color='green')
    axes[0, 1].set_title('Rule-based Reward')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(stats['kl'], color='purple')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('KL')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(stats['kept_ratio'], color='orange')
    axes[1, 1].set_title('Kept Ratio (Dynamic Sampling)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Kept Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('DAPO Training Dynamics (ByteDance Seed + Tsinghua AIR, 2025)')
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "dapo_demo.png"), dpi=150)
    plt.show()
