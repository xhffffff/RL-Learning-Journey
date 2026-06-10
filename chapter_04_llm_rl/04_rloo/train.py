"""
RLOO 演示脚本
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rloo import demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLOO Demo")
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()
    
    stats = demo()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].plot(stats['loss'])
    axes[0].set_title('RLOO Loss')
    axes[0].set_xlabel('Step')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stats['reward'], color='green')
    axes[1].set_title('Rule-based Reward')
    axes[1].set_xlabel('Step')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(stats['kl'], color='purple')
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('RLOO Training Dynamics (REINFORCE Leave-One-Out)')
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "rloo_demo.png"), dpi=150)
    plt.show()
