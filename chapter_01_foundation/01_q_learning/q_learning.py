"""
Q-Learning 算法实现 (1989)
-------------------------------
论文: Learning from Delayed Rewards (Chris Watkins, 1989)

核心思想:
- 离策略 (Off-Policy) 时序差分控制
- 直接学习最优动作价值函数 Q*(s,a)
- 无论实际采取什么策略，都学习最优策略的Q值

算法流程:
1. 初始化 Q(s,a) 表
2. 对于每个 episode:
   a. 观察当前状态 s
   b. 选择动作 a (epsilon-greedy)
   c. 执行 a，观察奖励 r 和下一个状态 s'
   d. 更新: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
   e. s ← s'，直到终止
"""

import numpy as np
import random
from typing import Tuple, Dict, Any


class QLearning:
    """
    Q-Learning 智能体
    
    使用Q表存储状态-动作价值，适用于离散状态和动作空间
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.001
    ):
        """
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率 α
            gamma: 折扣因子 γ
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 初始化Q表
        self.q_table = np.zeros((n_states, n_actions))
        
        # 统计信息
        self.episode_count = 0
    
    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Epsilon-Greedy 动作选择
        
        Args:
            state: 当前状态
            eval_mode: 是否为评估模式（不探索）
        """
        if eval_mode:
            return int(np.argmax(self.q_table[state]))
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Q-Learning 核心更新
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Returns:
            td_error: 时序差分误差
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon - self.epsilon_decay
        )
    
    def get_policy(self) -> np.ndarray:
        """获取当前贪婪策略"""
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath: str):
        """保存Q表"""
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str):
        """加载Q表"""
        self.q_table = np.load(filepath)


class DiscretizedQLearning:
    """
    连续状态空间的Q-Learning
    将连续状态离散化为有限个桶(bucket)
    """
    
    def __init__(
        self,
        state_bounds: np.ndarray,     # shape: (dim, 2) [[min, max], ...]
        n_buckets: Tuple[int, ...],    # 每个维度的桶数
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.001
    ):
        self.state_bounds = np.array(state_bounds)
        self.n_buckets = tuple(n_buckets)
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 创建多维Q表
        q_shape = n_buckets + (n_actions,)
        self.q_table = np.zeros(q_shape)
    
    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        将连续状态映射到离散桶
        
        Returns:
            多维索引元组
        """
        indices = []
        for i, (s, (low, high)) in enumerate(zip(state, self.state_bounds)):
            s_clipped = np.clip(s, low, high)
            bucket = int(
                (s_clipped - low) / (high - low) * self.n_buckets[i]
            )
            bucket = min(bucket, self.n_buckets[i] - 1)
            indices.append(bucket)
        return tuple(indices)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if eval_mode:
            idx = self.discretize(state)
            return int(np.argmax(self.q_table[idx]))
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            idx = self.discretize(state)
            return int(np.argmax(self.q_table[idx]))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        s_idx = self.discretize(state)
        ns_idx = self.discretize(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[ns_idx])
        
        td_error = target - self.q_table[s_idx][action]
        self.q_table[s_idx][action] += self.lr * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
