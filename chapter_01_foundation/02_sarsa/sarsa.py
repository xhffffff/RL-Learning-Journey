"""
SARSA 算法实现 (1994)
-------------------------------
论文: On-Line Q-Learning Using Connectionist Systems
作者: Gavin A. Rummery, Mahesan Niranjan

核心思想:
- 同策略 (On-Policy) 时序差分控制
- 学习实际执行策略的Q函数
- 更新公式: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

与Q-Learning的区别:
- Q-Learning: max_a' Q(s',a')
- SARSA: Q(s', a') (使用实际采取的下一个动作)
"""

import numpy as np
import random
from typing import Tuple


class SARSA:
    """
    SARSA 智能体
    
    名称由来: (State, Action, Reward, State, Action)
    学习 agent 实际遵循的 Q 值，更加保守和稳定
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
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_table = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, eval_mode: bool = False) -> int:
        if eval_mode:
            return int(np.argmax(self.q_table[state]))
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool
    ) -> float:
        """
        SARSA 更新规则
        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        
        注意: 使用实际采取的 next_action，而非 max
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
    
    def save(self, filepath: str):
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str):
        self.q_table = np.load(filepath)


class ExpectedSARSA:
    """
    Expected SARSA (期望SARSA)
    
    更新公式:
    Q(s,a) ← Q(s,a) + α[r + γ·Σ_a' π(a'|s') Q(s',a') - Q(s,a)]
    
    特点:
    - 介于Q-Learning和SARSA之间
    - 使用期望值而非max或采样值
    - 方差更小
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        if done:
            target = reward
        else:
            # 计算期望Q值
            best_action = np.argmax(self.q_table[next_state])
            prob_best = 1 - self.epsilon + self.epsilon / self.n_actions
            prob_other = self.epsilon / self.n_actions
            
            expected_q = 0.0
            for a in range(self.n_actions):
                if a == best_action:
                    expected_q += prob_best * self.q_table[next_state, a]
                else:
                    expected_q += prob_other * self.q_table[next_state, a]
            
            target = reward + self.gamma * expected_q
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        return abs(td_error)
