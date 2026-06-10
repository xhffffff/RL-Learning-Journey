"""
强化学习智能体基类
为所有算法提供统一的接口

注意: 此接口为可选参考接口，项目中的算法实现不一定从此基类继承。
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
import numpy as np


class BaseAgent(ABC):
    """所有强化学习智能体的抽象基类"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.training = True

    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        pass

    def save(self, filepath: str):
        raise NotImplementedError

    def load(self, filepath: str):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
