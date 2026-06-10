from setuptools import setup, find_packages

setup(
    name="rl-learning-journey",
    version="1.0.0",
    description="强化学习算法学习项目 - 从Q-Learning到DAPO（16个核心算法，覆盖经典RL到LLM对齐）",
    author="RL Learner",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
