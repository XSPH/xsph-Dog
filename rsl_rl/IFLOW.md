# RSL RL 项目上下文

## 项目概述

RSL RL 是一个快速且简单的强化学习算法实现库，专为在 GPU 上完全运行而设计。该代码是 NVIDIA Isaac GYM 提供的 `rl-pytorch` 的演进版本。

**主要特点:**
- 目前实现了 PPO (Proximal Policy Optimization) 算法
- 专为 GPU 加速设计
- 支持循环和标准 Actor-Critic 架构
- 由苏黎世联邦理工学院机器人系统实验室和 NVIDIA 联合开发

**项目维护者:** Nikita Rudin  
**所属机构:** 苏黎世联邦理工学院机器人系统实验室 & NVIDIA  
**许可证:** BSD-3-Clause

## 技术栈

- **编程语言:** Python 3.6+
- **深度学习框架:** PyTorch (>=1.4.0)
- **核心依赖:**
  - torch>=1.4.0
  - torchvision>=0.5.0
  - numpy>=1.16.4

## 项目结构

```
rsl_rl/
├── algorithms/          # 强化学习算法实现
│   └── ppo.py          # PPO 算法实现
├── modules/            # 神经网络模块
│   ├── actor_critic.py           # 标准 Actor-Critic 网络
│   └── actor_critic_recurrent.py # 循环 Actor-Critic 网络
├── runners/            # 训练运行器
│   └── on_policy_runner.py       # 在线策略训练运行器
├── storage/            # 数据存储
│   └── rollout_storage.py        # 经验回放存储
├── env/                # 环境接口
│   └── vec_env.py      # 向量化环境接口
└── utils/              # 工具函数
    └── utils.py        # 通用工具函数
```

## 安装和设置

### 安装步骤
```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

### 依赖要求
- Python >= 3.6
- PyTorch >= 1.4.0
- NumPy >= 1.16.4

## 核心组件

### 1. 算法 (Algorithms)
- **PPO (Proximal Policy Optimization):** 目前唯一实现的算法，位于 `rsl_rl/algorithms/ppo.py`

### 2. 网络模块 (Modules)
- **ActorCritic:** 标准的 Actor-Critic 神经网络架构
- **ActorCriticRecurrent:** 支持循环网络的 Actor-Critic 架构，适用于需要记忆的任务

### 3. 训练运行器 (Runners)
- **OnPolicyRunner:** 在线策略训练的主运行器，负责整个训练流程的协调

### 4. 存储系统 (Storage)
- **RolloutStorage:** 用于存储和管理训练过程中的经验数据

### 5. 环境接口 (Environment)
- **VecEnv:** 向量化环境的抽象接口，支持并行环境交互

## 开发约定

### 代码风格
- 所有文件使用 SPDX 许可证头
- 遵循 Python PEP 8 编码规范
- 使用类型提示 (Type Hints)

### 项目架构模式
- 模块化设计：算法、网络、运行器、存储分离
- 面向接口编程：通过抽象接口实现组件解耦
- GPU 优先：所有计算设计为在 GPU 上高效运行

### 测试和验证
- 目前项目未包含测试文件
- 建议在修改后进行功能验证

## 相关资源

- **示例用法:** https://github.com/leggedrobotics/legged_gym
- **项目网站:** https://leggedrobotics.github.io/legged_gym/
- **相关论文:** https://arxiv.org/abs/2109.11978

## 常用命令

### 安装项目
```bash
pip install -e .
```

### 运行示例
该项目主要作为库使用，具体运行命令取决于使用该库的上层应用（如 legged_gym）。

## 扩展和贡献

项目欢迎贡献，特别是：
- 新的强化学习算法实现
- 性能优化
- 文档改进
- 错误修复

## 注意事项

- 该项目主要设计用于机器人强化学习任务
- 需要配合相应的环境使用（如 legged_gym）
- 所有代码都针对 GPU 计算进行了优化