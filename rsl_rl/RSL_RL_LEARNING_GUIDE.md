# RSL RL 学习指南

## 📚 项目概述

**RSL RL** 是由ETH Zurich机器人系统实验室和NVIDIA联合开发的一个高性能强化学习库。项目致力于提供**快速、简洁的RL算法实现**，设计用于在GPU上完全运行。

### 核心特点
- 🚀 **GPU优化**：完全在GPU上运行，适合大规模并行化训练
- 🎯 **算法实现**：当前主要实现PPO（近端策略优化）算法
- 📊 **灵活框架**：支持从业位化策略到循环神经网络（RNN）的各种网络结构
- 🏗️ **模块化设计**：清晰的模块划分，便于扩展和定制

### 项目信息
- **版本**：1.0.2
- **维护者**：Nikita Rudin
- **许可证**：BSD-3-Clause
- **关联项目**：[legged_gym](https://github.com/leggedrobotics/legged_gym) - 四足机器人学习环境

---

## 📁 项目结构分析

```
rsl_rl/
├── algorithms/          # 强化学习算法实现
│   └── ppo.py          # PPO算法核心实现
├── env/                # 环境接口定义
│   └── vec_env.py      # 向量化环境的抽象基类
├── modules/            # 神经网络模块
│   ├── actor_critic.py              # 标准的Actor-Critic网络
│   └── actor_critic_recurrent.py    # 循环神经网络版本
├── runners/            # 训练运行器
│   └── on_policy_runner.py  # 同策略算法的训练器
├── storage/            # 数据存储
│   └── rollout_storage.py   # 经验回放存储
└── utils/              # 工具函数
    └── utils.py        # 辅助函数（轨迹处理等）
```

---

## 🔑 核心模块详解

### 1. **算法模块 (algorithms/ppo.py)**

#### 🎓 PPO算法简介
PPO（Proximal Policy Optimization）是一种**同策略**强化学习算法，通过限制策略更新的步长来提高训练稳定性。

#### 主要类：`PPO`

**核心职责**：
- 策略和值函数的更新
- 梯度计算和优化
- 超参数管理（学习率、剪裁参数等）

**关键方法**：

| 方法 | 功能 | 说明 |
|------|------|------|
| `__init__()` | 初始化PPO | 配置所有超参数、优化器、网络 |
| `init_storage()` | 初始化存储 | 为回放缓冲区分配内存 |
| `act()` | 采集数据 | 根据观测选择动作并记录日志概率 |
| `process_env_step()` | 处理环境反馈 | 保存过渡数据到存储器 |
| `compute_returns()` | 计算回报 | 计算GAE优势估计 |
| `update()` | 更新策略 | 执行PPO的梯度更新步骤 |

**关键超参数**：

```python
# 策略优化参数
clip_param = 0.2              # 策略梯度剪裁范围
num_learning_epochs = 1       # 每个回合的学习次数
num_mini_batches = 1          # 小批量数量

# 奖励处理参数
gamma = 0.998                 # 折扣因子（未来回报的权重）
lam = 0.95                    # GAE Lambda（偏差-方差权衡）

# 损失权重
value_loss_coef = 1.0         # 值函数损失的权重
entropy_coef = 0.0            # 熵正则化系数（鼓励探索）

# 学习参数
learning_rate = 1e-3          # 学习率
max_grad_norm = 1.0           # 梯度剪裁的最大范数

# 自适应学习率
schedule = "fixed"            # "fixed" 或 "adaptive"
desired_kl = 0.01             # 目标KL散度（自适应模式）
```

**工作流程**：

```
1. 采集阶段 (act & process_env_step)
   ├─ 在环境中采集轨迹
   ├─ 记录状态、动作、奖励等
   └─ 存储到RolloutStorage

2. 计算阶段 (compute_returns)
   ├─ 计算累积回报
   ├─ 计算优势函数
   └─ 准备批数据

3. 更新阶段 (update)
   ├─ 小批量遍历数据
   ├─ 计算PPO损失函数
   ├─ 反向传播和梯度更新
   └─ 返回训练统计信息
```

---

### 2. **神经网络模块 (modules/)**

#### 2.1 标准Actor-Critic (actor_critic.py)

**架构设计**：

```
输入观测
    ↓
Actor (策略网络)          Critic (价值网络)
├─ FC层                   ├─ FC层
├─ 激活函数               ├─ 激活函数
├─ FC层                   ├─ FC层
├─ 激活函数               ├─ 激活函数
└─ 输出层 → 动作均值      └─ 输出层 → 状态价值
```

**类：`ActorCritic`**

| 属性 | 说明 |
|------|------|
| `is_recurrent` | False（非循环网络标记） |
| `actor` | 策略网络（输出动作均值） |
| `critic` | 值函数网络（输出状态价值） |
| `std` | 可学习参数，作为动作的标准差 |
| `distribution` | 动作分布（高斯分布） |

**关键方法**：

```python
# 数据流方法
act(observations)                    # 采样动作
act_inference(observations)          # 推理模式（使用均值）
evaluate(critic_observations)        # 评估状态价值

# 分布相关
update_distribution(observations)    # 更新高斯分布参数
get_actions_log_prob(actions)       # 计算动作对数概率

# 属性（动态计算）
@property action_mean                # 当前分布的均值
@property action_std                 # 当前分布的标准差
@property entropy                    # 分布熵（鼓励探索）
```

**激活函数支持**：
- `elu` - 指数线性单元（默认）
- `relu` - 修正线性单元
- `selu` - 缩放指数线性单元
- `tanh` - 双曲正切
- 等等...

#### 2.2 循环Actor-Critic (actor_critic_recurrent.py)

**用途**：处理部分可观察环境或需要记忆的任务

**类：`ActorCriticRecurrent`**

继承自 `ActorCritic`，添加了RNN层：

```python
观测序列 → RNN层 → 隐藏状态 → Actor/Critic网络 → 动作/价值
            ↑              ↓
            └──隐藏状态缓存─┘
```

**核心改动**：

| 组件 | 说明 |
|------|------|
| `memory_a` | Actor的RNN模块（GRU/LSTM） |
| `memory_c` | Critic的RNN模块（GRU/LSTM） |
| `rnn_type` | RNN类型（'gru'或'lstm'） |
| `rnn_hidden_size` | 隐藏状态维度 |
| `rnn_num_layers` | RNN层数 |

**Memory类工作原理**：

```python
# 推理模式（采集数据）
input → RNN → 输出 + 保存隐藏状态

# 批模式（策略更新）
使用保存的隐藏状态进行完整序列处理
```

---

### 3. **存储模块 (storage/rollout_storage.py)**

**职责**：缓存经验数据，支持高效的批处理

#### 过渡数据结构 (Transition)

```python
class Transition:
    observations          # 环境观测
    critic_observations   # Critic观测（可能是特权信息）
    actions              # 执行的动作
    rewards              # 环境奖励
    dones                # 是否回合结束
    values               # 状态价值估计
    actions_log_prob     # 动作对数概率
    action_mean          # 动作均值（用于KL计算）
    action_sigma         # 动作标准差
    hidden_states        # RNN隐藏状态（可选）
```

#### 存储缓冲区结构

```python
# 数据形状：(num_transitions_per_env, num_envs, *feature_shape)

observations[T, E, obs_dim]       # T个时间步，E个环境
actions[T, E, action_dim]         # 动作序列
rewards[T, E, 1]                  # 奖励
values[T, E, 1]                   # 价值估计
returns[T, E, 1]                  # 计算的累积回报
advantages[T, E, 1]               # GAE优势
```

**关键方法**：

| 方法 | 功能 |
|------|------|
| `add_transitions()` | 逐步添加过渡到缓冲区 |
| `compute_returns()` | 使用GAE计算回报和优势 |
| `mini_batch_generator()` | 生成标准小批量（非循环） |
| `reccurent_mini_batch_generator()` | 生成为RNN优化的小批量 |
| `clear()` | 清空缓冲区 |

**GAE (广义优势估计)**：
$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

其中：
- $\gamma$ = 折扣因子
- $\lambda$ = GAE系数
- $V$ = 价值函数

---

### 4. **环境接口 (env/vec_env.py)**

**设计模式**：抽象基类定义接口

#### VecEnv 接口

```python
class VecEnv(ABC):
    # 属性
    num_envs              # 并行环境数
    num_obs               # 观测维度
    num_privileged_obs    # 特权观测维度
    num_actions           # 动作维度
    max_episode_length    # 最大回合长度
    
    # 必要方法
    step(actions)         # 执行动作，返回观测、奖励等
    reset(env_ids)        # 重置环境
    get_observations()    # 获取当前观测
    get_privileged_observations()  # 获取特权观测（可选）
```

**特权观测**：
- 仅在训练Critic时使用
- 不在推理时使用
- 常用于机器人学习中的隐藏参数（质量、摩擦力等）

---

### 5. **训练器 (runners/on_policy_runner.py)**

**职责**：协调所有组件的训练流程

#### OnPolicyRunner 流程

```
初始化
  ├─ 创建环境和Agent
  ├─ 初始化PPO算法
  └─ 设置日志

循环训练
  ├─ 采集数据
  │  ├─ 与环境交互
  │  ├─ 存储过渡
  │  └─ 累计奖励统计
  ├─ 计算回报
  │  └─ 运行价值引导程序计算优势
  ├─ 更新策略
  │  ├─ 多个学习轮次
  │  └─ 小批量梯度更新
  └─ 日志记录
     └─ TensorBoard可视化

推理/评估
  └─ 使用`act_inference()`获得确定性动作
```

**关键配置**（来自 train_cfg）：

```python
runner:
  num_steps_per_env      # 每个环境采集的步数
  save_interval          # 保存检查点的间隔
  policy_class_name      # "ActorCritic" 或 "ActorCriticRecurrent"
  algorithm_class_name   # "PPO"

algorithm:
  # PPO超参数
  num_learning_epochs
  num_mini_batches
  clip_param
  # 等等...

policy:
  # 网络结构
  actor_hidden_dims      # Actor隐藏层维度列表
  critic_hidden_dims     # Critic隐藏层维度列表
  activation             # 激活函数
  # RNN参数（如果使用ActorCriticRecurrent）
  rnn_type
  rnn_hidden_size
```

---

### 6. **工具函数 (utils/utils.py)**

#### 轨迹处理工具

**split_and_pad_trajectories()**
- 目标：处理可变长度轨迹
- 输入：完整轨迹张量 + done标记
- 输出：填充的轨迹 + 有效性掩码
- 用途：为RNN准备批数据

**unpad_trajectories()**
- 反向操作
- 恢复原始轨迹形状
- 用于数据处理管道

---

## 🎯 数据流完整图

```
┌─────────────────────────────────────────────────────────────┐
│                     初始化阶段                              │
├─────────────────────────────────────────────────────────────┤
│ OnPolicyRunner  PPO  ActorCritic  RolloutStorage  VecEnv   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   采集阶段（每个Agent步）                   │
├─────────────────────────────────────────────────────────────┤
│ 1. obs ← env.get_observations()                             │
│ 2. action ← agent.act(obs)                  [采样]          │
│ 3. obs', reward, done ← env.step(action)    [执行]         │
│ 4. storage.add_transitions(transition)      [存储]         │
│ 5. 重复N步...                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   计算阶段（一个回合）                      │
├─────────────────────────────────────────────────────────────┤
│ 1. 最后价值 ← agent.evaluate(final_obs)                     │
│ 2. compute_returns(最后价值)  [GAE计算]                     │
│    ├─ R_t = r_t + γV(s_{t+1})                             │
│    └─ A_t = R_t - V(s_t)                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   更新阶段（多个轮次）                      │
├─────────────────────────────────────────────────────────────┤
│ 对于每个小批量(obs, action, adv, return, old_logprob):     │
│   1. logprob ← agent.act(obs).logprob()                    │
│   2. value ← agent.evaluate(obs)                           │
│   3. ratio = exp(logprob - old_logprob)                    │
│   4. 代理损失 = -min(ratio·adv, clip(ratio)·adv)          │
│   5. 价值损失 = (return - value)²                          │
│   6. 总损失 = 代理 + α·价值 - β·熵                         │
│   7. optimizer.step()                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
                         重复训练循环
```

---

## 📊 超参数指南

### 🎓 对初学者的建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 贪婪且需要长期规划 |
| `lam` | 0.95 | 平衡偏差和方差 |
| `clip_param` | 0.2 | 标准PPO设置 |
| `learning_rate` | 3e-4 | 从小值开始并根据需要增加 |
| `num_mini_batches` | 4 | 如果内存允许，可增加以获得更好的统计 |
| `entropy_coef` | 0.001 | 鼓励探索但不要过度 |

### 🔧 调优策略

1. **如果训练不稳定**：
   - 减小学习率
   - 增加clip_param（但不超过0.3）
   - 增加num_mini_batches

2. **如果收敛太慢**：
   - 增加学习率
   - 增加Actor/Critic网络规模
   - 减少clip_param

3. **如果过拟合**：
   - 增加entropy_coef
   - 减小网络规模
   - 增加环境随机性

---

## 🚀 使用示例工作流

### 基本训练流程

```python
# 1. 导入
from rsl_rl.runners import OnPolicyRunner

# 2. 配置（通常来自YAML）
train_cfg = {
    "runner": {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "policy_class_name": "ActorCritic",
        "algorithm_class_name": "PPO"
    },
    "algorithm": {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.998,
        "learning_rate": 1e-3,
    },
    "policy": {
        "actor_hidden_dims": [256, 256, 256],
        "critic_hidden_dims": [256, 256, 256],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
}

# 3. 创建运行器
runner = OnPolicyRunner(
    env=your_env,           # 继承VecEnv的环境
    train_cfg=train_cfg,
    log_dir="/path/to/logs",
    device="cuda"
)

# 4. 训练
runner.learn(
    num_learning_iterations=1000,
    init_at_random_ep_len=False
)

# 5. 推理
obs = env.get_observations()
with torch.no_grad():
    actions = runner.alg.actor_critic.act_inference(obs)
```

---

## 🎓 学习建议

### 推荐学习顺序

1. **理解PPO算法基础**
   - 阅读原始PPO论文
   - 理解信任域和剪裁
   - 学习GAE的概念

2. **研究代码层次结构**
   - 从 `ppo.py` 开始
   - 理解 `actor_critic.py` 的网络设计
   - 学习 `rollout_storage.py` 的数据管理

3. **追踪数据流**
   - 在 `OnPolicyRunner.learn()` 中设置断点
   - 观察形状变化
   - 理解批处理逻辑

4. **实验修改**
   - 改变网络架构
   - 调整超参数
   - 添加自定义奖励塑形

### 关键概念

- **同策略 vs 异策略**：PPO是同策略，意味着必须使用刚收集的数据
- **优势函数**：相对于基线的回报，减少方差
- **政策梯度**：直接优化策略，而不是学习Q函数
- **信任域**：PPO限制单步中策略的变化

---

## 📚 重要文件速查表

| 文件 | 关键类/函数 | 学习重点 |
|------|-----------|---------|
| `algorithms/ppo.py` | `PPO.update()` | PPO损失函数和梯度步骤 |
| `modules/actor_critic.py` | `ActorCritic.act()` | 策略采样和值函数计算 |
| `storage/rollout_storage.py` | GAE计算 | 优势估计算法 |
| `runners/on_policy_runner.py` | `OnPolicyRunner.learn()` | 完整训练循环 |

---

## 🔗 相关资源

- **PPO原始论文**：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **相关项目**：[legged_gym](https://github.com/leggedrobotics/legged_gym)
- **项目网站**：https://leggedrobotics.github.io/legged_gym/

---

## 💡 常见问题解答

**Q: 为什么有两个不同的Actor-Critic实现？**
A: `ActorCritic` 用于马尔可夫环境，`ActorCriticRecurrent` 用于需要内存的部分可观察环境（如受限视野）。

**Q: 什么是特权观测？**
A: 仅在训练期间可用的信息（如隐藏参数），用于Critic但不用于推理策略。

**Q: 如何使用这个库进行自定义环境？**
A: 创建一个继承 `VecEnv` 的环境类并实现抽象方法，然后传递给 `OnPolicyRunner`。

**Q: 我应该使用多少个环境？**
A: 更多的环境=更好的梯度估计但更多的内存。从8-16开始，根据内存增加。

---

*本指南基于RSL RL v1.0.2 - 最后更新：2025年11月*
