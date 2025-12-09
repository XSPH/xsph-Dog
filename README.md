# xsph-Dog 项目

## 项目概述

xsph-Dog 是一个基于强化学习的四足机器人运动控制项目，结合了 `legged_gym` 和 `rsl_rl` 两个核心组件。该项目专注于使用 NVIDIA Isaac Gym 物理引擎训练四足机器人（如 ANYmal、A1 等）在不平坦地形上的行走能力。

**主要特点:**
- 使用 NVIDIA Isaac Gym 物理仿真环境
- 实现了 PPO (Proximal Policy Optimization) 强化学习算法
- 支持多种四足机器人模型：ANYmal、A1、Cassie
- 包含从仿真到实际应用的完整训练流程
- 支持地形课程学习和域随机化技术

**项目维护者:** Nikita Rudin  
**所属机构:** 苏黎世联邦理工学院机器人系统实验室 & NVIDIA  
**许可证:** BSD-3-Clause

## 项目结构

```
xsph-Dog-main/
├── legged_gym/          # 机器人环境和训练脚本
│   ├── legged_gym/
│   │   ├── envs/        # 各种机器人环境定义
│   │   │   ├── a1/      # A1 机器人配置
│   │   │   ├── anymal_b/ # ANYmal B 机器人配置
│   │   │   ├── anymal_c/ # ANYmal C 机器人配置
│   │   │   ├── base/    # 基础环境类
│   │   │   └── cassie/  # Cassie 机器人配置
│   │   ├── scripts/     # 训练和测试脚本
│   │   │   ├── train.py # 训练脚本
│   │   │   └── play.py  # 测试脚本
│   │   └── utils/       # 工具函数
│   └── resources/       # 机器人模型资源
└── rsl_rl/              # 强化学习算法实现
    ├── rsl_rl/
    │   ├── algorithms/  # 强化学习算法
    │   │   └── ppo.py   # PPO 算法实现
    │   ├── modules/     # 神经网络模块
    │   ├── runners/     # 训练运行器
    │   ├── storage/     # 数据存储
    │   └── env/         # 环境接口
```

## 技术栈

- **编程语言:** Python 3.6+
- **深度学习框架:** PyTorch (>=1.4.0)
- **物理仿真:** NVIDIA Isaac Gym Preview 3
- **核心依赖:**
  - torch>=1.4.0
  - torchvision>=0.5.0
  - numpy>=1.16.4
  - isaacgym
  - rsl-rl
  - matplotlib

## 安装和设置

### 环境要求
- Python 3.6, 3.7 或 3.8 (推荐 3.8)
- CUDA 11.3 (用于 GPU 加速)
- NVIDIA Isaac Gym Preview 3

### 安装步骤
1. 创建 Python 虚拟环境
2. 安装 PyTorch 1.10 with CUDA-11.3:
   ```bash
   pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
3. 安装 Isaac Gym
4. 安装 rsl_rl:
   ```bash
   cd rsl_rl && pip install -e .
   ```
5. 安装 legged_gym:
   ```bash
   cd legged_gym && pip install -e .
   ```

## 核心组件

### 1. 机器人环境 (legged_gym)
- **基础环境类:** `LeggedRobot` - 实现了不平坦地形上的运动任务
- **机器人特定环境:** 
  - `Anymal` - ANYmal 机器人环境
  - `Cassie` - Cassie 机器人环境
- **配置系统:** 每个环境都有对应的配置文件，包含环境参数和训练参数

### 2. 强化学习算法 (rsl_rl)
- **PPO 算法:** 目前唯一实现的算法，专为 GPU 加速设计
- **神经网络模块:** 
  - `ActorCritic` - 标准 Actor-Critic 架构
  - `ActorCriticRecurrent` - 支持循环网络的 Actor-Critic 架构
- **训练运行器:** `OnPolicyRunner` - 协调整个训练流程

### 3. 训练和测试脚本
- **训练脚本:** `train.py` - 用于训练强化学习策略
- **测试脚本:** `play.py` - 用于测试训练好的策略

## 常用命令

### 训练策略
```bash
python legged_gym/scripts/train.py --task=anymal_c_flat
```

### 测试策略
```bash
python legged_gym/scripts/play.py --task=anymal_c_flat
```

### 命令行参数
- `--task`: 任务名称
- `--resume`: 从检查点恢复训练
- `--experiment_name`: 实验名称
- `--run_name`: 运行名称
- `--headless`: 无头模式运行（无渲染）
- `--sim_device=cpu`: 在 CPU 上运行仿真
- `--rl_device=cpu`: 在 CPU 上运行强化学习

## 开发约定

### 代码风格
- 所有文件使用 SPDX 许可证头
- 遵循 Python PEP 8 编码规范
- 使用类型提示 (Type Hints)

### 项目架构模式
- 模块化设计：环境、算法、网络、运行器分离
- 配置驱动：通过配置文件控制环境和训练参数
- 继承机制：环境和配置类使用继承实现代码复用
- GPU 优先：所有计算设计为在 GPU 上高效运行

### 环境注册
新环境必须通过 `task_registry.register()` 注册，格式为：
```python
task_registry.register("环境名称", 环境类, 环境配置, 训练配置)
```

## 日志和模型保存

训练过程中，模型和日志保存在以下目录结构：
```
logs/<experiment_name>/<date_time>_<run_name>/
├── events.out.tfevents.*  # TensorBoard 日志
├── model_0.pt            # 初始模型
├── model_500.pt          # 第500次迭代的模型
└── model_1000.pt         # 第1000次迭代的模型
```

## 扩展和开发

### 添加新环境
1. 在 `legged_gym/envs/` 下创建新文件夹
2. 创建配置文件，继承现有配置
3. 如需要，实现环境类，继承现有环境类
4. 在 `envs/__init__.py` 中注册新环境

### 添加新机器人
1. 将机器人模型文件添加到 `resources/robots/`
2. 在配置文件中设置资产路径、身体名称、关节位置和 PD 增益
3. 指定训练配置和环境名称

## 注意事项

- 该项目需要 NVIDIA Isaac Gym，仅支持 NVIDIA GPU
- 训练时建议按 'v' 键停止渲染以提高性能
- 项目已迁移到 Isaac Lab 框架，此仓库将有限更新
- 联系方式：xsph2005@gmail.com

## 相关资源

- **项目网站:** https://leggedrobotics.github.io/legged_gym/
- **相关论文:** https://arxiv.org/abs/2109.11978
- **Isaac Lab 迁移信息:** https://github.com/isaac-sim/IsaacLab