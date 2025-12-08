# PPO 实现详解（针对 `rsl_rl/algorithms/ppo.py`）

本文档面向想逐行阅读并理解仓库中 `rsl_rl/algorithms/ppo.py` 实现的读者。

- 目标：解释文件中每一部分的功能、数学意义、张量形状、与其他模块的接口，并给出调试与小规模测试建议。
- 假设：你已熟悉 PyTorch 基本用法和PPO的基本概念（策略损失、价值损失、熵、GAE）。

---

## 一、文件定位与总体职责

文件：`rsl_rl/algorithms/ppo.py`

职责：实现近端策略优化（PPO）算法的训练循环与更新逻辑。负责：

- 接受 `ActorCritic` 网络，和 `RolloutStorage`（经验缓存）协同工作；
- 在采集阶段填充 `RolloutStorage`；
- 计算返回与优势；
- 对策略进行小批量、多轮次更新（包括可选的自适应学习率）；
- 支持循环（RNN）和非循环策略两种模式。

与其它模块的接口：

- 输入/依赖：`ActorCritic`（或 `ActorCriticRecurrent`）、`RolloutStorage`、PyTorch optimizer。
- 输出：每次 `update()` 返回平均值损失与平均代理损失（surrogate loss）。

---

## 二、先看类“契约”（inputs / outputs / side effects）

PPO 类（构造函数参数摘录）

Inputs（高层）：
- `actor_critic`: 模型实例（实现 `.act()`, `.evaluate()`, `.get_actions_log_prob()` 等）
- 超参数：`clip_param`, `gamma`, `lam`, `learning_rate`, `num_mini_batches`, `num_learning_epochs`, `value_loss_coef`, `entropy_coef`, `max_grad_norm`, `use_clipped_value_loss`, `schedule`, `desired_kl`

重要方法与输出：
- `init_storage(...)`：建立 `RolloutStorage` 实例（无返回）。
- `act(obs, critic_obs)`：基于当前策略产生动作，并把一个临时 `Transition` 填充好（返回动作）。
- `process_env_step(rewards, dones, infos)`：将上一步的 `Transition` 写入 `RolloutStorage` 并清空临时结构。
- `compute_returns(last_critic_obs)`：使用 `actor_critic` 对最后一个观测估值，调用 `storage.compute_returns()` 计算 GAE/returns（无返回）。
- `update()`：执行 PPO 更新返回 `(mean_value_loss, mean_surrogate_loss)`。

Side-effects（副作用）：
- 修改 `self.actor_critic` 的参数（优化步骤）。
- 修改 `self.storage` 内容（在训练轮后清空）。
- 当 `schedule == 'adaptive'` 时可能会调整 `self.learning_rate` 并修改 optimizer lr。

---

## 三、关键实现解析（按代码片段）

下面按逻辑分块讲解 `ppo.py` 的实现要点，并给出张量形状与数学公式。

### 1) 初始化（__init__）

要点：
- 将 `actor_critic` 移动到 `device`，创建 Adam optimizer。
- 保存超参数：`clip_param`, `gamma`, `lam`, `value_loss_coef`, `entropy_coef` 等。
- 创建 `self.transition = RolloutStorage.Transition()` 作为临时容器，用于每步采集。

为什么要用临时 `Transition`？
- 在多环境并行时，每个 step 会同时为所有环境生成一组数据；用 `Transition` 存放这些并一次写入 `RolloutStorage`，方便批量化处理。

### 2) act(obs, critic_obs)

源码要点：

```py
if self.actor_critic.is_recurrent:
    self.transition.hidden_states = self.actor_critic.get_hidden_states()
self.transition.actions = self.actor_critic.act(obs).detach()
self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
self.transition.action_mean = self.actor_critic.action_mean.detach()
self.transition.action_sigma = self.actor_critic.action_std.detach()
self.transition.observations = obs
self.transition.critic_observations = critic_obs
return self.transition.actions
```

说明与形状：
- 假设 `obs` 的形状为 `[num_envs, obs_dim]`。
- `.act(obs)` 返回 `[num_envs, action_dim]`。
- `.evaluate(critic_obs)` 返回 `[num_envs, 1]`（值函数标量）。
- `.get_actions_log_prob(actions)` 返回 `[num_envs]` 或 `[num_envs, 1]`（代码随后使用 view/ squeeze）。

注意 `.detach()`：防止将当前采集步骤的计算图与后续梯度计算混在一起，采集阶段不应保留计算图。

### 3) process_env_step(rewards, dones, infos)

要点：
- `rewards`、`dones` 从环境返回，可能是 `[num_envs]` 或 `[num_envs,1]`。
- 对时间结束（timeout）环境的 bootstrapping：如果 `infos` 中包含 `time_outs`，则把 `values` 按照 gamma 加入 reward。这是为了在环境因时间限制结束但未终止的情况下进行合理的引导。
- 调用 `self.storage.add_transitions(self.transition)`，然后 `transition.clear()`，并重置 actor_critic 内部状态 `self.actor_critic.reset(dones)`（RNN 情况下清理对应 env 的隐藏状态）。

为什么要处理 `time_outs`？
- 当环境到达最大步长时，不应把该步的 reward 当作终止回报；而是用 bootstrapping（使用价值函数估计下一个状态）来替代真实的下一个状态价值缺失。

### 4) compute_returns(last_critic_obs)

要点：
- 计算最后时刻的价值 `last_values = self.actor_critic.evaluate(last_critic_obs).detach()`，并传入 `self.storage.compute_returns(last_values, gamma, lam)`。
- `RolloutStorage.compute_returns()` 实现 GAE（参见 `rollout_storage.py`），输出 `returns` 和 `advantages`。

数学上，GAE:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$A_t = \sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}$$

返回（returns）通常为：

$$R_t = A_t + V(s_t)$$

### 5) update() （核心）

这是PPO实现的关键函数，分为若干步骤：

1. 生成小批量（recurrent 或 非recurrent）
2. 对每个小批量计算：新策略的 log_prob、value、mu、sigma、entropy
3. 计算 KL（如果使用自适应 schedule）并更新 lr
4. 计算 surrogate loss（带剪切）
5. 计算 value loss（可选剪切）
6. 总损失 = surrogate + value_loss_coef * value_loss - entropy_coef * entropy
7. 反向传播、梯度裁剪、优化器更新

细节解析：

- 小批量生成：
  - 如果 `actor_critic.is_recurrent` 为 True，使用 `storage.reccurent_mini_batch_generator()`（该 generator 会返回 `hid_states_batch` 和 `masks_batch`），否则使用 `storage.mini_batch_generator()`。

- 新旧概率比：
  - 代码计算 `ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))`
  - 这是 PPO 里常用的概率比（new / old），在连续动作高斯情况下通过对数概率差实现。

- 代理损失（surrogate loss）：

$$L^{	ext{sur}} = -E_t\Big[\min\big(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big]$$

在代码中，先计算未剪切的 `surrogate` 和 `surrogate_clipped`，然后取 `torch.max(surrogate, surrogate_clipped).mean()`。注意：实现使用的是负号在前（因为它把 `surrogate` 定义为 `-A*ratio`），因此 `mean()` 后即为要最小化的损失。

- 值函数损失（clipped value loss，可选）：

标准MSE: (returns - value)^2

当 `use_clipped_value_loss` 为 True 时，使用和论文类似的剪切机制，避免 value 函数出现过大跳变。

实现做法：
- 先计算 `value_clipped = target_values + (value - target_values).clamp(-clip, clip)`。
- 然后比较裁切前后的 value-loss，取较大的一个（对 value loss 使用 `torch.max`），并取平均。

- 熵项：用于鼓励探索。代码用 `entropy_batch.mean()`。

- 总损失：

```
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
```

- 梯度步骤：
  - `loss.backward()` 后做 `nn.utils.clip_grad_norm_(...)`，然后 `self.optimizer.step()`。

- 统计：累计 mean_value_loss 和 mean_surrogate_loss，最终按 `num_updates = num_learning_epochs * num_mini_batches` 归一化并返回。

### 6) 自适应 KL 调整学习率

如果 `schedule == 'adaptive'` 且 `desired_kl` 非空，代码在 update 时计算近似的 KL 散度（仅在高斯策略下用闭式式子近似）：

计算公式（近似）：

$$KL(N(\mu_{old}, \sigma_{old}) || N(\mu, \sigma)) = \sum \log(\sigma/\sigma_{old}) + \frac{\sigma_{old}^2 + (\mu_{old}-\mu)^2}{2\sigma^2} - 0.5$$

实现中对该KL求均值 `kl_mean`，然后根据 `desired_kl` 调整 learning_rate：
- 如果 `kl_mean > desired_kl * 2`，减小 lr（除以1.5，下限1e-5）
- 如果 `kl_mean < desired_kl / 2` 且 >0， 增加 lr（乘以1.5，上限1e-2）

最后把新的 lr 应用到 optimizer 的 param groups。

---

## 四、典型张量形状追踪（非循环）

假设：`num_envs = E`, `num_transitions_per_env = T`, `action_dim = A`, `obs_dim = O`。

- 在采集阶段（每 step）
  - `obs`: `[E, O]`
  - `actions` (采样): `[E, A]`
  - `values`: `[E, 1]`
  - `actions_log_prob`: `[E]` 或 `[E,1]`

- 在更新阶段（小批量）
  - `obs_batch`: `[batch_size, O]`（`batch_size` 来自 `RolloutStorage.mini_batch_generator()`）
  - `actions_batch`: `[batch_size, A]`
  - `advantages_batch`: `[batch_size, 1]`
  - `old_actions_log_prob_batch`: `[batch_size, 1]`

注意： `RolloutStorage` 中数据最初存储为 `[T, E, ...]`，generator 会把它展开成 `[batch_size, ...]`。

---

## 五、与仓库其他模块的关系

- `ActorCritic` / `ActorCriticRecurrent`：
  - `ppo.py` 假设网络支持 `.act()`, `.evaluate()`, `.get_actions_log_prob()`, `.action_mean`, `.action_std`, `.entropy` 等属性/方法。

- `RolloutStorage`：
  - `ppo.init_storage(...)` 初始化；采集阶段向 `RolloutStorage` 写入数据；更新阶段从 `RolloutStorage` 读取小批量。

- `OnPolicyRunner`：
  - 协调 `act()`、`process_env_step()`、`compute_returns()`、`update()` 的调用顺序，形成完整训练循环。

---

## 六、调试建议与常见陷阱

1. 形状不匹配（最常见）
   - 在 `actor_critic` 的方法中打印 `tensor.shape`，例如 `actions.shape`/`values.shape`，确保 `RolloutStorage.add_transitions()` 正确 copy。

2. log_prob 维度
   - 注意 `.log_prob()` 可能返回 `[batch, action_dim]`，代码里使用 `.sum(dim=-1)` 在 `ActorCritic.get_actions_log_prob()` 中合并维度，确认返回形状为 `[batch]` 或 `[batch,1]`。

3. RNN 隐藏状态
   - 对 `ActorCriticRecurrent`，确保 `hidden_states` 的形状和 `Memory.reset()` 的处理方式一致。`dones` 需要被正确广播到隐藏状态维度进行清零。

4. KL 自适应调度
   - 如果使用 adaptive schedule 要观察 `kl_mean` 的值，避免 lr 在训练早期剧烈波动。设置合适的 `desired_kl`。

5. entropy_coef
   - 熵系数过高会导致策略噪声过大；过低会过早收敛到确定策略。逐步调参。

6. use_clipped_value_loss
   - 有时剪切值函数有助于稳定训练，但在某些任务上会导致缓慢收敛，尝试切换开关。

---

## 七、快速运行与小规模单元测试建议

下面给出一个短脚本，用随机输入模拟一个完整的 `act -> process_env_step -> compute_returns -> update` 循环（仅用于功能测试，不用于学习）：

```python
# 保存为 ppo_smoke_test.py 并在仓库根目录运行
import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms.ppo import PPO

# 简单参数
num_envs = 4
num_steps = 8
obs_dim = 10
priv_obs_dim = 10
action_dim = 3

def make_dummy_env_step(actions):
    # 返回 obs, reward, done, infos
    obs = torch.randn(num_envs, obs_dim)
    rewards = torch.randn(num_envs)
    dones = torch.zeros(num_envs, dtype=torch.uint8)
    infos = {}
    return obs, rewards, dones, infos

# 创建网络 & 算法
ac = ActorCritic(num_actor_obs=obs_dim, num_critic_obs=priv_obs_dim, num_actions=action_dim)
ppo = PPO(ac, num_learning_epochs=1, num_mini_batches=1, device='cpu')
ppo.init_storage(num_envs, num_steps, [obs_dim], [priv_obs_dim], [action_dim])

# 模拟采集
obs = torch.randn(num_envs, obs_dim)
critic_obs = obs.clone()
for t in range(num_steps):
    actions = ppo.act(obs, critic_obs)
    obs, rewards, dones, infos = make_dummy_env_step(actions)
    ppo.process_env_step(rewards, dones, infos)

# compute returns using last critic obs
ppo.compute_returns(obs)

# update
v_loss, s_loss = ppo.update()
print('value loss', v_loss, 'surrogate loss', s_loss)
```

运行：

```bash
python3 ppo_smoke_test.py
```

如果脚本运行无异常并打印损失，即表明函数调用链基本通顺。

---

## 八、阅读建议（按优先级）

1. `rsl_rl/modules/actor_critic.py`（理解 policy 与 distribution 接口）
2. `rsl_rl/storage/rollout_storage.py`（理解 generator 的输出）
3. `rsl_rl/algorithms/ppo.py`（本文件，配合上面两个文件一行行看）
4. `rsl_rl/runners/on_policy_runner.py`（理解调度）

---

## 九、结语

这份文档旨在把 `rsl_rl` 的 PPO 实现与标准PPO算法的数学形式对应起来，指出实现细节与工程化权衡。

---

## 附：非对称网络 `ActorCriticAsymmetric`

**概念**：Actor 只看到主观测（如IMU、关节状态），Critic 看到主观测 + 线速度（作为特权观测）。这样在训练时 Critic 有更多信息做价值估计（更准确的baseline），但推理时 Actor 不依赖线速度（可部署）。

**结构**：
- Actor：`num_actor_obs` → 隐藏层 → `num_actions`
- Critic：`[num_actor_obs, velocity_dim]` → 隐藏层 → 1

**关键方法**：
- `extract_velocity_from_privileged_obs(privileged_obs)`：从特权观测中提取前 `velocity_dim` 维（线速度）
- `evaluate(critic_observations)`：将特权观测分解为速度 + agent 特征，送入 Critic 网络
- `act(observations)`：只用 agent 观测采样动作

**使用示例**：

```python
# 配置
train_cfg = {
    "runner": {
        "policy_class_name": "ActorCriticAsymmetric",
        "num_steps_per_env": 24,
    },
    "policy": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "velocity_dim": 3,  # or 2 for 2D velocity
    }
}

# 在 legged_gym 中
# env.num_obs = 48 (agent obs without velocity)
# env.num_privileged_obs = 51 (48 agent obs + 3 velocity)

runner = OnPolicyRunner(env, train_cfg, device='cuda')

# 训练循环中
obs = env.get_observations()  # shape [num_envs, 48]
privileged_obs = env.get_privileged_observations()  # shape [num_envs, 51]

# PPO.act() 接收 obs 和 critic_obs(privileged_obs)
actions = runner.alg.act(obs, privileged_obs)

# 推理时只需要 agent obs，不需要速度
actions_infer = runner.alg.actor_critic.act_inference(obs)
```

**优势**：
- 训练时 Critic 有全局信息，梯度更准确
- 推理时 Actor 自主决策，可在真实机器人上部署（不需要外部速度估计）
- 天然支持特权学习（privileged information）
- 或者把上面的 smoke-test 脚本加入到仓库并自动运行一次以验证？

告诉我你想要的下一步。