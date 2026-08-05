# Quant-Ultra (Quant-4) 工业级量化引擎下一阶段全面优化计划书

## —— 以 LLM 神经-符号闭环为主、每日盘后 Cython 计算重构为辅的全栈升级方案

---

## 一、 项目背景与架构定轨 (Background & System Scope)

### 1.1 系统定位与瓶颈诊断

Quant-Ultra（Quant-4）是一套涵盖数据基建、PIT 时点特征、机器学习概率校准、Black-Litterman 凸优化仓位求解、FSM 物理撮合回测、DSR 风控审计至 CIO 治理的双语可审计量化系统。在经过 8-4/8-5 分支的物理真实性重构后，系统已具备 PIT 防穿越、Purged K-Fold、Fail-Closed 硬杀伤及四元组运行指纹等安全红线。

在确定下一阶段演进方向时，**彻底排除高频/分钟级实盘撮合场景**，将系统定位锁定于：**“日频（Daily）非平稳环境下的自适应认知与大容量盘后决策流水线”**。在此定位下，系统面临两大关键瓶颈：

1. **认知与归因僵化（认知层痛点）**：传统 ML 模型仅能输出预测分值，面对宏观 Regime Switch 或突发黑天鹅时缺乏归因能力；现有 Phase 10 的大模型决策若缺乏刚性符号约束，易产生“交易幻觉”与未经校验的参数覆盖风险。
2. **多年全市场 Walk-Forward 盘后计算耗时（计算层痛点）**：在每日盘后拉起全市场（数千只标的）、数十年 Walk-Forward 矩阵重训、Monte Carlo 压力测试及 FSM 物理逐日撮合时，纯 Python 循环与 GIL 锁限制了批量算力吞吐。

---

### 1.2 主辅设计哲学 (Core Architecture Strategy)

```
                       ┌──────────────────────────────────────────┐
                       │   认知层 (Cognitive / Neural Layer)       │
                       │   - LLM 市场体制感知 & 归因分析           │
                       │   - 待审参数提案 (Parameter Proposals)    │
                       └────────────────────┬─────────────────────┘
                                            │ 白名单&范围约束
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │   风控与执行层 (Symbolic Safety Layer)    │
                       │   - Black-Litterman 凸优化 & 物理 FSM 撮合│
                       │   - Fail-Closed 熔断 & OBSERVATION_ONLY  │
                       └────────────────────┬─────────────────────┘
                                            │ 每日盘后矩阵批量计算
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │  每日盘后计算底座 (Daily HPC / Cython)    │
                       │  - 无高频/无分钟级，专注日频全市场批量加速 │
                       │  - 零拷贝内存共享 & NumPy 零差异回退     │
                       └──────────────────────────────────────────┘

```

* **主方向（灵魂）：LLM 神经-符号闭环（Neuro-Symbolic Governance Loop）**
* **神经元（Neural）**：LLM 智能体负责非结构化情绪提取、市场 Regime 识别、盘后归因及参数调整提案（Proposals）。
* **符号逻辑（Symbolic）**：凸优化求解器、FSM 物理红线与安全白名单构成硬性物理屏障，任何审计不通过强制降级为 `OBSERVATION_ONLY`。


* **辅方向（骨骼）：每日盘后 Cython 低层重构（Daily Post-Market HPC Refactoring）**
* **明确边界**：**不执行任何高频、Tick 级或分钟级实时操作**，仅聚焦每日收盘后的全市场日线批量计算、高维张量卷积与多折 Walk-Forward 回测提速。



---

## 二、 核心方向一：LLM 神经-符号闭环重构方案 (Primary Focus)

### 2.1 神经元组件：感官、归因与提案生成 (Neural Component)

1. **多源非结构化信息感官**：
* 在 Phase 3 中利用本地 Ollama/词典混合抽取新闻与论坛情绪，通过时间戳过滤杜绝未来信息。


2. **盘后业绩与漂移归因**：
* 每日收盘后，整合 Phase 7 真实净值、Phase 8 DSR 审计摘要 与 Phase 9 MAE/PSI 漂移指标，注入 LLM Prompt 上下文，生成结构化的策略状态诊断与归因日志。


3. **白名单约束下的参数提案（Parameter Proposals）**：
* LLM 不具备直接改写生产配置文件 (`default_param.yaml`) 的权限。其输出必须为 JSON 格式的待审提案文件（`parameter_proposal.json`），仅允许对风险偏好、超参搜索网格边界及先验置信度进行调整。



### 2.2 符号与物理红线层 (Symbolic Safety Layer)

1. **提案白名单与幅度校验器（Proposal Validator）**：
* 设立硬性参数白名单与单次调整幅度上限（如单标的上限只允许在 5%~10% 内微调，先验置信度变动幅不超过 ±15%）。超越边界的提案将被系统自动拦截并抛出告警。


2. **Black-Litterman 视角映射与凸优化硬约束**：
* 将经过校验的观点注入 Black-Litterman 矩阵，并经由 `convex_optimizer.py` 在现金缓冲（≥5%）、集中度上限（≤8%）、流动性容量（ADV20）及换手成本硬约束下求解最优目标仓位。


3.  Fail-Closed 熔断与 `OBSERVATION_ONLY` 降级：
* 当 Phase 8 压力测试抛出红线、Phase 9 影子对账 MAE 超过安全阈值或行情数据刷新不足时，系统触发物理硬熔断。
* Phase 11 交互式投顾自动强制降级为仅观察模式（`OBSERVATION_ONLY`），锁定所有买入信号。



### 2.3 闭环自愈交互链路 (Closed-Loop Workflow)

```text
[每日盘后收盘] 
   └──> 拉起 Phase 1~9 运行（生成真实净值、DSR、PSI 漂移与影子对账 MAE）
          ├──> 审计/对账通过 ──> Phase 10 (LLM 归因与参数提案) ──> 校验白名单 ──> 生成可执行建议 (Phase 11)
          └──> 审计/对账失败 ──> 触发物理硬杀伤 (Fail-Closed) ──> Phase 11 强制降级为 OBSERVATION_ONLY

```

---

## 三、 核心方向二：每日盘后计算层 Cython 加速方案 (Secondary Focus)

### 3.1 明确边界与设计原则

* **严禁过度设计**：彻底放弃 Tick/分钟级高频重构。所有 Cython 模块仅服务于每日盘后批量处理（Daily Post-Market Batch Processing）。
* **零拷贝与回退机制**：使用 Cython 内存视图（Memoryviews）实现 NumPy 数组零拷贝，同时必须提供数值完全等价的纯 Python/NumPy 回退路径（Fallback），确保在无 C 编译器环境下系统平滑降级运行。

### 3.2 每日盘后重构的三个核心模块

#### 模块 A：Phase 3 / Phase 6 全市场高维日线张量卷积

* **痛点**：每日盘后需要对全市场数千只标的近数千个交易日计算分数阶微分（FFT/滑动窗口）、Garman-Klass 波动率及协方差收缩矩阵。
* **重构方案**：在 `cython/quant_ultra_fast.pyx` 中实现 C 阶连续矩阵的并行滑动窗口计算，避免 Python 嵌套循环与频繁的内存分配。

#### 模块 B：Phase 7 FSM 逐日撮合引擎与物理摩擦模拟

* **痛点**：全市场多年 Walk-Forward 回测时，逐日更新整手截断（100股）、非线性平方根冲击滑点、停牌/涨跌停状态及退市残值计提在 Python 循环中较为繁琐。
* **重构方案**：将逐日 FSM 状态转移方程（持仓状态、现金账本、印花税/佣金扣减）下沉至 Cython 结构体运算，显著提升多组合历史回测吞吐量。

#### 模块 C：Phase 8 盘后 Monte Carlo 黑天鹅压力测试

* **痛点**：Phase 8 在盘后审计时需要针对历史极值（如 2015 股灾、2020 流动性危机）进行数万次 Monte Carlo 路径抽样与 Christoffersen 条件覆盖检验。
* **重构方案**：在 Cython 中释放 GIL（`with nogil:`），利用多线程并行生成压力测试路径与降阶夏普比率（DSR）的统计显著性分布。

---

## 四、 分阶段技术实施与代码映射表 (Phase Mapping)

| 流水线阶段 | 核心优化目标 | 主方向（LLM 神经-符号）映射点 | 辅方向（每日盘后 Cython）映射点 |
| --- | --- | --- | --- |
| **Phase 1~2**<br>

<br>(Data & Slices) | PIT 时间线保障与切片隔离 | 记录数据来源 SHA-256 与版本 Manifest，供 LLM 校验数据新鲜度 | 向量化构建标的存续矩阵 (`alive_mask`)，加速日历自愈 |
| **Phase 3**<br>

<br>(PIT Features) | 高维特征与另类情绪注入 | LLM 提取新闻/论坛情绪，受超时（20s）与字数硬限额保护 | Cython 加速高维面板张量 $(T \times N \times F)$ 的滑动窗口计算 |
| **Phase 4~5**<br>

<br>(Labels & Model) | 三屏障标签与 DSR 证据透传 | 透传 `num_trials` 尝试总数至 Context，为 LLM 提供准确过拟合证据 | 优化 CQR 概率校准与 Quantile 分位数日频拟合计算速度 |
| **Phase 6**<br>

<br>(Position Sizing) | 凸优化与 BL 视角融合 | 将 LLM 审核通过的提案注入 BL 先验与集中度硬上限 | 下沉 Shrinkage 协方差估计与二次规划矩阵构建至 C++ 层 |
| **Phase 7**<br>

<br>(FSM Backtest) | 高保真物理撮合与账本守恒 | 输出准确的费用分类账与净值，作为 LLM 归因的真实物理依据 | Cython 优化逐日 FSM 状态机、整手截断与平方根冲击滑点计算 |
| **Phase 8**<br>

<br>(Audit & Stress) | 一票否决硬审计与 DSR 惩罚 | 审计失败抛出标志位，阻断后续决策生成 | `nogil` 多线程并行计算 Monte Carlo 极值压测与 DSR 显著性 |
| **Phase 9**<br>

<br>(Live MLOps) | 影子对账与概念漂移监控 | 计算 MAE 与 PSI/KS 漂移，注入 Phase 10 LLM 上下文 | 增量特征矩阵漂移指标的高效向量化算子 |
| **Phase 10**<br>

<br>(CIO Governance) | 双语治理归因与白名单提案 | **核心落地点**：生成可解释归因报告与待审参数提案文件 | 无（纯文本/Prompt 交互） |
| **Phase 11**<br>

<br>(Interactive) | 四阶段决策输出与降级网 | **核心落地点**：根据风控结果输出四阶段建议，失败时强制降级为 `OBSERVATION_ONLY` | 无（纯交互与格式化输出） |

---

## 五、 质量保障与测试验证体系 (Quality Assurance)

### 5.1 未来扰动不变性测试 (Future Perturbation Invariance)

* **方法**：在盘后历史数据末端（$T+1$ 至 $T+N$）人工注入极端波动噪声。
* **标准**：运行流水线后，$T$ 日及之前的特征面板、目标仓位与 LLM 归因结果必须保持 **0 差异**，绝对证明系统无未来函数。

### 5.2 神经-符号降级与白名单拒绝测试

* **参数越界拦截测试**：人工构造超出限额的提案（如将单标上限设为 50%），验证 `parameter_governance.py` 必须 100% 拒绝并记录告警日志。
* **Fail-Closed 降级测试**：人工注入 10% 的影子对账持仓偏差，验证 Phase 11 必须在 1 秒内降级为 `OBSERVATION_ONLY` 并锁定买入功能。

### 5.3 Cython 与 NumPy 双轨零差异校验

* **测试套件**：编写 `tests/test_fast_math_precision.py`，全量比对 Cython 重构后的 FSM 撮合结果、凸优化目标函数值与纯 Python/NumPy 回退路径的计算结果。
* **标准**：浮点数绝对误差必须控制在 $10^{-12}$ 以内（机器精度级别）。

---

## 六、 实施路线图与里程碑 (Roadmap & Milestones)

重构工程划分为四个迭代阶段，历时 8 周完成全量代码闭环：

| 迭代阶段 | 时间节点 | 核心任务模块 | 核心交付物与验收标准 |
| --- | --- | --- | --- |
| **P0 阶段：白名单门禁与降级锁网** | 第 1–2 周 | • 实现 `parameter_governance.py` 白名单校验器<br>

<br>• 贯通 Phase 9 影子对账抛错至 Phase 11 降级网<br>

<br>• 移除生产环境中残留的调试剪裁代码 | • 参数越界提案 100% 被拦截<br>

<br>• 对账失败时 Phase 11 自动进入 `OBSERVATION_ONLY` 模式 |
| **P1 阶段：LLM 归因与参数提案闭环** | 第 3–4 周 | • 接入 Phase 7/8/9 真实指标至 Phase 10 LLM 上下文<br>

<br>• 规范待审参数提案 JSON 输出契约<br>

<br>• 实现四阶段决策链的 Softmax 门控与非线性池化 | • 生成包含证据 SHA-256 的双语 CIO 治理报告<br>

<br>• LLM 参数提案不直接覆写配置文件，安全进盘后待审区 |
| **P2 阶段：每日盘后 Cython 计算加速** | 第 5–6 周 | • 重构 `cython/quant_ultra_fast.pyx` 中 FSM 逐日撮合算子<br>

<br>• 实现高维日线张量卷积与 Monte Carlo 压测 `nogil` 多线程<br>

<br>• 完善 NumPy 零差异纯 Python 回退路径 | • 盘后多年全市场 Walk-Forward 回测与压测速度提升 3 倍以上<br>

<br>• 通过双轨数值精度零差异检验 |
| **P3 阶段：一键自动化与端到端报告** | 第 7–8 周 | • 统一入口 `main.py` 的四元组运行指纹校验<br>

<br>• 自动编译生成自包含 `execute_report.html`<br>

<br>• 完成全量 pytest 与未来扰动测试套件 | • 执行 `python main.py --non-interactive` 一键拉起 Phase 1~11<br>

<br>• 所有测试 100% 通过，无未来函数与幽灵交易 |

---

*本计划书作为 Quant-Ultra (Quant-4) 量化交易引擎下一阶段重构工程的标准规范，所有开发与重构代码需严格遵循上述业务逻辑、风控门禁及边界限制。*
