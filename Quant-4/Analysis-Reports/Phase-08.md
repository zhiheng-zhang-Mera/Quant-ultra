# Phase_8_Audit_Stress_Testing：系统级审查与压测深度解析

## 概述

Phase_8（Audit & Stress Testing）是 Quant‑4 系统的**最终合规审查与风险验证层**，负责在策略被允许进入实盘（MLOps）之前，对其进行多维度的统计学检验与极端情景压力测试。该阶段的核心使命是：通过严格的量化审计手段，识别并拦截因多重试验导致的过拟合（Data Snooping）、回撤聚集性风险（Clustering）、历史危机脆弱性（Stress Vulnerability）以及流动性承载不足（Capacity Constraint）等问题。

Phase_8 严格遵循 Quant‑4 系统手册中“阶段八”的要求，实现了四大核心审计算子：

1. **DSR（Deflated Sharpe Ratio）**：对多重参数搜索导致的夏普比率虚高进行统计学惩罚，挤掉“水分”。
2. **Christoffersen 条件覆盖检验**：检验极端损失事件是否在时间轴上呈现聚集性（而非独立随机），防范雪崩式爆仓。
3. **历史危机压力测试**：在预设的系统性崩盘窗口（2015 股灾、2016 熔断、2024 微盘股危机）中评估策略的最大回撤。
4. **静态账户容量审计**：基于平方根冲击模型测算大额换仓的滑点损耗与市场参与率，验证策略在真实流动性环境下的可执行性。

Phase_8 采用**一票否决（One-Vote Veto）**机制，任何一项审计不通过都将阻止策略进入实盘部署，是系统风控的最后一道物理防火墙。

---

## 模块总览与协作关系

| 文件 | 核心职责 | 依赖关系 |
|------|----------|----------|
| `__init__.py` | 暴露 `execute` 入口 | 被 `main_engine.py` 调用 |
| `config.py` | 定义 DSR、Christoffersen、压力场景、容量审计的各项阈值 | 被所有子模块引用 |
| `utils.py` | 提供上下文安全提取（影子副本）和最大回撤计算工具 | 被 `dsr_audit`、`coverage_test`、`stress_test`、`capacity_audit` 引用 |
| `dsr_audit.py` | 实现递减夏普比率（DSR）统计检验，惩罚多重试验过拟合 | 依赖 `utils.safe_get_shadow` 和 `scipy.stats.norm` |
| `coverage_test.py` | 实现 Christoffersen 条件覆盖似然比检验，含波动率体制划分 | 依赖 `utils.safe_get_shadow` 和 `scipy.stats.chi2` |
| `stress_test.py` | 在预设历史危机窗口上计算最大回撤 | 依赖 `utils.safe_get_shadow` 和 `utils.compute_max_drawdown` |
| `capacity_audit.py` | 基于平方根冲击模型测算账户容量与滑点损耗 | 依赖 `utils.safe_get_shadow` |
| `step8_audit_stress_test.py` | 主控编排器，串联四项审计，实施一票否决红绿灯 | 聚合所有子模块 |

---

## 各模块详解

### 1. `config.py` —— 审计阈值与场景注册表

**功能**  
定义 Phase_8 运行所需的全量阈值与预设场景：
- **Christoffersen 检验**：无条件覆盖率保底线 `min_coverage: 0.935`（即 95% VaR 的实际覆盖不得低于 93.5%），独立性检验 p 值门槛 `christoffersen_pval_threshold: 0.01`。
- **DSR 审计**：目标夏普阈值 `sharpe_threshold: 0.50`，DSR p 值门槛 `dsr_pval_threshold: 0.05`，最小样本数 `min_samples_for_dsr: 20`。
- **压力场景**：三个预设危机窗口——`2015_liq`（2015-06 至 2015-09）、`2016_meltdown`（2016-01 至 2016-02）、`2024_microcap`（2024-01 至 2024-02）。
- **容量审计**：账户本金 `total_equity: 10,000,000`（1000 万），最大参与率红线 `max_participation_threshold: 0.05`（5%）。

**设计亮点**  
- 阈值设定参照了学术界和业界的通行标准（如 DSR 的 0.05 p 值门槛、Christoffersen 的 1% 显著性水平）。
- 压力场景的 `2024_microcap` 反映了系统对最新市场风险的跟踪意识。

**潜在缺陷**  
- 所有阈值均为静态预设，未支持从外部配置动态覆写，在策略应用于不同市场或资金量级时缺乏灵活性。
- 压力场景仅涵盖 A 股特定历史时期，对美股或全球宏观冲击（如 2020 疫情、2022 俄乌冲突）未做覆盖。

---

### 2. `utils.py` —— 审计工具箱

**功能**  
- **`safe_get_shadow`**：从 `pipeline_context` 中提取指定键的值，若为 `pd.Series` 或 `pd.DataFrame` 则返回其**深拷贝副本**，防止审计过程中的篡改污染原始数据；若键缺失则抛出 `KeyError` 硬熔断。
- **`compute_max_drawdown`**：计算净值序列的标准前向最大回撤 `(nav - peak) / peak`，并捕获数值异常（溢出、空序列），降级返回 `0.0`。

**设计哲学**  
- “影子副本”机制确保各审计算子对原始回测结果（`daily_nav`、`daily_weights` 等）只有只读权限，物理隔离了审计层与核心数据层，防止意外修改。
- 最大回撤计算中的异常捕获体现了防御性编程——即使遇到数值溢出（如 `inf` 或 `nan`），也不会中断流水线。

**优点**  
- `safe_get_shadow` 的 `default` 参数支持优雅降级，但若 `default=None` 且键缺失则主动抛异常，防止静默空值传播。
- 回撤计算的 NaN 处理严谨，确保审计结论基于有效数据。

**客观缺陷**  
- `compute_max_drawdown` 对异常仅返回 `0.0` 并记录日志，但这种降级可能掩盖真实的极端回撤，导致压力测试结果失真（低估风险）。
- 影子副本仅对 pandas 对象进行 `.copy()`，对嵌套字典或复杂对象未做递归拷贝，存在浅拷贝风险。

---

### 3. `dsr_audit.py` —— 递减夏普比率多重试验审计

**功能**  
- 从上下文提取 `daily_nav` 净值序列，计算对数日收益率。
- 计算名义年化夏普比率：`sharpe = mean(returns) / std(returns) * sqrt(252)`。
- 从上下文读取 `num_trials`（参数搜索试验次数），若未提供则默认 `100`（在 `step8_audit_stress_test.py` 的 `safe_get_shadow` 调用中硬编码）。
- 计算收益率的偏度（`skew`）和峰度（`kurt`），构建渐进方差分母：`denom_sq = 1 - skew * sharpe + (kurt - 1)/4 * sharpe^2`。
- 计算 DSR 的 t 统计量，并引入多重试验惩罚因子 `penalty_factor = sqrt(2 * log(max(2, N)))`，对标准正态累积分布进行压缩。
- 输出 DSR p 值，判定 `dsr_pass = pval >= 0.05`。

**设计哲学**  
- DSR 是 Bailey 和 López de Prado（2012）提出的经典方法，用于修正因多次参数搜索导致的夏普比率高估。其核心思想是：在 N 次独立试验中，即使所有策略的真实夏普为 0，其最大夏普也会随 N 增大而增大。DSR 通过统计惩罚将这一“运气成分”剔除。
- 偏度与峰度的引入使检验对非正态收益率分布具有鲁棒性。

**优点**  
- 代码实现忠实于学术原文，`penalty_factor` 的 `sqrt(2 * log(N))` 形式正确反映了极值理论中的 Gumbel 分布校正。
- 当 `denom_sq <= 0` 时触发防御性降级（pval=0），防止因极端尾部风险导致方差为负的数学错误。

**客观缺陷**  
- **`num_trials` 来源可疑**：`num_trials` 通过 `safe_get_shadow(context, "num_trials", 100)` 获取，但在 Phase_5 的交叉验证和超参搜索中，实际的参数组合数并未显式记录到 `pipeline_context` 中。默认值 `100` 是一个随意猜测，而非真实试验次数，这严重削弱了 DSR 的惩罚效力——若实际搜索了 1000 组参数而仅用 `N=100` 惩罚，DSR 仍会严重高估。
- 偏度峰度校正公式 `denom_sq = 1 - skew * sharpe + (kurt - 1)/4 * sharpe^2` 在 `sharpe` 为负或 `kurt` 极大时可能产生负值，虽被捕获但降级为 `pval=0` 过于激进——应使用 `max(denom_sq, 1e-6)` 而非直接判负。
- 未考虑收益率的自相关性对夏普标准误的影响（需 Newey-West 调整）。

---

### 4. `coverage_test.py` —— Christoffersen 条件覆盖检验

**功能**  
- 从上下文提取 `violations`（违规记录，即实际收益超出共形预测区间的二进制序列）和 `daily_nav`。
- 计算日对数收益率，与 `violations` 对齐时间索引。
- 将全样本按滚动 20 日波动率分为三个体制（`low_vol`、`mid_vol`、`high_vol`），分别进行 Christoffersen 似然比检验。
- 检验核心 `christoffersen_lr_core`：统计一阶马尔可夫转移矩阵（`n00`、`n01`、`n10`、`n11`），分别计算零假设（独立性）和备择假设（一阶相关性）下的对数似然，构造 LR 统计量，查卡方分布（df=1）得 p 值。
- 无条件覆盖率 `emp_cov = 1 - mean(violations)`，要求 ≥ 93.5%。
- 各体制下独立性 p 值均 ≥ 1% 方可通过。

**设计哲学**  
- Christoffersen（1998）检验是 VaR 回测的标准工具，同时考察“覆盖率是否准确”（无条件覆盖）和“违规是否独立”（条件独立），两者兼具才能证明风险模型的可靠性。
- 波动率体制划分是对传统 Christoffersen 检验的拓展——因为高风险时期违规更可能聚集，分体制检验能更精细地定位模型失效的根源。

**优点**  
- 分体制检验的设计极具洞察力，能区分“全样本覆盖达标但高波时期失效”的隐蔽风险。
- 极小样本时（`len(ser) < 10`）自动跳过 LR 计算并返回 p=1.0，避免了小样本下卡方近似的失效。

**客观缺陷**  
- **`violations` 的数据质量依赖 Phase_7**：如前文 Phase_7 所述，`daily_intervals` 在 Phase_6 中被硬编码为全 0.02 的常量，导致 `violations` 序列反映的是“收益是否落在 ±2% 内”而非真正的共形预测违规。这使 Christoffersen 检验失去了统计学意义——它在检验一个与模型预测无关的固定区间。
- 波动率体制划分使用滚动 20 日波动率，但 `violations` 可能较稀疏，导致分体制后样本量严重不足，多个体制被跳过。
- 无条件覆盖率保底线 `0.935` 对应 95% 置信区间的容忍偏差，但该值未根据样本量自适应调整，小样本下可能过于严苛。

---

### 5. `stress_test.py` —— 历史危机压力测试

**功能**  
- 从 `config` 读取 `stress_scenarios` 字典（预设三个危机窗口）。
- 对每个场景，从 `daily_nav` 中截取对应时间窗口的净值子序列。
- 调用 `compute_max_drawdown` 计算该窗口内的最大回撤。
- 若窗口与回测期无重叠（数据真空），则记录 `nan`。
- 结果存入 `context["stress_drawdowns"]`。

**设计亮点**  
- **时区自适应对齐**：代码中包含完整的时区处理逻辑——若 `nav_series.index` 有时区，则将 `start`/`end` 也本地化到相同时区，反之亦然。这彻底消灭了因 `tz-naive` 与 `tz-aware` 比较导致的切片空结果。
- 对无重叠窗口返回 `nan` 而非 `0`，明确区分“未发生回撤”和“无法评估”。

**客观缺陷**  
- 压力场景**仅输出最大回撤**，未评估回撤后的修复速度（如回补天数）、波动率放大倍数或 Sharpe 崩溃程度，信息维度单一。
- 预设场景仅覆盖 2015-2024 年的 A 股危机，对 2008 全球金融危机、2020 疫情熔断等更极端的黑天鹅事件未做覆盖（尽管 2020 疫情在 Phase_4 的危机窗口中有，但未纳入 Phase_8 的压力测试）。
- 若回测周期短于某个危机窗口，该场景被跳过，可能导致策略在未经历完整危机周期的情况下获得“虚假通过”。

---

### 6. `capacity_audit.py` —— 静态账户容量与冲击损耗审计

**功能**  
- 从上下文提取 `daily_weights`（权重矩阵）、`daily_nav`（净值序列）和 `daily_adv20`（ADV20 面板）。
- 若 `daily_adv20` 缺失，则使用 2000 万均值作为防御性兜底。
- 计算每日调仓金额：`order_amounts = |diff(weights)| * total_equity`。
- 计算每日个股参与率：`participation_rates = order_amounts / adv20_df`。
- 计算平方根冲击损耗：`impact_loss = participation_rates^0.5 * 0.001 * order_amounts`。
- 输出全局最大参与率 `global_max_part` 和累计冲击损耗 `total_impact_loss_nav`。
- 判定 `capacity_pass = global_max_part <= 5%`。

**设计哲学**  
- 平方根冲击模型（`impact = κ * (turnover)^α`，κ=0.001，α=0.5）是学术界和业界广泛接受的经验模型，能合理估算大额订单对价格的瞬时影响。
- 参与率 5% 的红线是审慎的流动性管理标准，防止策略在低流动性个股上过度主导交易。

**优点**  
- 对 ADV20 缺失的降级处理（2000 万兜底）保证了审计在任何数据条件下都能运行。
- 累计冲击损耗的量化输出为策略的“隐性成本”提供了直观参考。

**客观缺陷**  
- **冲击参数硬编码**：κ=0.001 和 α=0.5 被写死在代码中，未引用 `config` 中的同名参数（实际上 `config.py` 中根本没有定义 `kappa_impact` 和 `alpha_impact`），导致调参和敏感性分析困难。
- **ADV20 的数据可靠性存疑**：如前文 Phase_3/6 分析，`daily_adv20` 在 Phase_6 中基于 `bulk_history_cache` 预计算生成，但若该缓存因任何原因缺失或数据质量不佳，将全部回退至 2000 万均值，使流动性约束完全失效。
- 参与率计算使用 `daily_weights.diff()`，若两日权重相同则调仓金额为 0，这正确反映了换手，但未考虑因价格变动导致的被动再平衡需求（权重漂移）。

---

### 7. `step8_audit_stress_test.py` —— 主控编排器与一票否决

**功能**  
- 依次调用 `run_dsr_audit`、`run_christoffersen_test`、`run_stress_test`、`run_capacity_audit`。
- 提取四项审计的通过标志：`dsr_pass`、`christoffersen_pass`、`capacity_audit_pass`（注意：压力测试未纳入一票否决，仅输出结果供人工审阅）。
- 计算最终通过标志 `overall_pass = dsr_pass and christoffersen_pass and capacity_audit_pass`。
- 若通过，记录 `AUDIT PASSED`；若失败，记录 `AUDIT FAILED (Veto Blocked)`。
- 汇总审计摘要字典 `audit_summary`，包含所有关键指标。

**设计亮点**  
- 明确的三重门禁（DSR + Christoffersen + 容量）体现了“零容忍”风控文化，任何一项不达标即拒绝上线。
- 审计摘要的结构化输出便于下游 MLOps 系统进行可视化和人工复核。

**客观缺陷**  
- **压力测试未被纳入一票否决**：`stress_drawdowns` 仅记录在 `audit_summary` 中，但不影响 `overall_pass`。这意味着即使策略在历史危机中回撤 90%，仍可能通过审计（只要 DSR/Coverage/Capacity 达标），这严重削弱了压力测试的实际效用。
- `num_trials` 的传递问题：`safe_get_shadow(context, "num_trials", 100)` 在 `run_dsr_audit` 内部被调用，但 Phase_5 从未将实际搜索次数写入上下文。建议在 Phase_5 结束时将参数网格大小显式记录到 `pipeline_context['num_trials']`。
- 未实现 `sharpe_threshold` 的硬性检查——DSR p 值通过即放行，即使名义夏普低于 0.5 也可能通过（当 p 值 > 0.05 时），这与“目标夏普底线”的语义不完全一致。

---

## 整体评价与改进方向

### 系统优势
- **学术严谨性**：DSR 和 Christoffersen 检验均为量化金融领域的黄金标准，其正确实现体现了系统的专业深度。
- **一票否决机制**：将审计结果直接与策略上线挂钩，是工业化风控的最高级别实践。
- **多维度覆盖**：从过拟合（DSR）到尾部聚集（Christoffersen）到极端场景（Stress）到流动性（Capacity），全面覆盖了策略失效的主要模态。
- **防御性编程**：影子副本、时区对齐、异常捕获等设计确保了审计过程本身的稳定性。

### 当前痛点
- **压力测试未纳入一票否决**：这是最严重的风控逻辑漏洞，需要立即修正。
- **`num_trials` 传递断裂**：DSR 的惩罚因子依赖真实的参数搜索次数，但 Phase_5 未提供该信息，导致 DSR 惩罚失真。
- **`violations` 数据质量污染**：Phase_6 的硬编码区间导致 Christoffersen 检验失去统计意义，这是从 Phase_6 到 Phase_8 的数据契约严重断裂。
- **冲击参数硬编码**：容量审计的 κ 和 α 无法通过配置调整，限制了模型的可调性和敏感性分析能力。
- **压力场景覆盖不足**：缺少 2020 疫情、2008 金融危机等更极端的全球性崩盘场景。

### 未来演进
- **将压力测试纳入一票否决**：设定最大回撤阈值（如 `stress_drawdown_threshold: -0.50`），若任一危机场景回撤超过该阈值，则 `overall_pass = False`。
- **修复 `num_trials` 传递链**：在 Phase_5 的 `step_5_1_cv.py` 中，将实际搜索的 `D_MIN_SEARCH_SPACE` 与 `CV_FOLDS` 的乘积记录到 `context['num_trials']`。
- **修复共形区间数据流**：要求 Phase_6 输出真实的 `q_low`/`q_high` 面板（非占位常量），或由 Phase_8 在 `violations` 缺失时自动禁用 Christoffersen 检验并发出警报。
- **参数化冲击模型**：在 `config.py` 中增加 `kapha_impact` 和 `alpha_impact` 配置项，并让 `capacity_audit.py` 从配置读取。
- **扩展压力场景**：将 Phase_4 的 `CRISIS_WINDOWS`（含 2020 疫情）同步到 Phase_8 的 `stress_scenarios`，并补充 2008 年数据（若回测覆盖）。
- **引入极端值指标**：除最大回撤外，增加 Calmar 比率、回撤恢复天数、VaR 超标倍数等二次指标，丰富压力测试的评价维度。

---

## 结语

Phase_8 是 Quant‑4 系统的最终安全闸门，它通过 DSR、Christoffersen、压力测试和容量审计四重防线，对策略的统计显著性、风险聚集性、极端脆弱性和流动性承载力进行了全面体检。其学术深度和工程防御性在量化回测系统中实属罕见。然而，`num_trials` 传递断裂、`violations` 数据污染和压力测试未纳入一票否决三大缺陷，严重削弱了审计的实际效力。修复这些数据契约断裂和逻辑漏洞，是将 Phase_8 从“形式上的审计”升级为“真正的风控壁垒”的关键一步。

---
*本文档基于 Quant‑4 系统 Phase_8 代码（版本 V2）编写，旨在为开发者和研究员提供内部架构透视。*