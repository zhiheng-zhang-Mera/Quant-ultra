# Phase_9_Live_MLOps：实盘机器学习运维深度解析

## 概述

Phase_9（Live MLOps）是 Quant‑4 系统的**生产环境运行与持续监控层**，负责将经过 Phase_8 审计通过的策略部署到实盘或模拟实盘环境，并对其进行全天候的运维保障。该阶段的核心使命是：确保策略在实盘中持续保持有效性，及时发现并应对模型衰退（Decay）、特征漂移（Feature Drift）、执行偏差（Execution Slippage）以及极端市场风险（Crowded Risk）。

Phase_9 严格遵循 Quant‑4 系统手册中“阶段九”的要求，实现了三个核心功能模块：

1. **影子对账（Shadow Reconciliation）**：将理论目标权重与实盘/回测实际执行权重进行逐日比对，计算平均绝对误差（MAE），若超过阈值则触发硬杀伤开关强制平仓。
2. **特征分布漂移检测（PSI Auditing）**：基于 Phase_5 产出的分数阶特征立方体，计算当前特征分布与基准分布的群体稳定性指标（PSI），当 PSI 连续超标时触发全量重训。
3. **嵌套风险遥测（Nested Risk Telemetry）**：实时监控策略净值波动率压缩（低波泡沫）与风格因子拥挤度（与基准指数相关性），当两者同时触发时，强制将多头仓位上限压缩至预设比例（如 20%），以防范系统性闪崩。

Phase_9 的设计将学术前沿的 MLOps 理念（如 PSI 监控、分层更新协议）与工程实战的硬杀伤防御（Hard Kill Switch）相结合，是策略从“回测成功”迈向“实盘可靠”的关键最后一公里。

---

## 模块总览与协作关系

| 文件 | 核心职责 | 依赖关系 |
|------|----------|----------|
| `__init__.py` | 暴露 `execute` 入口 | 被 `main_engine.py` 调用 |
| `config.py` | 定义 MLOps 全部阈值（MAE、PSI、波动率压缩、拥挤度、缓存路径） | 被所有子模块引用 |
| `shadow_reconciliation.py` | 计算目标权重与执行权重的 MAE，提供紧急硬杀伤清仓函数 | 依赖 `fsm_engine`（回测态）或 `counterparty_gateway`（实盘态） |
| `tiered_updater.py` | 计算 PSI 特征漂移，管理三层更新协议（在线/微调/全量重训）及新老模型平滑过渡 | 依赖 `fractional_features_cube` 和 `selected_features` |
| `telemetry_alerts.py` | 检测波动率压缩与风格拥挤度，双重触发时实施仓位上限强制约束 | 依赖 `data_bus` 和 `nav_history` |
| `step9_live_mlops.py` | 主控编排器，串联对账、漂移检测、遥测，导出状态快照并落盘 | 聚合所有子模块 |

---

## 各模块详解

### 1. `config.py` —— MLOps 阈值与缓存路径

**功能**  
定义 Phase_9 运行的全量参数：
- **影子对账**：`MAE_THRESHOLD = 1e-5`（平均绝对误差上限），`WATCHDOG_TIMEOUT = 30` 秒。
- **PSI 漂移**：`PSI_THRESHOLD = 0.25`，`PSI_CONSECUTIVE_DAYS = 5`（连续超标天数触发重训），`SMOOTHING_PERIOD = 25`（新老模型线性步进切换周期）。
- **嵌套风险**：`CROWDED_CORR_THRESHOLD = 0.95`（风格相关性上限），`VOL_COMPRESS_QUANTILE = 0.10`（波动率压缩分位数底线），`CROWDED_RISK_CAP = 0.20`（双重触发时最大多头仓位）。
- **缓存路径**：`CACHE_PARQUET_DIR` 和 `CACHE_FEATHER_DIR`，用于持久化每日 MLOps 状态快照。
- `DEFAULT_MLOPS_CONFIG` 统一封装所有参数，供下游安全引用。

**设计亮点**  
- 严格遵循系统手册 README 规范，将缓存路径显式分离，便于灾备追溯。
- 参数集中且带有明确注释，便于人工调优。

**客观缺陷**  
- `MAE_THRESHOLD = 1e-5` 对于实盘滑点来说过于严苛（实际执行价格与目标价格偏差可能达到 10 bp 以上，对应权重误差可能达 0.0001 量级），可能导致频繁误报硬杀伤。
- 所有阈值均为静态预设，未支持从外部配置动态覆写，在策略适配不同资金量级或市场时缺乏灵活性。

---

### 2. `shadow_reconciliation.py` —— 影子对账与硬杀伤开关

**功能**  
- **`run_shadow_reconciliation`**：根据 `is_live` 标志，从 `counterparty_gateway`（实盘）或 `fsm_engine`（回测）获取实际持仓权重 `executed_weights`，与 `target_weights` 比较，计算所有资产的 MAE。若 MAE 超过 `MAE_THRESHOLD`，则标记 `recon_passed = False`。
- **`trigger_physical_hard_kill_switch`**：紧急特权函数，强制平仓所有持仓，将账户全部转换为现金，并返回清算报告。同时支持实盘网关和回测引擎。

**设计哲学**  
- 影子对账是 MLOps 的“照妖镜”，能及时发现因滑点、流动性不足或系统延迟导致的执行偏差，防止隐性亏损累积。
- 硬杀伤开关是终极风险控制手段，当系统检测到不可恢复的异常（如网关连接丢失、MAE 严重超标）时，可以一键清仓，保全本金。

**优点**  
- 对实盘和回测态的统一处理接口（`counterparty_gateway` vs `fsm_engine`）设计优雅，便于测试和调试。
- 硬杀伤函数的实现覆盖了两种执行环境，提供了统一的应急出口。

**客观缺陷**  
- **MAE 计算过于简化**：仅计算算术平均绝对误差，未考虑资产权重差异的重要性（例如大市值股票和小市值股票的误差对组合风险的影响不同），也未区分因价格变动导致的被动漂移与主动调仓偏差。
- **硬杀伤的触发条件不明确**：`run_shadow_reconciliation` 仅计算 MAE 并标记 `recon_passed`，但并未在 MAE 超标时自动调用 `trigger_physical_hard_kill_switch`，实际调用责任交给了上层调用者（`step9_live_mlops.py` 中未调用），导致硬杀伤开关形同虚设。
- `counterparty_gateway` 的接口规范未定义（假设存在 `get_portfolio_snapshot` 和 `close_all_market_positions` 方法），缺乏抽象基类约束，易产生运行时接口不匹配。

---

### 3. `tiered_updater.py` —— 分布漂移检测与分层更新协议

**功能**  
- **`_compute_conformal_psi`**：计算单个特征的群体稳定性指标（PSI），通过等频分箱（默认 10 箱）比较基准分布与目标分布的差异，加入极小正数防止对数溢出。
- **`evaluate_distribution_drift`**：
  1. 从 `pipeline_context` 中提取 Phase_5 产出的 `fractional_features_cube`（三维：日期×资产×特征）和 `selected_features`（特征索引列表）。
  2. 取前 `lookback` 个交易日作为基准分布，最近 10 个交易日作为目标分布，对每个选中特征计算 PSI，取平均得 `mean_psi`。
  3. 若 `mean_psi >= PSI_THRESHOLD`，则 `psi_consecutive_breaches` 加 1；否则清零。
  4. 若连续超标天数达到 `PSI_CONSECUTIVE_DAYS`，则触发 `tier3_retrain_active = True`（全量重训）。
  5. 管理新老模型平滑过渡：若重训触发，设置 `alpha_new_model = 1/SMOOTHING_PERIOD`，之后每日递增，直至 `alpha_new_model = 1`，完成切换。

**设计哲学**  
- PSI 是业界标准的风控指标，用于监测模型输入分布的变化，是判断模型是否需要重新训练的重要依据。
- 三层更新协议（在线学习、微调、全量重训）在代码中表现为两级：`tier3_retrain_active` 对应全量重训，而在线学习或微调未显式实现，但可通过 `alpha_new_model` 的平滑过渡机制间接支持增量更新。
- 平滑步进切换避免了突然切换模型带来的净值跳跃，是稳健的 MLOps 实践。

**优点**  
- 直接从 Phase_5 的特征立方体读取数据，而非重新计算特征，确保了漂移监控与模型训练的特征定义完全一致（修复了 Flaw A-3）。
- PSI 计算中使用了分箱技术，对异常值鲁棒，且支持多维特征的平均化，简化了监控指标。

**客观缺陷**  
- **PSI 计算基准选取粗糙**：使用前 `lookback` 个交易日作为固定基准，未考虑季节性或市场 regime 变化，可能导致因自然市场周期变化而误报漂移。
- **目标窗口仅 10 天**：过短的目标窗口会使 PSI 对短期噪声过于敏感，建议使用至少 20~30 个交易日。
- **特征维度对齐风险**：`selected_features` 是特征索引列表，但 `feature_cube` 的形状为 `(T, N, F)`，其中 `F` 可能大于 `max(selected_features)`，索引访问安全，但若 `selected_features` 中包含超出 `F` 的索引，则会引发 IndexError（未做边界检查）。
- **三层更新协议未完整实现**：仅提供了全量重训（Tier 3）的触发，Tier 1（在线学习）和 Tier 2（微调）未体现，且重训的具体执行逻辑（重新运行 Phase_5）未在本阶段实现，仅通过标志位传递到外部。

---

### 4. `telemetry_alerts.py` —— 嵌套风险主动遥测

**功能**  
- 提取净值历史 `nav_history`，计算滚动 `volatility_window`（默认 20 日）的收益率波动率。
- **条件 B（低波泡沫）**：若当前波动率低于历史波动率的 `VOL_COMPRESS_QUANTILE`（默认 10%）分位数，则标记 `condition_b_triggered = True`。
- **条件 A（风格拥挤度）**：通过 `data_bus.get_benchmark_code()` 获取基准指数（如沪深 300），加载其历史价格，计算策略收益与基准收益的滚动相关性，若相关性超过 `CROWDED_CORR_THRESHOLD`（0.95），则标记 `condition_a_triggered = True`。
- **嵌套触发**：若条件 A 与 B 同时触发，则将 `enforce_crowded_allocation_cap` 设置为 `CROWDED_RISK_CAP`（20%），强制限制多头总仓位，否则设为 1.0（无额外限制）。

**设计亮点**  
- 双重条件的设计极具洞察力：低波压缩 + 高风格拥挤往往是“暴风雨前的宁静”，此时降低仓位是审慎的风控行为。
- 直接使用 `data_bus.get_benchmark_code()` 获取指数代码，保证了与数据总线的一致性。

**客观缺陷**  
- **相关性计算可能使用未来数据**：`load_asset_history` 加载指数数据时，结束日期为 `current_date_str`，但若 `current_date_str` 为当日收盘前，则会包含当日未完成数据，导致前向偏差。
- **策略收益与基准收益的对齐不严谨**：`bench_returns.index` 与 `portfolio_returns.index` 分别通过不同方式转换为字符串，可能因时区或格式差异导致 `common_dates` 为空，从而跳过拥挤度检测。
- **低波条件触发过于敏感**：使用历史 10% 分位数作为阈值，在市场长期低波时期可能频繁触发，导致不必要的仓位限制。
- **仓位上限实施未真正生效**：虽然设置了 `enforce_crowded_allocation_cap`，但该值仅记录在 `context` 中，并未被 Phase_6/7 的实际优化器或执行层使用，需在 Phase_6 的凸优化器中显式引入该约束才能生效。

---

### 5. `step9_live_mlops.py` —— 主控编排器与状态快照

**功能**  
- 获取当前日期（`current_date`），若缺失则使用系统时间。
- 依次调用：
  1. `run_shadow_reconciliation`（影子对账）
  2. `evaluate_distribution_drift`（PSI 漂移检测）
  3. `process_nested_risk_telemetry`（嵌套风险遥测）
- 打包结果字典 `result_update`，包含 MAE、recon_passed、mean_PSI、连续超标天数、`alpha_new_model`、重训标志、仓位上限等。
- 将结果导出为 Parquet 和 Feather 格式，存入 `Phase Result/parquet/Phase 9/` 和 `Phase Result/feather/Phase 9/` 目录，实现每日状态持久化。
- 更新 `pipeline_context` 并返回。

**设计亮点**  
- 双格式落盘（Parquet + Feather）兼顾了压缩比（Parquet）和读取速度（Feather），符合手册中的最佳实践。
- 显式记录 `execution_timestamp`，便于事后审计和回放。

**客观缺陷**  
- **未调用硬杀伤开关**：尽管 `run_shadow_reconciliation` 计算了 `recon_passed`，但 `step9_live_mlops` 中未对该标志进行任何动作（如报警或触发硬杀伤），仅将其作为字段输出，削弱了其防御价值。
- **重训标志未被外部消费**：`tier3_retrain_active` 虽被设置，但 `step9_live_mlops` 并未触发实际的重训流程（例如通过消息队列通知 Phase_5 重新运行），该标志仅被记录，需由外部调度器轮询并执行。
- **缓存路径硬编码**：`CACHE_PARQUET_DIR` 和 `CACHE_FEATHER_DIR` 在 `config.py` 中基于 `PROJECT_ROOT` 构建，未提供动态指定或云存储支持，在分布式环境中可能不适用。

---

## 整体评价与改进方向

### 系统优势
- **完整的 MLOps 闭环**：从执行对账、分布漂移到风险遥测，覆盖了实盘运维的三大核心领域。
- **学术与工程并重**：PSI、滑动相关性、波动率分位数等统计手段专业，硬杀伤开关和仓位上限等工程手段果断。
- **高可审计性**：每日状态快照的持久化为事后分析和模型回滚提供了基础。
- **特征层面的一致性**：PSI 直接复用 Phase_5 的特征立方体，避免了特征定义偏差。

### 当前痛点
- **硬杀伤逻辑未串联**：`recon_passed` 仅作为数据输出，未与硬杀伤函数联动，紧急情况无法自动响应。
- **嵌套风险的仓位约束未实际执行**：`enforce_crowded_allocation_cap` 仅记录，但未被 Phase_6 的凸优化器或 Phase_7 的执行引擎使用，须在优化器中添加对该参数的动态约束。
- **PSI 重训标志未被消费**：`tier3_retrain_active` 仅作为状态标记，但 Phase_9 不负责触发重训，需外部调度系统监控该标志并调用 Phase_1~5 的流水线。
- **MAE 阈值过于严苛**：`1e-5` 在实盘滑点下极易触发，导致频繁误报，应基于历史滑点统计动态设定。

### 未来演进
- **串联硬杀伤逻辑**：在 `step9_live_mlops` 中，若 `recon_passed == False` 且 `is_live == True`，则自动调用 `trigger_physical_hard_kill_switch`，并发送告警。
- **集成仓位上限到优化器**：在 Phase_6 的 `convex_optimizer.py` 中，增加从 `context` 读取 `enforce_crowded_allocation_cap` 的逻辑，作为全局权重上限约束。
- **建立重训触发器**：设计一个外部调度器（如定时任务），每天检查 `mlops_status` 快照中的 `tier3_retrain_active`，若为 True，则触发完整流水线（Phase_1~5）重新运行并生成新模型。
- **动态 MAE 阈值**：基于历史滑点的滚动分位数（如 95% 分位数）动态调整 `MAE_THRESHOLD`，避免人为设定的固定值过于极端。
- **扩展 PSI 监控维度**：除特征分布外，增加对标签分布、模型输出分布（预测概率）的监控，形成多维度漂移检测体系。

---

## 结语

Phase_9 是 Quant‑4 系统中从“离线回测”走向“在线实盘”的运维中枢。它在影子对账、特征漂移监控和嵌套风险预警方面的设计体现了工业级 MLOps 的成熟思考。然而，当前实现中“硬杀伤未串联”、“仓位上限未生效”和“重训标志未消费”三大断层，使得 Phase_9 目前更像一个“监控仪表盘”而非“全自动运维系统”。补齐这些执行链路，将 Phase_9 与 Phase_6/7/外部调度器深度集成，才能使 Quant‑4 真正具备全天候自动化的生产级可靠性。

---
*本文档基于 Quant‑4 系统 Phase_9 代码（版本 V2）编写，旨在为开发者和研究员提供内部架构透视。*