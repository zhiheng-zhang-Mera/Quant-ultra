# Quant-Ultra

[中文](#中文) · [English](#english)

> 研究与工程验证系统，不构成投资建议。回测、情绪评分和确定性测试均不能保证未来收益。

## 中文

Quant-Ultra 是面向 A 股、ETF 与美股研究的可审计量化流水线，覆盖数据质量、时点特征、另类数据、机器学习、运筹优化、交易成本、回测、压力测试、MLOps、CIO 治理与观察型投顾。每个阶段输出中英双语 Markdown/JSON 证据报告。

### 核心原则

- Point-in-Time：新闻、论坛、价格和标签只使用当时已经发布的信息。
- 风险优先：仓位受现金缓冲、单标的/行业上限、换手和流动性约束。
- 成本后收益：佣金最低收费、经手费、证管费、卖出印花税、滑点及 ETF 管理费均进入计算。
- 失败关闭：审计或对账不通过时返回 `HOLD_FOR_REVIEW`，Phase 11 降级为 `OBSERVATION_ONLY`。
- 可复核：每阶段报告包含 Git 哈希、运行时间、结构摘要和 SHA-256。

### 十一阶段完整功能

| 阶段 | 功能 | 主要输出与结论含义 |
|---|---|---|
| Phase 1 | 标的池、退市残值、交易状态、流动性和容量筛选 | `assets`、`adv_data`、存续矩阵、AUM 上限；判断数据底座能否进入研究流程 |
| Phase 2 | 训练/验证/测试切片、跨市场日历对齐、embargo | 时间隔离证据；防止训练与测试窗口重叠 |
| Phase 3 | PIT 特征、市场状态、新闻/论坛情绪、资金池变化 | `feature_panel_*`、`alternative_signals`、来源哈希；缺源时明确标记而不伪造情绪 |
| Phase 4 | 方向/收益标签、事件去重、样本权重 | `y_clf_all`、`y_reg_all`、`sample_weights`；限制重复事件过度计权 |
| Phase 5 | 方向分类、分位数模型、特征选择与校准 | 模型、特征、预测区间和误差证据；不直接等同交易信号 |
| Phase 6 | Black–Litterman、稳健协方差、风险预算、凸优化 | 每日目标权重、区间、ADV20；包含现金、集中度、换手和成本约束 |
| Phase 7 | FSM 回测、停牌/涨跌停、整手成交、冲击和费用 | 净值、收益、违规、费用分类账；体现可执行结果而非理想权重 |
| Phase 8 | DSR/覆盖率、容量、冲击和历史压力测试 | `audit_summary`、`audit_passed`；关键红线失败即阻止执行 |
| Phase 9 | 目标/执行仓位对账、PSI 漂移、拥挤度治理 | MAE、漂移和拥挤门禁；异常时保持人工复核 |
| Phase 10 | CIO 双语治理汇总和参数提案 | `cio_decision`、证据清单；不自动接受未经验证的参数 |
| Phase 11 | 候选、买点、仓位、止盈止损和持仓问答 | 双语报告与 CSV；治理未通过时仅观察、不可执行 |

### 另类数据输入

Phase 3 支持 `news_input_path` 和 `forum_input_path`，文件为 CSV 或 JSONL，至少包含：

```text
published_at,symbol,text
2026-08-03T08:00:00+08:00,600519.SH,公司披露增长与回购计划
```

晚于运行时点的记录会被排除。系统分别计算新闻和论坛词典情绪，并用 `close × volume` 构造 5 日相对 20 日资金池变化；综合权重为新闻 35%、论坛 25%、资金池 40%。词典模型不能可靠理解反讽、否定、传闻或操纵性发帖，正式使用应接入授权来源和经过验证的中文模型。

Phase 3 还会动态探测本地 Ollama 和 `local_llm_model`。模型存在时，系统仅对最新的有限文本做额外情绪分析；默认总计不超过 6 条、每个标的不超过 2 条、每条不超过 300 字，关闭推理过程并将单次批量调用硬限制为 20 秒。模型不存在、Ollama 未启动、超时或返回格式异常时，系统自动保留词典结果并继续流水线。相关开关和限额位于 `Main/default_param.yaml`。

### D 盘安装与运行

```powershell
Set-Location D:\Quant-Ultra\Quant-4
$env:TEMP='D:\Quant-Ultra-Env\tmp'
$env:TMP='D:\Quant-Ultra-Env\tmp'
D:\Quant-Ultra-Env\venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Quant-Ultra-Env\venv\Scripts\python.exe Main\main.py --symbols "600519.SH,000001.SZ,510300.SH" --force-recompute --non-interactive
```

常用参数：`--only-phase 6` 运行目标及依赖；`--resume-from 6` 从指定阶段继续；`--offline` 只用缓存；`--download-workers 1` 限制并发。报告位于 `Quant-4/reports/runs/<run-id>/`，含双语标题、结论解读、输出摘要、术语表和证据哈希。全部请求阶段结束后，同一目录会自动生成自包含的 `execute_report.html` 和结构化底稿 `execute_report_data.json`，汇总执行结论、资源分配、阶段证据与耗时、治理门禁、成本、净值以及 Phase 11 四阶段主导方法。

### 自适应分布式计算

初始化时系统检查逻辑/物理 CPU、可用内存、GPU、D 盘剩余空间、市场 HTTPS、DNS 和本地 Ollama。随后生成 CPU、I/O、下载、数据加载、优化和模型训练预算，并写入 `compute_audit`。CPU 工作数同时受物理核心、保留核心、每工作进程 1.5 GB 可用内存和配置上限约束；离线时下载并发自动降为 1。Phase 1/3 的 I/O 池、Phase 5 的 LightGBM 线程、Phase 6 的优化池以及 NumPy/BLAS/OpenMP 线程统一使用该预算，避免不同模块各自占满设备导致过度订阅。GPU 会被探测并记录，但只有确认对应库已构建 GPU 后端并显式设置 `distributed_gpu_backend_ready` 才会启用，避免把“检测到显卡”误当成“已使用显卡”。联机失败时自动保持 CPU/缓存路径，用户显式设置的工作线程数优先保留。

### 风险与成本默认值

- 现金缓冲 5%，动态最低有效投资仓位 10%，单日换手上限 25%。
- 单标的建议不高于 8%，并受 0.75% 组合损失预算约束。
- 买点基于 MA20/ATR，禁止给出高于最新价的追高区间。
- 止盈 3%–12%，须覆盖预计往返成本并保留最低净利润目标。
- 实际券商佣金和基金管理费必须在 `Main/default_param.yaml` 按合同调整。

### 四阶段动态决策链

选择、建仓、持仓和止盈各自包含至少八种方法：趋势、动量、均值回归、风险调整、回撤韧性、波动突破、流动性和情绪选择；ATR 回撤、均线回踩、突破确认、波动分批、流动性、价值区、动量延续和风险预算建仓；趋势跟随、跟踪止损、波动控制、回撤防护、信号持续、流动性监控、时间止损和利润保护持仓；ATR、波动带、跟踪退出、风险收益、阻力位、时间衰减、流动性退出和分批止盈。

每阶段先根据趋势、波动、回撤、成交活跃度和情绪状态调整 softmax 门控权重。上一阶段权重最高的方法通过显式转移矩阵对下一阶段兼容方法增加先验、对不兼容方法减权。最终结论使用 `atanh` 非线性池化和前两名方法协同项，而不是分数的线性加权平均。报告会保存各方法分数、动态权重、主导方法和跨阶段转移增益，便于复核。

### 验证与术语

```powershell
D:\Quant-Ultra-Env\venv\Scripts\python.exe -m pytest tests -q
D:\Quant-Ultra-Env\venv\Scripts\python.exe tests\run_acceptance.py
git diff --check
```

PIT = 时点可见信息；NAV = 账户净值；ADV20 = 20 日平均成交额；VaR/CVaR = 风险价值/条件风险价值；PSI = 群体稳定性指数；DSR = 校正多重尝试后的夏普证据；Embargo = 训练与测试间的时间隔离带。

单元测试证明接口和规则在测试样本上成立，不证明第三方数据真实或未来盈利。免费源可能限流/改版；模拟成交不能替代券商回单；小标的池运行只是工程验收。

## English

Quant-Ultra is an auditable research pipeline for China A-shares, ETFs, and US equities. It covers data quality, PIT features, alternative data, ML, operations-research allocation, execution costs, backtesting, stress testing, MLOps, CIO governance, and observation-only advisory output.

### Complete workflow

1. Phase 1 builds the survivorship-aware universe, liquidity evidence, and capacity limits.
2. Phase 2 creates isolated train, validation, test, and embargo windows.
3. Phase 3 builds PIT features and processes optional news/forum sentiment and turnover-pool changes.
4. Phase 4 creates labels and event-aware sample weights.
5. Phase 5 trains and calibrates direction and quantile models.
6. Phase 6 solves robust weights under cash, concentration, turnover, liquidity, and cost constraints.
7. Phase 7 runs the execution FSM with board lots, halts, slippage, taxes, commissions, and ETF fees.
8. Phase 8 performs coverage, capacity, statistical, and stress audits.
9. Phase 9 reconciles target/executed holdings and monitors drift and crowding.
10. Phase 10 produces the CIO governance decision and controlled parameter proposals.
11. Phase 11 emits entry ranges, sizing, exits, and friction estimates; failed gates force observation-only mode.

Configure `news_input_path` and `forum_input_path` with CSV/JSONL records containing `published_at`, `symbol`, and `text`. Future-dated records are excluded. Missing optional sources are reported and receive neutral scores; the system never fabricates sentiment evidence.

Phase 3 dynamically checks the local Ollama model configured by `local_llm_model`. If present, it enhances only a bounded recent sample (6 records total, 2 per symbol, 300 characters each, reasoning disabled, and a 20-second batch timeout by default). A missing model, stopped service, timeout, or malformed response falls back to lexical sentiment without blocking the pipeline.

Each phase writes bilingual Markdown and machine-readable JSON to `Quant-4/reports/runs/<run-id>/`, including an interpretation, glossary, output summary, Git hash, and evidence digest. After all requested phases finish, the same directory receives a self-contained `execute_report.html` and auditable `execute_report_data.json` consolidating execution status, compute allocation, phase evidence, governance gates, costs, final NAV, and Phase 11 dominant methods.

At startup, the adaptive compute orchestrator inspects CPU, available memory, GPU, D-drive capacity, market/DNS connectivity, and local Ollama. It produces bounded budgets for downloads, I/O loading, optimization, model training, and numerical libraries. Offline downloads fall back to one worker; missing GPUs or connectivity never block the CPU/cache path, and explicit user worker overrides are preserved.

Engineering acceptance, backtests, sentiment scores, and deterministic reconciliation do not guarantee investment performance. A failed audit or reconciliation produces `HOLD_FOR_REVIEW`; Phase 11 remains `OBSERVATION_ONLY` and must not be treated as executable advice.

Selection, entry, holding, and take-profit now form a regime-gated expert chain with at least eight methods per stage. Market trend, volatility, drawdown, liquidity, and sentiment change each stage's softmax weights. The dominant method directly shifts compatible priors in the following stage through an explicit transition matrix. Decisions use nonlinear `atanh` pooling plus a top-expert interaction term, rather than a linear weighted sum.
