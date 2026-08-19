# Quant-Ultra — Auditable Quantitative Research & Adaptive Rotation System

**可审计量化研究流水线 · 自适应多因子轮动引擎**

> Research & engineering validation system. **This is not investment advice.** Backtests, sentiment scores and deterministic tests cannot guarantee future returns. / 研究与工程验证系统，**不构成投资建议**。回测、情绪评分与确定性测试均不能保证未来收益。

[中文](#中文) · [English](#english)

---

## 中文

Quant-Ultra 是一套面向 A 股、ETF 与美股研究的**可审计端到端量化流水线**，覆盖数据质量、时点（Point-in-Time）特征、另类数据、机器学习、运筹优化、交易成本、回测、压力测试、MLOps、CIO 治理与观察型投顾，并内置一套**自适应多因子周轮动引擎**（`weekly_rotation`）。每个阶段均输出中英双语 Markdown/JSON 证据报告，含 Git 哈希、运行时间与 SHA-256，可完全复现。

本项目可作为**研究型工程作品**用于申请材料：其方法论强调无未来函数（PIT）、防数据泄露（walk-forward + embargo）、多重检验校正（DSR）、失败关闭治理（fail-closed），以及对样本外表现与口径变更的诚实披露。

### ✦ 核心原则

| 原则 | 说明 |
|---|---|
| Point-in-Time | 新闻、论坛、价格与标签只使用当时已发布的信息，杜绝未来函数 |
| 风险优先 | 现金缓冲、单标的/行业上限、换手与流动性约束贯穿仓位构建 |
| 成本后收益 | 佣金、经手费、证管费、印花税、滑点与 ETF 管理费全部进入计算 |
| 失败关闭 | 审计或对账不通过返回 `HOLD_FOR_REVIEW`，Phase 11 降级 `OBSERVATION_ONLY` |
| 可复核 | 每阶段报告包含 Git 哈希、运行时间、结构摘要与 SHA-256 |

### ✦ 十一阶段完整流水线

| 阶段 | 功能 | 主要输出 |
|---|---|---|
| Phase 1 | 标的池、退市残值、交易状态、流动性与容量筛选 | `assets`、`adv_data`、存续矩阵、AUM 上限 |
| Phase 2 | 训练/验证/测试切片、跨市场日历对齐、embargo | 时间隔离证据，防窗口重叠 |
| Phase 3 | PIT 特征、市场状态、新闻/论坛情绪、资金池变化 | `feature_panel_*`、`alternative_signals`、来源哈希 |
| Phase 4 | 方向/收益标签、事件去重、样本权重 | `y_clf_all`、`y_reg_all`、`sample_weights` |
| Phase 5 | 方向分类、分位数模型、特征选择与校准 | 模型、特征、预测区间与误差证据 |
| Phase 6 | Black–Litterman、稳健协方差、风险预算、凸优化 | 每日目标权重、区间、ADV20 |
| Phase 7 | FSM 回测、停牌/涨跌停、整手成交、冲击与费用 | 净值、收益、违规、费用分类账 |
| Phase 8 | DSR/覆盖率、容量、冲击与历史压力测试 | `audit_summary`、`audit_passed` |
| Phase 9 | 目标/执行仓位对账、PSI 漂移、拥挤度治理 | MAE、漂移与拥挤门禁 |
| Phase 10 | CIO 双语治理汇总与参数提案 | `cio_decision`、证据清单 |
| Phase 11 | 候选、买点、仓位、止盈止损与持仓问答 | 双语报告与 CSV；治理未通过仅观察 |

### ✦ 系统架构

```mermaid
flowchart LR
    A[Phase 1-2<br/>数据底座与切片] --> B[Phase 3-4<br/>PIT特征与标签]
    B --> C[Phase 5<br/>ML模型与校准]
    C --> D[Phase 6<br/>凸优化仓位]
    D --> E[Phase 7<br/>FSM物理回测]
    E --> F[Phase 8-9<br/>审计/压力/对账]
    F --> G[Phase 10-11<br/>CIO治理与投顾]
    G -->|HOLD_FOR_REVIEW / OBSERVATION_ONLY| A
```

### ✦ 自适应多因子轮动引擎

`Quant-4/Main/weekly_rotation.py` 实现收盘信号 → 次日开盘执行的无未来函数周轮动，并在传统多因子打分层之上叠加**自适应风险调整**：

```mermaid
flowchart TD
    S[复合因子打分<br/>动量/趋势/反转/低波/股息] --> R[市场状态检测<br/>MA40/MA10 + ML logit]
    R --> E{防御状态?}
    E -->|熊市/风险关闭/亢奋| H[避险资产轮动<br/>国债/黄金/货币 4选1]
    E -->|牛市| B[满仓强势标的<br/>全程无杠杆]
    H --> X[周内6%止盈/8%止损<br/>3个月持有上限]
    B --> X
    X --> P[收盘信号 → 次日开盘执行]
```

关键机制：

| 机制 | 说明 |
|---|---|
| ML 连续概率敞口 | logit `P(下月上涨)` 平滑映射敞口，替代二元空仓否决，避免长期空仓 |
| 避险资产轮动 | 防御状态下持有 120 日动量最强且 20 日趋势为正的国债/黄金/货币 ETF |
| 事件冲击 + 亢奋过滤 | 单日暴跌与 20 日暴涨极端区自动转入安全资产 |
| 小额收割 | 6% 止盈 / 8% 止损多次收割小利润，控制回撤深度 |
| 3 个月持有上限 | 强制轮出，不设最短持有 |
| 杠杆 | 全程禁用（个人资金不负债）：`confirm_leverage=1.0`、`max_gross_exposure=1.0`，无做空 |

### ✦ 回测表现（2016-01 ~ 2026-08，2578 交易日，诚实口径：10 万元本金，全池 PIT 复验后采纳的生产默认）

| 指标 | 诚实回测（PIT 全市场 5478 只、覆盖 100%、月频、显式费用、整手、无杠杆） | 目标 / 对照 |
|---|---|---|
| 年化收益 | **6.54%**（OOS 2022+：8.35%） | 沪深300(510300) ~5.3% ✅ |
| 夏普比率 | **1.01**（OOS 1.31） | ≥0.9 ✅ |
| 卡玛比率 | **0.64**（OOS 0.79） | ≥1.2 ❌ 未达（结构性上限，详见计划文档） |
| 最大回撤 | **-10.29%** | 沪深300 ~-44.8% ✅ |
| 回撤修复期（近3年窗口） | **181 交易日** | ≤126 ❌ 未达（进行中的 2026 回撤拉长修复期） |
| 季度超等权基准胜率 | **51.2%** | ≥50% ✅ |
| 季度超沪深300胜率 | 53.5% | ≥60% ❌ 披露 |
| 累计交易成本 | 10.6%（93 笔） | ✅ |
| 杠杆 | 0.00x（禁用） | ✅ |
| 平均总仓位 | ~73% | — |

> **2026-08-15 全池 PIT 复验里程碑（生产默认采纳）**：接入网络后，从新浪 klc 全历史通道（东财/腾讯/baostock 均被风控或限流，已按授权寻找替代途径）重建全池日线 5478/5478（覆盖率 100%，退市 248/248），巨潮 cninfo 股息 5420 只/50947 行。全池复验（106 次尝试注册，DSR 多重检验）：**`--profile robust`（固定 3% + z-score 3.0 冲击检测、8% 止损）6.54%/1.01/-10.29%（OOS 8.35%/1.31，DSR p=3.4e-06 通过）全面胜出旧生产默认（4.79%/0.73/-13.40%，OOS 4.80%/0.71），暖启动折线 4/4 折年化胜出（2024-2026 折 13.52% vs 8.62%、回撤 -8.6% vs -13.4%）。**已采纳为生产默认**；旧参数保留为 `--profile legacy` 供 A/B 对比。注意：新鲜数据源下旧生产默认的复测为 4.79%（与 2026-08-11 文档数字 6.25% 存在数据源差异），以全池复验口径为准。

> **2026-08-14 离线评估轮**（详见 `Quant-4/update plans/8-14-offline-evaluation-round1.md`）：在 421 只离线缓存池上完成引擎评估与风险控制修复——(1) 修复真实缺陷：**再平衡日的事件冲击风险处置曾被常规再平衡覆盖**（风险优先序，默认即生效）；(2) 新增默认关闭的机制：`rebalance_min_turnover`（最小换手再平衡门槛）、`event_shock_zscore`（波动自适应 z-score 事件冲击检测，固定 2.5% 阈值在小盘高波动池上实测误触发 158 次）与 `trailing_stop_pct`（峰值追踪止损，实测在本小盘池上回撤翻倍、仅作可选）；(3) 清除 `rank_candidates` 与 `weekly_rotation` 中未使用的死代码（字节兼容）；(4) 报告生成器修复过时/自相矛盾的"第三视角审查"文本（改为动态引用 summary）；(5) **账务自洽审计**：净值复利、敞口无杠杆、无执行日漂移界恒等式全部精确成立，止损路径"成本计入、换手不计"已注释明确；(6) 因子库额外因子（rsi14/idll20/MAX/Amihud）实测全部劣于基座——复合因子组合已近最优；(7) 套筒组合 40/30/20/10 用 `robust` 基准年化 1.14%→1.90%；(8) 另类信号通路端到端验证：治理门 PASS、方向响应正确（动量代理信号 OOS 9.96%/0.92，反转代理信号 -27.9% 回撤——**接线验证而非新 alpha**）；(9) 开源排名新增标注子集行（子集基线→robust→alt 综合百分位 0.31→0.42→0.50）；(10) 新增市场宽度门 `min_breadth_for_buys`（默认关，本池实测负结果——拒绝）与**暖启动折线验证**（每档一次全程回测后按 4 折切片：`robust` 在年化/夏普/卡玛上 **4/4 折胜出**、2/4 折回撤更小——全窗口优势并非单一时期伪影；冷启动折线方法被证实会引入不对称的冷启动噪声并反转结论，已记录方法论警告）；(11) **性能向量化**：per-day 循环 11.9 万次 pandas 标量 `.at` 访问（占运行时长 ~30%、随池规模线性增长）改为 numpy 数组 + 索引映射，421 只回测 **~40-60s → 9.9s（约 4-5×）**，重构前后 10 项指标字节一致；(12) **参数邻域稳定性**：robust 三参数单点扰动均显著优于基线（4.4-5.3% 年化平台，无刀锋）；(13) **资金敏感性披露**：robust 档 10万→30万→50万元本金年化 5.25%→6.10%→7.19%（夏普 0.55→0.71），整手约束随账户规模缓解——10 万口径为保守下界；(14) **2026-08-15 全池复验与采纳**：网络恢复后以替代数据源重建全池（5478/5478 覆盖 100%、cninfo 股息 5420 只），复验 `robust` 档全面胜出并**采纳为生产默认**（6.54%/1.01/-10.29%，OOS 8.35%/1.31，DSR p=3.4e-06，折线 4/4）；旧参数保留为 `--profile legacy`。(15) 测试套件 **173/173 全绿**。

> **2026-08-16 真实另类信号复验轮（Round 16-18）**（详见 `Quant-4/update plans/8-16-real-news-fullpool-revalidation.md`）：网络诊断确认此前"无网络"为**误报**——PowerShell/curl 走 Windows Schannel 层被沙箱限制（`SEC_E_NO_CREDENTIALS`），而 **Python/OpenSSL 通道实际可用**（baidu/sina/pypi 及全部项目数据源 TLS/HTTP 均 200）。新工具 `Quant-4/tools/fetch_real_news_sina.py` 从新浪个股新闻页抓取**生产池全部 38 只 12066 条真实新闻**（契约全列、`is_synthetic=0`、sha256 入不可变缓存）。真实信号 A/B（采纳档基座）：通路方向响应正确，但 **38 只全生产池上方向与 20 只子样本相反**（20 只时正情绪拖累/反转改善；38 只时真实信号轻微改善窗口 -3.89%→-3.07%、反转更差 -3.94%）且全窗口/OOS 差异均在 ±0.05pp 内——**真实新闻情绪无稳健方向性 alpha**（R16 小样本方向属偶然，不可外推）；免费源仅约 5 个月窗口、28% 覆盖率，无法支撑全池历史复验，`alternative_signal_weight` 保持 0。测试套件 **175/175 全绿**。

> **2026-08-16 Round 19.8：代码落盘审计 + 位置整理 + 利润/资产负债表因子激活**（详见 `Quant-4/update plans/8-16-round19-8-code-audit-fundamentals-activation.md`）：(1) **代码落盘审计**：生产/引擎/工具/测试全部已追踪；19 个研究/证据脚本从 gitignored 的 reports/_iter 纳入版本控制（`research/llm/`、`research/`、`research/walkforward/`、`tools/`），根锚定路径任意 CWD 可运行，sweep 治理测试更新（195/195 全绿）。(2) **激活利润/资产负债表因子**：baostock 利润表 598 只 + 资产负债表 595 只（pubDate PIT 对齐、断点续跑），与现金流合并 753 只 10 字段——生产默认 `--fundamental-factors` 从"缓存缺失静默空转"变为全部真正生效；修复合并排序确定性。(3) **激活后回测（全池 PIT 套筒 40/30/20/10 + 全基本面）**：**年化 9.21%、夏普 1.470、卡玛 1.609、最大回撤 -5.73%、月度胜率 67.2%、OOS 11.69%/1.83**（vs 仅 ocf_np：8.70%/1.366/1.316/-6.61%）——生产默认更新为全基本面激活配置。测试套件 **195/195 全绿**。

> **2026-08-16 Round 19.7：正交因子族——现金流质量因子 ocf_np 首个通过全部证据门并采纳**（详见 `Quant-4/update plans/8-16-round19-7-orthogonal-factors.md`）：(1) 事件因子（股东行为/业绩预告）38 只池全负 → 拒绝；(2) 拥挤度（amount_share）38 只池伪正、全池复核全面更差 → 拒绝（小池伪象第三次证实）；(3) 资金流端点被拒 → 数据不可得；(4) **现金流质量 ocf_np（经营现金流/净利润，Sloan 应计异常代理）**：并行抓取 top-600 流动性名单 506 只（东财现金流表，NOTICE_DATE 公告日 PIT 对齐，集成进 fundamental_factors 框架并修复周末公告被丢弃的 PIT 缺陷）；**套筒级（生产默认层）PIT 修复版证据：年化 6.55%→8.53%、夏普 1.085→1.356、卡玛 0.661→1.228（首次突破 1.0）、回撤 -9.91%→-6.94%、月度胜率 60.2%→64.8%、OOS 8.35%/1.35→10.07%/1.58——研究门全部检查通过 → 采纳进生产默认**；单书亦显著改善（年化 +2.2pp、OOS +1.2pp，卡玛 0.82 未达 1.0，如实披露）。测试套件 **194/194 全绿**。

> **2026-08-16 Round 19.6：开源/免费历史数据源——东财公告解除"全池历史复验"数据约束**（详见 `Quant-4/update plans/8-16-round19-6-history-sources-eastern-notices.md`）：(1) 实测 8 类候选源，**东财公告（data.eastmoney.com/notices）可用且完整历史**（✅ 采纳）；东财/新浪新闻均限 ~5 个月、财新付费、chinascope SSL 错误、HuggingFace CN-Market-Corpus 债券向无 A 股代码——其余不可用。(2) 新工具 `tools/fetch_history_notices_eastmoney.py` 抓取生产池 38 只 **73792 条历史公告**（1993-2026 IPO 至今，契约 LOADED，2016-2026 窗口 54898 条）。(3) **历史窗口 A/B（2016-2026 全 10 年、123 再平衡日、覆盖 94-96%）**：公告词典情绪无稳健 alpha（正权重全劣于 base、反转不优）——**R16/R18"无 alpha"结论在完整历史上最终确认，非覆盖不足伪影**。(4) **LLM 差异化在公告上远弱于新闻**（相关 0.71 vs 0.53、符号分歧 13.4% vs 59.8%、LLM 中性 80% vs 18%）——正式披露标题基本中性，8-13 假设不适用公告文体；LLM 仍修正词典系统性错误（"回购注销"误判、"计提减值"漏判）。测试套件 **189/189 全绿**。

> **2026-08-16 Round 19.5：方向 2 继续——回撤修复/卡玛剩余杠杆全测（证据门拒绝）**（详见 `Quant-4/update plans/8-16-round19-5-direction2-reentry-stops-sleeve.md`）：(1) **机制归因修正**——2026 回撤修复滞后的绑定约束是 **MA200 长期确认均线**（08-05 基准已收复 MA40/MA10>MA40/mom20>0，仅 close<MA200；R10"收复 MA40"归因不精确）。(2) **早期再入场规则**（`reentry_skip_confirmation_periods`，默认关，保留风险层敞口缩放的书切换）：全池 3 变体全负（年化 6.25-6.32% vs 6.54%、回撤 -12.5~-13.4% vs -10.3%，2016-18 折显著受损——熊市反弹陷阱；修复期 181 日不变）→ **拒绝**——策略等待完整确认的纪律价值高于错过反弹。(3) **止损/止盈再校准**（全池 5 变体）：收窄止损 6% 改善回撤/卡玛（-9.66%/0.658）但 **OOS 8.35%→7.18% 显著劣化**；其余全面更差 → **拒绝，8%/12% 为局部最优**。(4) **套筒权重再平衡**（全池 4 书一次运行 + 4 权重组合）：去 sprint（40/35/25/0）全窗口最优（年化 6.70%、卡玛 0.699、回撤 -9.57%）但 **OOS 8.35%→8.32% 轻微劣化、预注册 OOS 判据不通过 → 拒绝采纳，记录为邻近可选配置**；增 sprint 证伪（OOS 最好但全窗口最差，制度敏感）。**方向 2 结论：3 机制家族累计 23 配置全测，181 日修复期与卡玛 0.66 确认为结构性上限；生产默认保持套筒 40/30/20/10**。测试套件 **187/187 全绿**。

> **2026-08-16 Round 19：DeepSeek LLM 差异化验证 + 套筒正式采纳 + 股吧可用源**（详见 `Quant-4/update plans/8-16-round19-deepseek-llm-sleeve-adoption.md`）：(1) **LLM vs 词典差异化验证完成（硬件阻塞解除）**——用 DeepSeek 云端 API 对 **12066 条真实新闻全量评分**（403 批/0 失败/~45 分钟），LLM 与词典 **Pearson 相关仅 0.53**（8-13 合成文本上为 0.97）、**符号分歧 59.8%**、词典对 90% 真实标题判定中性而 LLM 仅 18%——**8-13 假设确认：真实文本让 LLM 产生显著差异化**；LLM 面板 A/B（38 只生产池）：w=0.20 信号窗口翻正 **+1.61%**（词典同权重 -2.59%）、OOS 5.82%→6.03%，但全窗口差异 ±0.2pp 内、`alternative_signal_weight` 保持 0（决定性验证需付费历史数据）。(2) **套筒 40/30/20/10 正式采纳为生产默认风险层**（R12/R13 全池证据 + 本轮生产池演示：年化 6.47%→6.75%、夏普 1.018→1.092、月度胜率 59.4%→61.7%、换手 0.349→0.061）；卡玛 0.66 与回撤修复期 181 日为结构性上限，单书保留为对照。(3) **股吧论坛情绪源解除 403**：JSON API 被风控但 HTML 列表页可用，新工具 `tools/fetch_forum_guba.py`（UTC+8 解析 + PIT 守卫）抓取生产池 38 只 **6036 条真实帖子**（契约 LOADED）。(4) **E2E 验证 PASS**：`build_alternative_signals`（main.py Phase-3 路径）Ollama 不可用→DeepSeek→词典回退全链走通（provider=deepseek、ANALYZED 6/6）。测试套件 **182/182 全绿**。

> **2026-08-15 采纳档后续轮（Round 10-13）**（详见 `Quant-4/update plans/8-15-selector-gate-rerun-recovery-iteration.md`、`Quant-4/update plans/8-15-alt-signal-fullpool-wiring.md`、`Quant-4/update plans/8-15-fullpool-sleeve-revalidation.md`、`Quant-4/update plans/8-15-walkforward-loader-fix-sleeve-folds.md`）：(1) **策略选择器证据门在采纳档全池重跑**：选择器 OOS 夏普 1.28 < 最优固定原型 momentum 1.37（OOS 年化 8.20% vs 8.64%）→ **保持禁用**（`strategy_selector=""`）；诚实披露选择器全窗口卡玛 0.75/回撤 -8.57% 为全部原型最优，但按预注册 OOS 门不通过，未改门放行；(2) **回撤修复机制迭代（全池 14 配置）全负结果**：`daily_regime_monitoring` 显著更差（5.01%/0.74/-15.53%）、`rebalance_min_turnover`/`drawdown_guard`/`ml_bear_floor`/`defensive_hold_safe_frac`/ML 曝光带端点（`ml_bear_low`/`ml_bull_high`）均无实质改进或更差——181 日修复期经 2026 回撤剖析确认为结构性滞后（策略下跌段 -7.16% 优于基准 -9.52%，修复需等基准收复 MA40、数据最后一日才达成），生产默认字节不变；(3) **另类信号通路全池接线复验**：情绪形态合成 PIT 面板（事件尖峰+衰减，非动量代理）在 5478 只采纳档上治理门 PASS、精确 as-of、方向响应正确，但全权重负贡献（5.30-5.40% vs 6.54%）、反转型 ≈ 中性——**接线验证而非新 alpha**，与 R4 结论一致；真实新闻/论坛语料全池复验仍受无网络/无数据缓存约束，`alternative_signal_weight` 保持 0；(4) **全池套筒复验（R3.5/R9 挂起项）**：套筒 40/30/20/10 在采纳档全池首次获得诚实证据——夏普 1.01→**1.09**（OOS 1.31→**1.35**）、最大回撤 -10.29%→**-9.91%**（研究证据门 `max_drawdown` 从 HOLD 翻转为 **PASS**）、月度胜率 57.8%→**60.2%**；折线验证 3/4 夏普/卡玛胜出、但 2024-26 强牛折让渡收益（12.24% vs 14.49%）——**套筒为风险平滑可选层，非严格占优**；生产默认保持单书，套筒证据如实记录；(5) **折线加载口径修复**：R9 折线原用 scratch loader（5503 只含 25 只 B 股），规范 PIT loader（5478 只）重跑后判决不变（robust 4/4 年化、3/4 夏普/卡玛），2024-26 折修正为 **14.49%**；(6) 开源排名百分位更正为采纳档口径 **0.74/0.85**（0.86/0.92 为旧默认）；测试套件 **174/174 全绿**。

![周轮动净值与回撤曲线](docs/images/weekly_rotation_equity.png)

![月度收益热力图](docs/images/weekly_rotation_monthly_heatmap.png)

> 口径说明：回撤修复期采用“最近 3 年窗口内峰值→新高最大回撤天数”（滚动监测口径，用户授权定义）；全窗口口径同时披露。OOS 定义为 2022-01 之后的样本。

### ✦ 策略决策层与基准治理（2026-08-19 同口径升级）

- 引擎已拆解并插入可选**策略决策层**（`Quant-4/Main/strategy_selector.py`：balanced / momentum / defensive / safe / sprint 五个原型 + 滞回切换），用样本外证据门控。**2026-08-15 在采纳档（生产默认）上重跑全池证据门**：选择器 OOS 夏普 1.28 低于最优固定原型 momentum 1.37（OOS 年化 8.20% vs 8.64%），**生产默认保持禁用**（`strategy_selector=""`），避免“为自适应而自适应”的过拟合。诚实披露：选择器全窗口卡玛 0.75 / 回撤 -8.57% 为全部原型中**最优**（滞回切换确实平滑了风险曲线），但按预注册的 OOS 证据门（须同时胜过最优固定原型）仍不通过——已如实记录，未改门放行。
- 2026-08-11 参数网格（月频+持仓延续+止盈带）将诚实成绩提升到 **6.35%/0.99**；早期“safe 7.71%”等数字被证实为稀疏股息面板导致的候选池坍缩伪影，已撤销。
- 生产研究结论改由 `Quant-4/Main/benchmark_governance.py` 的**内部同口径基准契约**支撑：被动、经典规则、因子、ML、当前 Quant-Ultra 与至少一个历史版本必须使用相同交易日索引、PIT 数据版本、账户规模、成本和交易约束，并自动关联 Git commit、参数版本及结果哈希。
- 既有 16 个开源非高频策略排名（`Quant-4/benchmark/open_source_comparison.py`）存在跨市场、跨时期和口径差异，现明确降级为 **AUXILIARY_REFERENCE_ONLY（辅助背景）**；历史 0.74/0.85 百分位不再作为 Quant-Ultra 性能优势或生产准入的核心证据。
- 生产候选须额外通过 `Quant-4/Main/statistical_validation_v2.py`：PSR、Holm 多重检验校正、PBO、块 bootstrap 区间、预定义参数邻域稳定性及 OOS 门槛缺一不可；证据缺失一律 `HOLD / OBSERVATION_ONLY`。正式实验与负结果由 `Quant-4/Main/experiment_registry.py` 的追加式哈希链统一登记，`ACCEPT / REJECT / HOLD` 均保留。

### ✦ 研究深度治理（2026-08-19 Wave 2）

- `Quant-4/Main/factor_research.py` 为生产因子提供独立诊断：Pearson IC / Rank IC / ICIR、月度稳定性、分组单调性、多期限衰减、换手与成本后价差、容量、行业/市值暴露、状态敏感性、冗余和组合增量信息。容量、暴露或状态证据缺失时因子保持 `HOLD / RESEARCH_ONLY`。
- LLM 输出契约由单一情绪分数升级为 `financial-event/v1`：事件类型、公司/行业/宏观/监管作用域、方向、重要性、新颖性、不确定性、影响期限、可靠性、score 与 confidence 均为结构化字段；同时记录与传统词典分数的差异。未完成独立 PIT/OOS 信号验证前，事件输出固定为 `RESEARCH_ONLY`，不改变生产权重。
- `Quant-4/Main/alternative_data_research.py` 将新闻、公告、论坛纳入统一覆盖率、双时间戳 PIT、延迟、来源版本、可靠性和跨来源冲突视图，并明确区分 `DATA_MISSING` 与 `NO_ELIGIBLE_EVENT`；资金流作为非文本另类数据在同一 Phase 3 结果中保留。三类文本源不完整时统一门禁保持 `HOLD / OBSERVATION_ONLY`。

### ✦ 外部泛化治理（2026-08-19 Wave 3）

- `Quant-4/Main/generalization_validation.py` 固化源市场代码、参数、因子、风险与执行版本，目标市场 `target_search_trials` 必须为 0，且参数哈希必须与冻结版本一致；外部市场结果按因子、风险和执行成本分解，迁移失败也输出正式 `TRANSFER_FAILED` 证据。
- 同一模块提供预注册的牛市、熊市、震荡、高波动、低波动、流动性冲击和极端日分类，并统一输出策略、基准、核心因子及风险保护机制的分状态表现，明确优势环境与弱势环境。
- 时间外验证窗口必须预先给定且互不重叠；样本、PIT、市场日历、币种、成本或交易制度元数据不完整时保持 `HOLD / OBSERVATION_ONLY`。合成测试只验证工程契约，不作为任何市场迁移有效性的实证结论。
- 冻结入口 `Quant-4/research/run_us_etf_generalization.py` 已在 2026-08-19 对固定美国 ETF 池完成 2,544 个交易日、零目标调参验证，结论为 **`TRANSFER_MIXED`**：年化收益高于 SPY，但 Sharpe 更低且回撤明显更差，因此只证明部分收益迁移，不支持风险调整后的普适优势。运行报告和行情缓存按 source-only 发布规则保留本地且被 Git 忽略；脚本会记录数据哈希、参数哈希、Git commit 与限制说明，可重复生成证据。

### ✦ 快速开始

```powershell
# 1) 创建或复用虚拟环境并安装依赖（含导入、pip check、编译与单元测试验收）
.\setup.ps1

# 2) 运行完整十一阶段流水线
.\Quant-4\.venv-full\Scripts\python.exe Quant-4\Main\main.py --force-recompute --non-interactive

# 3) 运行自适应周轮动回测并生成双语报告
.\Quant-4\.venv-full\Scripts\python.exe Quant-4\run_weekly_rotation.py

# 4) 运行防泄露的按标的自适应回测
.\Quant-4\.venv-full\Scripts\python.exe Quant-4\run_adaptive_backtest.py 600519 --kind stock --years 8

# 5) 验证
.\Quant-4\.venv-full\Scripts\python.exe -m pytest Quant-4\tests -q
```

完整命令、参数与配置说明见 **[UserGuide.md](UserGuide.md)**。

### ✦ 命令行速查

| 命令 | 用途 |
|---|---|
| `main.py --only-phase 6` | 仅执行 Phase 6 及其依赖 |
| `main.py --resume-from 6` | 从 Phase 6 断点恢复 |
| `main.py --offline` | 全离线调试（仅缓存） |
| `main.py --symbols 600519.SH,510300.SH` | 受限真实数据运行 |
| `run_weekly_rotation.py --start 2020-01-01` | 自定义回测起点 |
| `run_sleeve_portfolio.py [--pit] [--dynamic-stops]` | 四层 40/30/20/10 资金配置组合（保底/平衡/先锋/冲刺），可选动态 ATR 止盈止损 |
| `run_adaptive_backtest.py <code> --kind etf --disable-self-optimize` | 禁用跨运行参数迭代的诊断模式 |

### ✦ 项目结构

```text
Quant-Ultra/
├── Quant-4/
│   ├── Main/                  # 主引擎：流水线、数据总线、周轮动、ML 门控
│   ├── Phase_1 … Phase_11/    # 十一阶段模块
│   ├── run_weekly_rotation.py # 自适应周轮动回测入口
│   ├── run_adaptive_backtest.py
│   ├── benchmark/             # 开源非高频策略对比与综合排名
│   ├── tests/                 # 单元与验收测试
│   └── reports/               # 运行报告、CIO 决策、周轮动报告（gitignore）
├── docs/images/               # 文档配图（受版本控制）
├── README.md                  # 本文档
└── UserGuide.md               # 用户指南（双语）
```

### ✦ 验证、治理与术语

```powershell
python -m pytest tests -q
python tests\run_acceptance.py
git diff --check
```

- **PIT** = 时点可见信息；**NAV** = 账户净值；**ADV20** = 20 日平均成交额；**PSI** = 群体稳定性指数；**DSR** = 校正多重尝试后的夏普证据；**Embargo** = 训练与测试间的时间隔离带。
- 治理：任何阶段审计失败即 `HOLD_FOR_REVIEW`；Phase 11 在治理未通过时仅输出观察建议（`OBSERVATION_ONLY`），不产生交易授权。
- 免责：单元测试证明接口与规则在测试样本上成立，不证明第三方数据真实或未来盈利；免费数据源可能限流/改版；模拟成交不能替代券商回单。

### ✦ 相关文档

- [UserGuide.md](UserGuide.md) — 双语用户指南（安装、配置、运行、解读报告、故障排查）
- [docs/ENGINE_EVALUATION.md](docs/ENGINE_EVALUATION.md) — 引擎评估报告：结构/逻辑检查、盈利能力、综合评分与多轮迭代证据链（2026-08-15 更新至 R11）
- `Quant-4/update plans/8-8-adaptive-model-frontier.md` — 自适应模型修正与四目标可行性证据档案
- `Quant-4/update plans/8-9-update-plan.md` — 投产前迭代计划与诚实化减法记录
- `Quant-4/update plans/8-11-sleeve-dynamic-stops.md` — 四层资金配置 + 动态止盈止损的解耦模块化设计与证据门控
- `Quant-4/update plans/8-14-offline-evaluation-round1.md` — 离线评估轮 R1-R9（含全池复验与生产采纳证据链）
- `Quant-4/update plans/8-15-selector-gate-rerun-recovery-iteration.md` — R10：采纳档上策略选择器证据门重跑 + 回撤修复机制迭代
- `Quant-4/update plans/8-15-alt-signal-fullpool-wiring.md` — R11：全池另类信号通路接线复验
- `Quant-4/update plans/8-15-fullpool-sleeve-revalidation.md` — R12：全池套筒复验（挂起项闭合）+ 回撤机制收尾
- `Quant-4/update plans/8-15-walkforward-loader-fix-sleeve-folds.md` — R13：折线加载口径修复（规范 PIT 重跑，判决不变）+ 套筒折线证据（3/4 夏普/卡玛，强牛折让渡收益）
- `Quant-4/update plans/8-16-real-news-fullpool-revalidation.md` — R16-18：真实另类信号复验（网络误报诊断 + 真实新闻抓取 38 只 12066 条 + 方向敏感性确认，无稳健 alpha）
- `Quant-4/update plans/8-16-round19-deepseek-llm-sleeve-adoption.md` — R19：DeepSeek LLM 差异化验证（12066 条全量评分，相关 0.53/分歧 60%）+ 套筒正式采纳 + 股吧可用源 + E2E
- `Quant-4/update plans/8-16-round19-5-direction2-reentry-stops-sleeve.md` — R19.5：方向 2 剩余杠杆全测（早期再入场/止损止盈/套筒权重，3 机制家族 23 配置全拒绝，结构性上限确认）
- `Quant-4/update plans/8-16-round19-6-history-sources-eastern-notices.md` — R19.6：开源/免费历史数据源探索（东财公告 73792 条全历史 + 10 年窗口复验无 alpha + 公告 LLM 差异化远弱于新闻）
- `Quant-4/update plans/8-16-round19-7-orthogonal-factors.md` — R19.7：正交因子族（事件/拥挤度拒绝、现金流质量 ocf_np 采纳进生产默认，卡玛首次破 1.0）
- `Quant-4/update plans/8-16-round19-8-code-audit-fundamentals-activation.md` — R19.8：代码落盘审计 + 位置整理（research/ 纳入版本控制）+ 利润/资产负债表因子激活（9.21%/1.47/1.61/-5.73%）
- `Quant-4/benchmark/open_source_comparison.py` — 开源对比排名脚本（排名见其生成的 JSON）

---

## English

Quant-Ultra is an **auditable, end-to-end quantitative research pipeline** for China A-shares, ETFs and US equities, plus an **adaptive multi-factor weekly-rotation engine** (`weekly_rotation`). It covers data quality, Point-in-Time features, alternative data, machine learning, operations-research allocation, execution costs, backtesting, stress testing, MLOps, CIO governance and observation-only advisory output. Every phase emits bilingual Markdown/JSON evidence reports with Git hash, run time and SHA-256 for full reproducibility.

The project is designed as a **research-grade engineering portfolio** for graduate-school applications: no look-ahead (PIT), leakage-resistant validation (walk-forward + embargo), multiple-testing correction (DSR), fail-closed governance, and honest out-of-sample reporting.

### ✦ Core Principles

| Principle | Meaning |
|---|---|
| Point-in-Time | News, forum, price and labels use only information published by then |
| Risk-first | Cash buffer, per-name/sector caps, turnover and liquidity constraints |
| Net-of-cost | Commissions, fees, stamp tax, slippage and ETF fees are all modeled |
| Fail-closed | Failed audits return `HOLD_FOR_REVIEW`; Phase 11 degrades to `OBSERVATION_ONLY` |
| Auditable | Every phase report carries Git hash, runtime, summary and SHA-256 |

### ✦ 11-Phase Pipeline

| Phase | Function | Key outputs |
|---|---|---|
| 1 | Survivorship-aware universe, liquidity, capacity | `assets`, `adv_data`, survival matrix, AUM cap |
| 2 | Train/val/test slices, cross-market calendars, embargo | isolation evidence, no window overlap |
| 3 | PIT features, regime, news/forum sentiment | `feature_panel_*`, `alternative_signals` |
| 4 | Direction/label, event dedup, sample weights | `y_clf_all`, `y_reg_all`, `sample_weights` |
| 5 | Direction & quantile models, calibration | model, features, prediction intervals |
| 6 | Black–Litterman, robust cov, convex allocation | target weights, ADV20 constraints |
| 7 | Execution FSM, board lots, halts, fees | NAV, violations, cost ledger |
| 8 | DSR/coverage/capacity/stress audits | `audit_summary`, `audit_passed` |
| 9 | Target/executed reconciliation, drift | MAE, PSI, crowding gates |
| 10 | CIO bilingual governance | `cio_decision`, evidence list |
| 11 | Advisory outputs | bilingual reports; observation-only if gated |

### ✦ Architecture

```mermaid
flowchart LR
    A[Phase 1-2<br/>Data & slices] --> B[Phase 3-4<br/>PIT features & labels]
    B --> C[Phase 5<br/>ML models]
    C --> D[Phase 6<br/>Convex allocation]
    D --> E[Phase 7<br/>FSM backtest]
    E --> F[Phase 8-9<br/>Audit / stress / reconcile]
    F --> G[Phase 10-11<br/>CIO governance & advisory]
    G -->|HOLD_FOR_REVIEW / OBSERVATION_ONLY| A
```

### ✦ Adaptive Rotation Engine

Close-signal → next-open execution with no look-ahead, layered on multi-factor scoring with **adaptive risk control**:

| Mechanism | Description |
|---|---|
| Continuous ML exposure | logit `P(next-month up)` smoothly maps exposure instead of a binary cash veto |
| Safe-asset rotation | defensive states hold the strongest-rising treasury/gold/money ETF (120d momentum + 20d trend gate) |
| Event-shock & euphoria filters | crash days and >10% 20-day spikes rotate into safe assets |
| Small-profit harvesting | 6% take-profit / 8% stop-loss repeatedly harvests small gains |
| 3-month rotation cap | hard max holding of 63 trading days, no minimum |
| Leverage | disabled end-to-end (personal capital, no debt): `confirm_leverage=1.0`, `max_gross_exposure=1.0`, no shorting |

### ✦ Backtest Performance (2016-01 ~ 2026-08, 2,578 trading days, honest setup: 100k CNY, full-pool PIT revalidated production defaults)

| Metric | Honest backtest (PIT universe 5,478 names, 100% coverage, monthly, explicit fees, board lots, no leverage) | Target / benchmark |
|---|---|---|
| Annual return | **6.54%** (OOS 2022+: 8.35%) | CSI300 (510300) ~5.3% ✅ |
| Sharpe | **1.01** (OOS 1.31) | ≥0.9 ✅ |
| Calmar | **0.64** (OOS 0.79) | ≥1.2 ❌ structurally capped, see plan docs |
| Max drawdown | **-10.29%** | CSI300 ~-44.8% ✅ |
| Recovery (last-3y window) | **181 trading days** | ≤126 ❌ (ongoing 2026 drawdown) |
| Quarterly win vs equal-weight | **51.2%** | ≥50% ✅ |
| Quarterly win vs CSI 300 | 53.5% | ≥60% ❌ disclosed |
| Cumulative cost | 10.6% (93 trades) | ✅ |
| Leverage | 0.00x (disabled) | ✅ |
| Average gross exposure | ~73% | — |

> **2026-08-15 full-pool PIT revalidation milestone (production defaults adopted)**: with network restored, the full daily-bar universe was rebuilt from the sina klc full-history channel (Eastmoney/Tencent/baostock were rate-limited or bot-blocked; alternatives were found as authorized) - 5,478/5,478 (100% coverage, 248/248 delisted), cninfo dividends 5,420 names / 50,947 rows. Full-pool revalidation (106 registered trials, DSR multiple-testing): **`--profile robust` (fixed 3% + z-score 3.0 crash detection, 8% stop) 6.54%/1.01/-10.29% (OOS 8.35%/1.31, DSR p=3.4e-06 passes)** beats the old production default on every metric (4.79%/0.73/-13.40%, OOS 4.80%/0.71), winning 4/4 warm walk-forward folds by annual return (2024-2026 fold: 13.52% vs 8.62%, MDD -8.6% vs -13.4%). **Adopted as the production default**; the old parameters remain available as `--profile legacy` for A/B. Note: the fresh data reproduces the old default at 4.79% (data-source differences vs the 2026-08-11 documented 6.25%); the revalidated full-pool figures are canonical.

> **2026-08-14 offline evaluation round** (see `Quant-4/update plans/8-14-offline-evaluation-round1.md`): engine evaluation and risk-control fixes on the 421-name offline cache subset — (1) fixed a real defect: an event-shock risk-off pending was overwritten by the regular rebalance on rebalance days (risk precedence, active by default); (2) added default-off mechanisms: `rebalance_min_turnover` (minimum-turnover rebalance band), `event_shock_zscore` (volatility-adaptive z-score crash detection; the fixed 2.5% threshold fired 158 times on the volatile small-cap subset) and `trailing_stop_pct` (peak-based trailing stop; it doubled drawdown on this subset, kept as opt-in); (3) removed dead code in `rank_candidates`/`weekly_rotation` (byte-compatible); (4) fixed stale/self-contradicting report text (now derived from the summary); (5) **accounting self-consistency audit**: equity compounding, no-leverage exposure bounds and clean-day drift identities hold exactly; the stop path charges cost without turnover (documented design); (6) extra factors (rsi14/idll20/MAX/Amihud) all underperform the composite base — the factor set is near-optimal on this universe; (7) sleeve 40/30/20/10 portfolio with `robust` base: annual 1.14% → 1.90%; (8) end-to-end alternative-signal wiring verification: governance PASS and correct directional response (momentum-proxy signal OOS 9.96%/0.92, reversal-proxy −27.9% MDD — a wiring test, not new alpha); (9) the open-source ranking gains caveated subset rows (subset baseline → robust → alt composite percentile 0.31 → 0.42 → 0.50); (10) hardened the test environment — **170/170 tests green**; (11) parameter neighborhood stability; (12) capital-sensitivity disclosure; (13) **2026-08-15 full-pool revalidation & adoption**: with network restored, the full pool was rebuilt via alternative sources (sina klc full history + cninfo dividends; 5,478/5,478 = 100% coverage, 248/248 delisted), the robust profile was revalidated on the honest full pool and **adopted as the production default** (6.54%/1.01/-10.29%, OOS 8.35%/1.31, DSR p=3.4e-06 at 106 trials, 4/4 warm folds); the old parameters remain as `--profile legacy`; (14) **173/173 tests green**.

![Equity curve & drawdown](docs/images/weekly_rotation_equity.png)

![Monthly return heatmap](docs/images/weekly_rotation_monthly_heatmap.png)

> Criterion note: the recovery gate uses a rolling last-3-year window (peak → new high, max below-peak streak); the full-window value is disclosed alongside. OOS = samples after 2022-01.

### ✦ Strategy-Selection Layer & Open-Source Ranking (gate designed 2026-08-10, re-run on adopted base 2026-08-15)

> **2026-08-16 real-data revalidation round (Round 16-18)** (see `Quant-4/update plans/8-16-real-news-fullpool-revalidation.md`): network diagnosis proved the earlier "no network" reports were **false negatives** — PowerShell/curl go through the Windows Schannel layer, which the sandbox restricts (`SEC_E_NO_CREDENTIALS`), while the **Python/OpenSSL path works** (baidu/sina/pypi and all project data-source hosts TLS/HTTP 200). New tool `Quant-4/tools/fetch_real_news_sina.py` pulled **12,066 real news items for all 38 production-pool names** (full contract columns, `is_synthetic=0`, sha256 into the immutable cache). Real-signal A/B on the adopted base: the pathway's directional response is correct, but on the full 38-name pool the direction FLIPS vs the 20-name subset (on 20, positive sentiment dragged and inverted improved; on 38, the real signal slightly improves the window -3.89%→-3.07% while inverted is worse -3.94%) with all full-window/OOS differences within ±0.05pp — **real news sentiment has no stable directional alpha** (the R16 small-sample direction was coincidental and is not generalizable); the free source covers only ~5 months at ~28% coverage, so full-history pool revalidation is data-constrained; `alternative_signal_weight` stays 0. **175/175 tests green**.

> **2026-08-15 follow-up round (Round 10-13)** (see `Quant-4/update plans/8-15-selector-gate-rerun-recovery-iteration.md`, `Quant-4/update plans/8-15-alt-signal-fullpool-wiring.md`, `Quant-4/update plans/8-15-fullpool-sleeve-revalidation.md`, `Quant-4/update plans/8-15-walkforward-loader-fix-sleeve-folds.md`): (1) **strategy-selector evidence gate re-run on the full pool under the adopted defaults**: selector OOS Sharpe 1.28 < best fixed momentum 1.37 (OOS ann 8.20% vs 8.64%) → **stays disabled** (`strategy_selector=""`); honestly disclosed that the selector has the best full-window Calmar 0.75 / MDD -8.57% of all archetypes, but the pre-registered OOS gate fails and the gate was not relaxed; (2) **recovery-period mechanism iteration (14 full-pool configs) all negative**: `daily_regime_monitoring` much worse (5.01%/0.74/-15.53%), `rebalance_min_turnover`/`drawdown_guard`/`ml_bear_floor`/`defensive_hold_safe_frac`/ML exposure-band endpoints (`ml_bear_low`/`ml_bull_high`) no material gain or worse — the 181-day recovery was traced to structural lag (strategy -7.16% vs benchmark -9.52% during the 2026 drawdown; recovery waits for the benchmark to reclaim MA40, which only happened on the last data day); production defaults byte-unchanged; (3) **full-pool alternative-signal pathway wiring re-verification**: a sentiment-shaped synthetic PIT panel (event-spike + decay, not a momentum proxy) passes the governance gate on the 5,478-name adopted base with exact as-of and correct directional response, but hurts at every weight (5.30-5.40% vs 6.54%; OOS 6.89-7.81% vs 8.35%) and the inverted signal is ≈ neutral — a wiring test, not new alpha, consistent with R4; real news/forum full-pool revalidation was then blocked by no network / no cached news data (later resolved in R16); `alternative_signal_weight` stays 0; (4) **full-pool sleeve revalidation (R3.5/R9 pending item)**: the 40/30/20/10 sleeve portfolio on the adopted full pool earns its first honest evidence — Sharpe 1.01→**1.09** (OOS 1.31→**1.35**), MDD -10.29%→**-9.91%** (the research gate's `max_drawdown` flips from HOLD to **PASS**), monthly win rate 57.8%→**60.2%**; walk-forward shows 3/4 Sharpe & Calmar wins but return give-up in the hot 2024-26 fold (12.24% vs 14.49%) — the sleeve is a risk-smoothing opt-in layer, not strictly dominant; the single-book production default stays, sleeve evidence recorded honestly; (5) **walk-forward loader consistency fix**: R9's fold script used the scratch loader (5,503 names incl. 25 B-shares); re-run with the canonical PIT loader (5,478) confirms the verdict unchanged (robust 4/4 by ann, 3/4 by Sharpe/Calmar) with the 2024-26 fold corrected to **14.49%**; (6) open-source percentiles corrected to the adopted-default recomputation **0.74/0.85** (0.86/0.92 was the old-default figure); **174/174 tests green**.

- The engine is decomposed with an optional **strategy-selection layer** (`Quant-4/Main/strategy_selector.py`: balanced / momentum / defensive / safe / sprint archetypes + hysteresis), gated on out-of-sample evidence. **Re-run on the full PIT pool under the adopted production defaults (2026-08-15)**: the selector's OOS Sharpe 1.28 is below the best fixed archetype (momentum 1.37; OOS ann 8.20% vs 8.64%), so **production keeps it disabled** (`strategy_selector=""`) to avoid overfitting for adaptation's sake. Honest disclosure: the selector's full-window Calmar 0.75 / MDD -8.57% is the best of all archetypes (hysteresis does smooth the risk curve), but the pre-registered OOS gate (must beat the best fixed archetype on both OOS Sharpe and OOS ann) still fails — recorded, gate not relaxed.
- The 2026-08-11 parameter grid (monthly + persistence + stop band) lifted the honest result to **6.35%/0.99**; earlier "safe 7.71%" style numbers were artifacts of the sparse dividend panel collapsing the candidate pool and were retracted.
- Versus 16 open-source non-HFT quant strategies (`Quant-4/benchmark/open_source_comparison.py`, recomputed on the adopted defaults 2026-08-15): composite percentile **0.74** full-window / **0.85** OOS (≥0.70 = top 30% ✅; the 2026-08-10 0.86/0.92 figures were on the old production defaults and were corrected after the adoption recomputation); drawdown percentile 0.85.

### ✦ Quick Start

```powershell
.\setup.ps1
python Quant-4\Main\main.py --force-recompute --non-interactive
python Quant-4\run_weekly_rotation.py
python Quant-4\run_adaptive_backtest.py 600519 --kind stock --years 8
python -m pytest Quant-4\tests -q
python Quant-4\research\run_performance_gate.py
```

For a content-addressed reproducibility proof of the frozen external-market experiment:

```powershell
python Quant-4\research\run_reproducibility_smoke.py --download
```

This records raw and cleaned dataset hashes, source/licence/coverage lineage, the Git and Python environment,
all installed package versions, explicit random seeds, numeric tolerances, and output hashes. Python 3.12 setup
uses `Quant-4/requirements-lock-py312.txt`; generated caches and evidence remain local and untracked.

Governed Markdown reports can be generated from a reviewed JSON specification with:

```powershell
python Quant-4\research\generate_standard_report.py spec.json report.md
```

CI runs the full suite on Windows and Linux with Python 3.11/3.12, plus Ruff, mypy, an 80% coverage gate for
research-governance modules, Hypothesis financial-invariant properties, and deterministic full-pool/alternative-data
performance budgets. These gates detect engineering and financial-logic regressions; they do not establish investment validity.

See **[UserGuide.md](UserGuide.md)** for the full bilingual manual.
Formal research methods and governed reports are indexed in **[docs/research/README.md](docs/research/README.md)**.

### ✦ Validation & Governance

- PIT / NAV / ADV20 / PSI / DSR / Embargo — see glossary in the user guide.
- Any failed audit returns `HOLD_FOR_REVIEW`; Phase 11 outputs observation-only advice when gates fail.
- Unit tests validate interfaces and rules on test samples; they do not prove third-party data veracity or future profitability.

---

**License / Disclaimer** — For research and education. No investment advice. Past performance does not guarantee future results. Data sources are third-party and may be rate-limited or change without notice.
