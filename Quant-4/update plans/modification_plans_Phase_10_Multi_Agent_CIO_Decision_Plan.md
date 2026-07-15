# Phase-10 全向修改计划书：白盒化多智能体决策层与自适应超参闭环

## 1. 现存系统缺陷剖析（尖锐审计）
* **信息链传输的机械与片面**：上一版的分析师模型为了防止裁判模型上下文过载，将输出完全压缩为极简的 JSON。这导致不同专业视角的分析师之间失去了深度的逻辑推导细节，CIO 只能看到苍白的结论，极易因“信息过度降维”做出错误评判。
* **纯观赏性报告（Static Report）**：大模型只是生成了一篇 PDF 或 Markdown 报告供人类阅读，无法对底层的代码参数进行实质性的**逆向反馈控制**。

## 2. 全向改造计划与设计架构
### A. 双轨汇报机制 (Double-Track Reporting Architecture)
* 分析师模型在输出时采用双轨制：
  1. **极简风控 JSON**：用于确定性的规则拦截与轻量传输。
  2. **高度浓缩逻辑流文本链 (Condensed Logic Chain)**：限制在 150 字以内，保留其推导链条中的核心因果逻辑（例如，Qwen 指出：‘由于20日均线斜率转负且波动率放大，系统性 Beta 存在转弱风险，动量因子失效几率上升。’）。

### B. 3+1 异构分析师与首席裁判矩阵
* **Analyst-1 (Feature & Model Auditor - Qwen-2.5-Coder-7B)**：专项审查 Phase-5 和 Phase-9 的模型特征。聚焦于 SHAP 值漂移、概率校准曲线的畸变程度、Brier Score 的恶化。
* **Analyst-2 (Risk & Stress Auditor - Llama-3.1-8B)**：专项审计 Phase-6 和 Phase-8 的回撤、CVaR 暴露以及极端压力测试表现。
* **Analyst-3 (Execution & Slippage Auditor - Mistral-7B)**：专项审计 Phase-7 的执行滑点、借券费用损耗、调仓频率与双边换手损耗。
* **CIO Judge (Gemma-2-9B / Qwen-2.5-14B)**：收集上述三方的双轨报告，利用原始客观时序指标，撰写最终决策报告。

### C. 闭环超参自动修正回路 (Auto-Parameter Adjust Loop)
* 裁判长（CIO）模型不仅生成文本报告，还必须输出一个格式化的参数调节 JSON 指向底层的量化参数：
  * **超参微调反馈**：若 Analyst-2 报警市场相关性迅速逼近 1，且 CIO 认可该观点，CIO 输出调整指令 `{"HRP_shrinkage_alpha": 0.85, "max_sector_exposure": 0.15}`，直接通过 API 写入底层 Phase-6 的 `config.py`，实现策略的物理自适应抗震。

## 3. 具体修改步骤与代码骨架
1. 重构 `Phase_10/step10_llm_reporting.py`，实现异构分析师模型的串行加载与释放 (`keep_alive=0`)。
2. 扩展 `data_aggregator.py`，支持抓取更详细的模型校准度、滑点比、以及 PSI 指标。
3. 新增 `parameter_effector.py`，专门承接 CIO 模型输出的调参 JSON，动态覆盖各 Phase 的 `config.py`，完成完整的控制论闭环（Cybernetic Feedback Loop）。
