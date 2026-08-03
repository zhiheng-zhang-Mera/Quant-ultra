# Phase-09 全向修改计划书：实盘概念漂移监测与自愈式滚动训练闭环

## 1. 现存系统缺陷剖析（尖锐审计）
* **监控的滞后性与惰性**：目前 `step9_live_mlops.py` 只是定期打印一些系统日志，当实盘特征分布已经发生剧烈偏移、模型吐出完全随机的垃圾信号时，系统无法主动识别。
* **缺乏闭环自愈（No Auto-Feedback Loop）**：监测到漂移后，仅仅发送报警邮件，需要人工干预重训。在非平稳市场中，人工反应的一两天滞后足以引发净值的断崖式回撤。

## 2. 全向改造计划与设计架构
### A. 前置特征统计退化硬核监测 (Population Stability Index Monitor)
* 在每日收盘后，提取当天入库的新样本特征。
* 计算新特征分布与模型训练基准期分布之间的 **PSI (Population Stability Index, 群体稳定性指标)** 以及 进行 **KS 检验 (Kolmogorov-Smirnov Test)**：
  $$PSI = \sum_{i=1}^{k} \left( Actual_i - Expected_i \right) \times \ln\left( \frac{Actual_i}{Expected_i} \right)$$
* **漂移分级警报机制**：
  * $PSI < 0.1$：特征分布极其平稳。
  * $0.1 \le PSI < 0.25$：轻度概念漂移（Concept Drift），发出黄色预警。
  * $PSI \ge 0.25$：严重概念漂移，触发自愈重训逻辑。

### B. 自愈式滚动训练触发引擎 (Triggered Auto-Retraining Loop)
* 当 PSI 连续 3 天超过黄色警戒线，或者任意一天触发红色警报（$PSI > 0.35$）时，系统自动拉起后台守护进程：
  1. 自动截取最新的时序数据。
  2. 启动 Phase-1 到 Phase-5 的增量 Walk-Forward 滚动训练。
  3. 新生成一组模型参数，先放入“影子系统（Shadow System）”进行冷启动验证。

### C. 影子对账与线上无感切流 (Shadow Portfolio Reconciliation)
* 维持两套投资组合：主策略组合（Active Portfolio）和影子策略组合（Shadow Portfolio）。
* 影子组合使用重训后的模型在实盘环境下模拟运行。如果影子组合的各项风控指标（Brier Score, 预测胜率）连续 5 个交易日优于主组合，系统自动执行**无感线上切流（Silent Switch）**，将主模型的推理权交接到重训模型，并向 Phase-10 汇报归档。

## 3. 具体修改步骤与代码骨架
1. 修改 `step9_live_mlops.py`，引入 `FeatureDriftDetector` 计算每日 PSI。
2. 在 `telemetry_alerts.py` 中编写触发重训逻辑。
3. 建立 `shadow_reconciliation.py` 实现两套参数下的推理结果影子比对。

## 4. 2026-08-03 实施核验与补全

- **正式主链已实现**：PSI 分级更新、影子对账和遥测接口。
- **仅旁路原型**：KS、自愈重训和自动切流细节主要位于 `enhance_dlc/phase_09_enhance.py`。
- **未完成**：漂移基准版本、报警去重、影子模型最小观察期、回滚事务和人工审批边界。
- **补全验收**：自动重训可以触发，但资金切流默认必须审批；所有模型切换可原子回滚。
