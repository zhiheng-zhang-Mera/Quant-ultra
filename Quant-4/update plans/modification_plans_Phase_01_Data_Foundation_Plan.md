# Phase-01 全向修改计划书：数据基础与非结构化替代数据清洗升级

## 1. 现存系统缺陷剖析（尖锐审计）
* **生存偏差未做防护**：当前筛选逻辑中对历史已退市、暂停上市的标的缺乏动态库追踪，容易导致“幸存者偏差”，大幅虚增历史回测夏普。
* **线性填充的无脑化**：对于缺失值的处理，目前主要依赖简单的前向填充（Forward Fill）或均值填充，这在日级时序数据中会严重扭曲真实的量价波动，引入虚假的低频平稳信号。
* **信息同质化拥挤**：纯量价因子的拥挤度极高，日级交易中缺乏中高频信号的对冲，极易在市场风格突变时产生共振回撤。

## 2. 全向改造计划与设计架构
### A. 动态生存偏差纠正引擎 (Survival Bias Eliminator)
* 引入动态成分股跟踪机制。不使用当前静态成分股，而是根据历史每个时间节点的实际指数成分股和上市交易状态进行动态过滤。
* 重新编写 `step1_1_screening.py` 中的筛选逻辑，引入 `Delisted_Security_Tracker`，确保回测中包含历史退市股票的完整生命周期。

### B. 非线性时序插补与低秩矩阵完成 (MICE & Matrix Completion)
* 弃用无脑 Forward Fill。对于缺失值（如由于停牌、交易异常中断导致的缺口），引入 **MICE (Multivariate Imputation by Chained Equations)** 或 **奇异值阈值算法 (SVT)** 进行多变量低秩重构。
* 在 `step1_data_foundation.py` 中重构数据补全模块，针对高维股票截面，利用截面相关性对个股缺失量价进行联合估计，确保协方差矩阵不因“断点”而失真。

### C. 盘后非结构化替代数据抽取（Alternative Data Ingestion）
* 新增非结构化舆情与财报情绪特征抓取子模块。
* 利用本地部署的轻量级大模型（如 `Qwen-2.5-Coder-7B`），在盘后 16:00 自动解析当日财报电话会议文本（Earnings Call Transcripts）、管理层讨论与分析（MD&A）及行业研报。
* 输出高信息浓度的替代特征：
  * `sentiment_score` (情绪得分)
  * `tonal_shift` (语调偏移度，当前财报对比上一财报的措辞变化)
  * `regulatory_risk_density` (合规风险密度词频)

## 3. 具体修改步骤与代码骨架
1. 修改 `step1_1_screening.py`，加入动态生存股库。
2. 重构 `step1_data_foundation.py`，提供 `mice_imputation` 函数替代 `fillna(method='ffill')`。
3. 新增子脚本 `alternative_extractor.py`，实现盘后自动化文本情感量化因子的入库。
