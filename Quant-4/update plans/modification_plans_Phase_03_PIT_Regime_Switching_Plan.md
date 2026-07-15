# Phase-03 全向修改计划书：PIT时点对齐与非平稳市场机制划分

## 1. 现存系统缺陷剖析（尖锐审计）
* **PIT 幻觉（Look-Ahead Bias via Fundamentals）**：基础财务报表数据（如 EPS, PE）的发布日期（Publication Date）与财报所属期末（Fiscal Period End）在时间上存在严重的不对称。直接在财报截止日使用这些数据回测会导致致命的“未来眼”。
* **宏观机制的无知（Regime Blindness）**：模型在训练时不加区分地将牛市、熊市和震荡市的数据混合。由于不同机制下特征分布差异巨大，导致模型学出了一套“无所适从”的平均参数，实盘极易被反复两边打脸。

## 2. 全向改造计划与设计架构
### A. PIT（Point-In-Time）时点硬对齐引擎 (Point-In-Time Alignment Engine)
* 建立严格的双时间轴（Bi-temporal）数据库结构。一条轴是**财报所属期（As-of Date）**，另一条轴是**公开披露期（Publication Date/Availability Date）**。
* 重新封装 `data_loader.py`，确保在任何回测时间 $T$，系统只能加载在 $T$ 之前已经真实公开披露的财报数据。严禁在 12 月 31 日读取尚未公布的年报数据。

### B. 隐马尔可夫模型机制划分器 (HMM Regime Classifier)
* 引入 **HMM (Hidden Markov Model)** 或是 **GMM (Gaussian Mixture Model)** 对市场环境进行无监督分类。
* 选取高频波动率、长短周期均线利差、市场交易量变化率作为状态特征，自动将市场划分为：
  1. 高波熊市 (High Volatility Bear)
  2. 低波牛市 (Low Volatility Bull)
  3. 震荡盘整 (Mean-Reverting Sideways)
* 市场机制（Regime State）将作为一个动态的时序因子，直接注入到后置的模型特征中，或者作为不同子模型条件训练的路由依据。

## 3. 具体修改步骤与代码骨架
1. 修改 `data_loader.py`，实现 `get_pit_fundamental_data` 接口。
2. 重构 `step_3_1_regime.py`，使用 `hmmlearn` 库建立 HMM 状态分类器，并在每日收盘后更新市场机制分类。
3. 在 `step_3_4_features.py` 中，将当前的 `regime_state` 以 One-Hot 编码的形式合并至特征截面上。
