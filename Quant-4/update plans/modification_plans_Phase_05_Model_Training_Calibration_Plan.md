# Phase-05 全向修改计划书：模型滚动集成与严苛概率校准

## 1. 现存系统缺陷剖析（尖锐审计）
* **概率幻觉（Uncalibrated Probabilities）**：类似 XGBoost, LGBM 这样的树模型输出的 `predict_proba` 并非真实的物理概率，而是高度集中于 0 和 1 两端的变形得分。直接将此概率用于后置的头寸配比（如凯利公式或BL融合）会导致头寸分配极端化，造成爆仓风险。
* **时序概念漂移（Concept Drift Over Training）**：使用静态数据集训练出来的模型无法适应非平稳金融市场的时间迁移。

## 2. 全向改造计划与设计架构
### A. 滚动Walk-Forward训练管线 (Walk-Forward Rolling Pipeline)
* 废弃传统的静态 train/test 划分。构建严格的**前向走样（Walk-Forward）滚动交叉验证**架构。
* 设置训练窗口（如 3年滚动），测试窗口（如 3个月），每次前移 3 个月，模型自动增量训练或完全重训。

### B. 多树异构集成与特征选择 (Ensemble Matrix)
* 构建包含 **LightGBM**, **XGBoost**, **CatBoost** 的异构集成矩阵。
* 使用 **SHAP (SHapley Additive exPlanations)** 进行特征贡献度审计。每日盘后剔除 SHAP 值连续 5 天处于噪声水平的弱特征，防止模型在无意义维度上过拟合。

### C. 概率校准算法 (Probability Calibration Engine)
* 为了向后置头寸管理（Phase-6）输出真正具有物理意义的概率（置信度），引入 **Platt Scaling (逻辑回归校准)** 或 **Isotonic Regression (保序回归校准)**。
* 评估校准度：计算并监控 **Brier Score (布莱尔得分)**。
* 在 `step_5_5_calibration.py` 中，对集成的概率输出进行校准，确保当模型给出 70% 胜率时，历史实际获胜比例精确贴合 70%。

## 3. 具体修改步骤与代码骨架
1. 修改 `step_5_1_cv.py`，替换为无泄漏滚动 Walk-Forward 分割器。
2. 重构 `step_5_5_calibration.py`，利用 `sklearn.calibration.CalibratedClassifierCV` 对 LGBM/XGBoost 集成进行校准。
3. 输出经过校准后的概率矩阵 `calibrated_probas.pkl`，供 Phase-6 使用。
