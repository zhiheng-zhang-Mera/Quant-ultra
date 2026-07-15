# Phase-02 全向修改计划书：时序切片与前置平稳性/自相关性审计

## 1. 现存系统缺陷剖析（尖锐审计）
* **时序泄漏（Look-Ahead Leakage）**：当前在进行时序切片和数据归一化（Normalization）时，部分全局参数（如全局均值、全局方差）被误用于局部切片，导致未来的信息通过标准化因子隐蔽地向历史样本渗透。
* **ACF 孤立验证**：目前虽然有 `acf_analyzer.py`，但它纯粹是一个“只读”的可视化工具，分析结果并未与特征工程、模型训练产生硬性的拦截挂钩。
* **时间相关性混淆**：在进行 K-Fold 交叉验证时使用随机划分，彻底破坏了时序依赖性，使得测试集与训练集高度相关。

## 2. 全向改造计划与设计架构
### A. 无泄漏滚动窗口归一化 (Online Rolling Scaler)
* 强制要求所有标准化操作（Z-Score, MinMax）必须在**滚动历史窗口（Lookback Window）**内计算，严禁使用全局未来数据。
* 重新编写 `step2_1_slicing.py`，在构建 DataSlicing 时，每个 slicing step 只能访问 $t$ 时刻之前的均值 $\mu_{t}$ 和标准差 $\sigma_{t}$。

### B. 时序验证前置硬拦截 (Stationarity Hard-Gate)
* 改造 `acf_analyzer.py`。在特征进入 Phase-5 训练前，必须对所有时序特征进行 **ADF (Augmented Dickey-Fuller) 检验** 和 **KPSS 检验**。
* 定义拦截机制：如果某特征在 95% 置信度下不满足平稳性，自动对其进行**分数阶微分 (Fractional Differentiation)** 处理，在保留记忆（Memory）的同时确保平稳性。
* 计算时序特征的 **ACF / PACF 衰减指数**，若自相关性衰减过慢（表明存在单位根或长期记忆），自动触发一阶差分或剔除特征。

### C. 时序隔离区机制 (Purged & Embargo CV)
* 为防止交叉验证中的信息泄露，实现 Marcos Lopez de Prado 的 **Purged and Embargo** 算法。
* **Purging (清除)**：从训练集中删除其标签在测试集时间跨度内的样本。
* **Embargo (禁运)**：由于自相关性的存在，在测试集之后的一段窗口内（如日频交易中的次日到一周）的训练样本必须被剔除。

## 3. 具体修改步骤与代码骨架
1. 在 `Phase_2/config.py` 中加入 `EMBARGO_PCT = 0.01` 及平稳性检验严格度。
2. 重构 `step2_2_validation.py`，引入 `purged_k_fold` 交叉验证分割器。
3. 在 `acf_analyzer.py` 中添加平稳性自动微分修复逻辑。
