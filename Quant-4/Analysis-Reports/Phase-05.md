# Phase_5_Model_Training_Calibration：模型训练与概率校准深度解析

## 概述

Phase_5（Model Training & Calibration）是 Quant‑4 系统的**核心机器学习与因果推断层**，负责将 Phase_4 产生的标签（Labels）与 Phase_3 的特征（Features）进行深度融合，训练出可解释、稳健且经过概率校准的预测模型。该阶段实现了从“特征工程”到“可交易信号”的质的飞跃，其架构围绕三个核心支柱展开：（1）**长记忆特征工程**（通过分数阶微分保留时间序列记忆）；（2）**联邦净化交叉验证**（防止信息泄露与负迁移）；（3）**共形预测与级联校准**（输出具有统计覆盖率保证的置信区间）。

Phase_5 严格遵循 Quant‑4 系统手册中“阶段五”的要求，实现了分数阶微分（`fractional_diff_series`）、基于 MMD（最大均值差异）的域自适应权重、以及不依赖参数分布假设的 CQR（Conformal Quantile Regression）误差校准。其设计极具学术前沿性与工程防御性，是系统 Alpha 生成的智慧中枢。

---

## 模块总览与协作关系

| 文件 | 核心职责 | 依赖关系 |
|------|----------|----------|
| `__init__.py` | 暴露 `execute` 入口 | 被 `main_engine.py` 调用 |
| `config.py` | 定义 LightGBM 基础参数、分数阶搜索空间（d*）、VIF 阈值、共形推断网格 | 被所有子模块引用 |
| `maths_utils.py` | 提供 FFT 分数阶卷积算子与白盒特征计算（动量、GK 波动率） | 被特征生成和 CV 模块引用 |
| `dataset_utils.py` | 基于切片与存活矩阵构建训练/验证的展平数据集（X, y） | 被 CV 和拟合模块引用 |
| `step_5_1_cv.py` | 净化递进式交叉验证，搜索最优分数阶微分阶数 d*，监控负迁移 | 依赖 `maths_utils` 和 `dataset_utils` |
| `step_5_2_3_features.py` | 构建三维特征立方体（日期×资产×特征），执行 VIF 与层次聚类特征过滤 | 依赖 `maths_utils` 和 `dataset_utils` |
| `step_5_4_fitting.py` | 拟合 LGBM 方向分类器与三分位数回归模型，包含 MMD 域自适应加权 | 依赖 `dataset_utils` |
| `step_5_5_calibration.py` | 优化决策阈值 gamma*，基于 Train-B2 计算 CQR 个体误差安全垫 | 依赖 `maths_utils` |
| `step5_model_training_calibration.py` | 主控编排器，实现切片缺失自愈，串联 5.1~5.5 子步骤 | 聚合所有子模块 |

---

## 各模块详解

### 1. `config.py` —— 超参数与搜索空间定义

**功能**  
定义了 Phase_5 所需的关键超参数：
- `BASE_LGB_PARAMS`：LightGBM 的确定性运行配置（`n_estimators=100`, `num_leaves=31`, `deterministic=True`, `num_threads=1`），确保实验结果可复现。
- `D_MIN_SEARCH_SPACE`：分数阶微分阶数 d* 的搜索网格 `[0.1, 0.3, 0.5, 0.7, 0.9]`。
- `CV_FOLDS = 3`：向前走验证的折叠数。
- `VIF_THRESHOLD = 30.0`：方差膨胀因子阈值，用于剔除多重共线性特征。
- `CLUSTER_SELECT_RATIO = 0.8`：特征层次聚类的累积重要性保留比例。
- `GAMMA_GRID`：级联决策阈值 gamma 的搜索网格（0.3 到 0.7 线性插值 9 个点）。
- `ERROR_THRESHOLD_WINDOW = 252`：共形误差计算的回望窗口（约 1 年交易日）。
- `TAU_BL = 0.02`：Black-Litterman 融合的先验置信度参数。
- 负迁移熔断参数：`NEGATIVE_TRANSFER_PATIENCE = 3`, `MMD_ALPHA = 0.1`。

**设计亮点**  
- `deterministic=True` 和 `num_threads=1` 强制消除了并行计算的随机性，保障金融建模中极其重要的结果可复现性。
- 分数阶微分搜索空间的设定覆盖了从“接近整数阶差分”（0.1）到“长记忆保留”（0.9）的广泛区间。

**潜在缺陷**  
- `GAMMA_GRID` 虽已定义，但在 `step_5_5_calibration.py` 中并未真正用于基于数据效用的搜索（而是优化了一个固定二次函数），导致该配置形同虚设（详见下文分析）。
- 所有参数硬编码于 Python 文件，未支持从外部 YAML 动态覆写，降低了调参灵活性。

---

### 2. `maths_utils.py` —— 高性能信号卷积与特征物理引擎

**功能**  
- **`fractional_diff_series`**：使用 `scipy.signal.fftconvolve` 实现分数阶微分的快速傅里叶卷积计算，将传统递推的 O(n²) 时间复杂度压至 O(n log n)。通过截取卷积结果的 `mode='full'` 前 n 项，完美确保了**因果性**（不引入未来信息）。
- **`compute_whitebox_features`**：为单只资产计算 5 维白盒特征：
  1. 对数收益率（Log Return）
  2. 5 日动量（Momentum 5D）
  3. 20 日动量（Momentum 20D）
  4. Garman-Klass 波动率（基于 OHLC 的无偏估计）
  5. 成交额偏离度（相对 20 日均值的冲击）

**设计哲学**  
- 分数阶微分是金融量化的前沿技术，相比整数阶差分（如 I(1) 平稳化），能更好地保留时间序列的长记忆性，同时达到弱平稳状态。
- FFT 卷积实现是高性能计算的典型应用。

**优点**  
- 代码极度简洁且高效，注释清晰。
- Garman-Klass 波动率利用 OHLC 信息，比单纯 Close-to-Close 波动率更精确，且方差项中的 `2 * log(2) - 1` 系数校正了日内跳跃偏差。

**客观缺陷**  
- `fractional_diff_series` 中的权重迭代 `weights[k] = -weights[k-1] * (d - k + 1) / k` 基于二项式展开，对于非平稳序列（d > 0.5）可能产生较大截断误差，且未对序列进行去均值处理，可能导致低频趋势残存。
- `compute_whitebox_features` 中的动量和成交额冲击均使用循环（`for t in range(...)`），在资产数极大且时间序列极长时效率不高，可考虑使用 `pandas` 的 `rolling().sum()` 向量化替代。

---

### 3. `dataset_utils.py` —— 自适应横截面面板数据摄入

**功能**  
- **`build_partition_dataset`**：根据指定的分区名称（如 `Train-A`），从三维特征立方体 `fractional_features_cube` 和存活矩阵 `alive_mask_matrix` 中提取该分区的展平数据集。
- 遍历资产轴和时间轴，利用 `y_clf_all` 和 `y_reg_all` 字典获取标签。
- 仅保留存活（`alive_mask` 为 True）且有标签的样本，并确保每个资产至少有 10 个有效样本才纳入。

**设计亮点**  
- 多键回退查找（`(dt, sym)`, `(dt_str, sym)`, `(pd.Timestamp(dt), sym)`），极大增强了对抗时区类型不一致的鲁棒性。
- 资产级最低样本数门槛（`len(y_c_list) >= 10`）防止了冷门资产因数据稀疏导致训练过拟合。

**客观缺陷**  
- 该函数返回展平的 `np.ndarray`（样本数 × 特征维度），丢失了资产与时间的索引信息，不利于后续的资产级个性化校准（尽管 Phase_5 后续通过字典弥补了这一点）。
- 三层嵌套循环（日期 → 资产 → 标签查找）在资产 > 3000、日期 > 2000 时性能极差。应利用 `DataFrame` 的 `merge` 或 `MultiIndex` 切片进行向量化替代。

---

### 4. `step_5_1_cv.py` —— 净化向前走验证与负迁移熔断

**功能**  
- 对 `D_MIN_SEARCH_SPACE` 中的每个 d 值执行 3-Fold 向前走验证（Purged Walk-Forward）。
- 在每个折叠内：
  - 对原始 5 维白盒特征应用分数阶微分，构建微分特征矩阵。
  - 使用 `StandardScaler` 标准化。
  - 训练基础 LGBM 分类器（A 股本土模型）和模拟迁移模型（注入 5% 高斯噪声模拟跨境特征扰动）。
  - 比较迁移损失与基础损失，若迁移损失连续超过基础损失达到 `patience` 次数，则触发负迁移熔断（`monitor["triggered_melt"] = True`）。
- 选择交叉验证准确率最高的 d 作为 `best_d`。

**设计哲学**  
- “负迁移熔断”机制极具创新性，它模拟了在引入美股或其他市场特征时，若这些特征损害了 A 股模型的表现，系统将自动回退到纯本土模型，防止多市场联合训练中的“1+1 < 1”现象。

**优点**  
- 严格的时间序列隔离（`tr_idx` 和 `val_idx` 不重叠），符合 PIT 原则。
- 动态响应机制（`negative_transfer_monitor`）能在运行时决定是否采纳跨域知识。

**客观缺陷**  
- **性能瓶颈**：对每个 d，都要遍历所有资产并重新计算分数阶微分，在资产数千、日期数千时，计算量极为庞大。虽然在 `cv_timeline` 中做了时间降采样（`step`），但资产维度未降维。
- **模拟迁移噪声不合理**：`X_tr_tuned = X_tr_scaled * (1.0 + 0.05 * np.random.normal(...))` 产生的是随机噪声，而非真实的美股特征分布，这使得“负迁移检测”失去了统计学意义——模型因为随机噪声而性能下降，属于预期之内，并不能证明真实跨域特征有害。
- `best_scaler` 始终为 `None`（未在循环中正确赋值），导致后续特征生成时 scaler 缺失，虽然后续 `step_5_4` 会重新拟合 scaler，但造成冗余计算。

---

### 5. `step_5_2_3_features.py` —— 分数阶立方体生成与共线性净化

**功能**  
- **`generate_fractional_features`**：
  - 基于 `best_d`，对所有资产的全量历史数据应用 `fractional_diff_series`，生成形状为 `(T, N, F)` 的三维特征立方体。
  - 利用 Phase_1 的 `alive_mask` 填充存活矩阵 `alive_mask_matrix`，确保非上市日期被标记为不可用。
- **`run_feature_filtering`**：
  - 在 Train-A 数据集上计算 VIF（方差膨胀因子），剔除共线性过高的特征（阈值 30）。
  - 使用 `FeatureAgglomeration`（层次特征聚类）和 PCA 重要性排序，在每个聚类内按累积重要性 80% 选择代表性特征。
  - 输出 `selected_features` 索引列表。

**设计亮点**  
- 特征选择流程严谨：先 VIF 剔除共线黑洞，再聚类降维保留信息，避免了手工特征筛选的主观性。
- `alive_mask_matrix` 的同步填充确保了后续数据摄入不会误用退市或未上市标的。

**客观缺陷**  
- VIF 计算在特征数极少（F=5）时意义有限，且若 `X_scaled` 存在零方差列（特征为常数），`variance_inflation_factor` 会直接报错，虽被 `try-except` 捕获但静默跳过可能导致严重共线性特征未被处理。
- 特征聚类选择使用了 `n_clusters=min(3, len(keep_idx))`，硬编码 3 个簇，若特征空间本身只有 2 个正交方向，强行聚类为 3 会引入随机分裂。

---

### 6. `step_5_4_fitting.py` —— 三阶段联邦优化与 Log-Odds 核拟合

**功能**  
- **字典索引升维**：为了应对时区类型混乱，将 `y_clf_all` 和 `y_reg_all` 的键扩展为 `(Timestamp)`, `(date)`, `(str)` 三种格式，实现“无死角”查找。
- **核心修复（Fallback 引擎）**：若 `build_partition_dataset` 返回的数据为空（因时区污染或类型不匹配），启动“自主重构引擎”——直接穿透 `fractional_features_cube` 和 `trading_days_dt` 强行组装 X 和 y，以物理级别夺回数据血缘。
- **MMD 域自适应权重**：若存在美股私有特征面板（`feature_panel_private_us`），计算 A 股样本与美股样本在 RKHS（再生核希尔伯特空间）中的分布差异，生成重要性权重 `mmd_weights`，用以在训练时拉齐分布。
- **模型拟合**：
  - 计算初始边际对数几率（`init_score`）作为 LGBM 的初始预测值。
  - 拟合 `LGBMClassifier`（3 分类，对应 0/中性, 1/多头）。
  - 拟合 3 个分位数回归模型（0.025, 0.5, 0.975），用于共形预测区间。

**设计亮点**  
- **Fallback 引擎**是防御性编程的巅峰之作——它彻底解耦了 Phase_2 切片格式与 Phase_3 数据结构之间的依赖，即使上游数据传递出现格式偏差，仍能保证模型训练不中断。
- **MMD 权重**是迁移学习在量化中的优雅应用，通过匹配一阶和二阶矩（核方法），让模型更关注分布相似的部分，减少域偏移影响。
- `init_score` 的设定利用了先验类别分布，加速了 LGBM 的收敛。

**客观缺陷**  
- **MMD 权重计算效率低**：`_compute_rbf_mmd_weights` 中计算了 `(n_s + n_t) x (n_s + n_t)` 的成对距离矩阵，当样本数 > 50000 时内存爆炸。
- 分位数回归模型使用了与分类器相同的 `BASE_LGB_PARAMS`（`num_class` 未移除），但 LGBM 的 `quantile` objective 不应有 `num_class` 参数，代码中通过 `set_params(objective='quantile', alpha=q)` 覆盖，虽能运行，但 `num_class` 残留可能引发警告或内部逻辑混乱。
- Fallback 引擎构建的 `X_fallback` 未应用 `selected_features` 过滤，导致后续切片可能与正式流程不一致。

---

### 7. `step_5_5_calibration.py` —— 级联共形校准与阈值优化

**功能**  
- **Gamma 优化（严重缺陷）**：遍历 `gamma_grid`，计算 `simulated_utility = 0.01 * g_test - 0.02 * (g_test ** 2)`，取最大值作为 `best_gamma`。
- **CQR 误差校准**：
  - 在 Train-B2 分区上，对每个资产使用训练好的分位数模型预测 2.5%、50%、97.5% 分位数。
  - 强制单调性修复（`q_low <= q_mid <= q_high`）。
  - 计算经验误差 `error = max(q_low - y_true, y_true - q_high, 0.0)`（即真实值超出预测区间的偏离量）。
  - 对每个资产，取最近 252 个交易日误差的 95% 分位数作为该资产的个性化安全垫 `q_error_threshold_dict[sym]`。

**设计哲学**  
- 共形预测（Conformal Prediction）不依赖误差分布假设，仅凭经验分位数即可保证有限样本覆盖率（约 95%），是金融风险控制的理想工具。

**优点**  
- 资产级个性化误差阈值比全局阈值更精确，能适应不同流动性和波动特征的标的。
- 分位数单调性修复防止了模型输出荒谬的区间（如下限高于上限）。

**客观缺陷**  
- **Gamma 优化逻辑严重错误**：`simulated_utility` 是一个与数据无关的二次函数（`0.01*g - 0.02*g^2`），其极值恒在 `g=0.25` 处取得（求导 `0.01 - 0.04g = 0`）。这意味着无论 `gamma_grid` 如何定义，`best_gamma` 永远最接近 0.25，完全无视了真实数据的验证性能。这导致“级联决策门槛优化”形同虚设。
- CQR 校准仅在 Train-B2 上计算，但 Train-B2 本身也用于其他目的（如早停），虽未直接参与拟合，但理论上仍存在信息泄露风险，应使用完全独立的 Validation 集或 Test 集。

---

### 8. `step5_model_training_calibration.py` —— 主控编排器与自愈调度

**功能**  
- **时空自愈**：若 `slices` 缺失或 `Train-A` 不存在，启动硬编码的日期范围重建（2010-2018 等分段）。这是最后的防御手段。
- 依次调用 `run_walk_forward_cv`, `generate_fractional_features`, `run_feature_filtering`, `fit_model_bundle`, `run_cascade_calibration`。
- 返回包含模型对象、参数和特征立方体的字典。

**设计亮点**  
- 硬编码自愈日期虽显粗暴，但确保了在极端配置丢失情况下流水线仍能运行，避免生产环境崩溃。

**客观缺陷**  
- **硬编码日期炸弹**：自愈逻辑中的日期范围（如 `"2010-01-04"` 至 `"2018-06-25"`）是静态的，未来若系统运行于 2030 年，这些区间将严重过时，且完全忽略了数据实际起止年份，可能引入大量缺失值或空切片。
- 未检查 `pipeline_context` 是否包含 Phase_4 产出的 `y_clf_all` 等关键对象，若 Phase_4 执行失败，Phase_5 的 Fallback 引擎也无法凭空重建标签，最终抛出难以追踪的 KeyError。

---

## 整体评价与改进方向

### 系统优势
- **学术前沿性**：分数阶微分 + MMD 域自适应 + 共形预测的组合在量化交易系统中极为罕见，体现了极高的技术品味。
- **工程健壮性**：Fallback 引擎、多键查找、时区兼容、硬编码自愈等防御机制，使得系统在数据质量不佳时仍可推进。
- **确定性复现**：LGBM 的 `deterministic=True` 和固定随机种子，确保了结果可复现，这是量化研究的生命线。
- **精细的资产级校准**：为每只个股独立计算误差阈值，比行业通用的全局阈值更符合微观结构差异。

### 当前痛点
- **Gamma 优化逻辑名存实亡**：核心的阈值调优并未基于数据效用（如实际 PnL 或 Sharpe），而是优化了一个固定二次函数，导致 `gamma_star` 恒为常数，彻底违背了“级联决策门槛优化”的业务初衷。
- **交叉验证性能瓶颈**：`step_5_1_cv` 中对每个 d 重复进行全量数据加载和 FFT 卷积，在资产规模庞大时极耗时间（可能需要数小时）。
- **负迁移检测失真**：使用随机噪声模拟跨境特征，不能反映真实跨市场特征分布，该检测机制目前无实际防护价值。
- **硬编码日期隐患**：自愈逻辑中的静态日期范围将成为未来维护的致命陷阱。

### 未来演进
- **重构 Gamma 优化**：应在 Train-B1 或 Validation 集上，遍历 `gamma_grid`，模拟真实交易（如设置 `if prob > gamma: long`），计算 Sharpe 或收益风险比作为效用函数，选择最大化效用的 gamma。
- **缓存分数阶结果**：在 CV 阶段，对每个 d 计算一次分数阶特征并缓存于内存，避免在折叠循环中重复计算。
- **真实跨域特征集成**：将 Phase_3 产出的 `feature_panel_private_us` 真实数据（非随机噪声）注入 CV 的迁移实验，使负迁移检测具备实际统计意义。
- **动态自愈日期**：从 `pipeline_context` 中提取实际数据的最早和最晚日期，动态生成切片边界，彻底移除硬编码。
- **引入 GPU 加速**：将 `fractional_diff_series` 和 `MMD` 计算迁移至 CuPy 或 PyTorch，在 GPU 集群上加速大规模矩阵运算。

---

## 结语

Phase_5 是 Quant‑4 系统中数学逻辑最密集、工程防御最复杂的核心模块。其在分数阶长记忆建模和共形推断方面的探索极具启发性，展示了从传统多因子模型向 AI 原生量化框架的演进路径。然而，当前实现中的“伪 Gamma 优化”和“硬编码自愈日期”是两个致命的业务逻辑缺陷，必须在下一次迭代中彻底修复。若能解决上述痛点，Phase_5 将成为一个极具竞争力的工业级机器学习量化引擎。
