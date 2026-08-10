# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_5 Hyperparameters & Federated Optimizer Spaces
"""
import numpy as np

# LightGBM 基础确定性运行参数（消除多线程随机噪音污染）
BASE_LGB_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "deterministic": True,
    "num_threads": 1,
    "random_state": 42,
    "verbosity": -1,
}

# 阶段 5.1 联邦长记忆分数阶微分阶数 d* 搜索空间
D_MIN_SEARCH_SPACE = [0.1, 0.3, 0.5, 0.7, 0.9]
CV_FOLDS = 3

# 阶段 5.3 统计学多重共线性特征净化空间参数
VIF_THRESHOLD = 30.0
CLUSTER_SELECT_RATIO = 0.8

# 阶段 5.5 双轨级联决策门槛优化与共形推断标定参数
GAMMA_GRID = np.linspace(0.3, 0.7, 9)
ERROR_THRESHOLD_WINDOW = 252
ERROR_MIN_SAMPLES = 50
TAU_BL = 0.02

# 领域对抗网络与负迁移熔断硬红线哨兵指标
TRANSFER_WEIGHT_INITIAL = 1.0
NEGATIVE_TRANSFER_PATIENCE = 3
MMD_ALPHA = 0.1
TOP_K_GRADIENT_RATIO = 0.1

# 8-9 计划任务四：特征漂移治理（Rolling Z-Score）
# 开启后特征面板按“仅用历史窗口”的滚动均值/标准差标准化，替代全样本 StandardScaler，
# 消除跨周期分布偏移（PSI），且不引入未来函数。
ROLLING_ZSCORE_FEATURES = False
ROLLING_ZSCORE_LOOKBACK = 60
ROLLING_ZSCORE_MIN_PERIODS = 20
