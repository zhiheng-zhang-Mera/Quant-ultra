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