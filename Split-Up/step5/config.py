# -*- coding: utf-8 -*-
"""
step5/config.py
局部超参数定义：收拢模型底座配置、搜索网格及统计过滤阈值，拒绝硬编码污染。
"""
import numpy as np

# LightGBM 基础确定性配置
BASE_LGB_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "deterministic": True,
    "num_threads": 1,
    "random_state": 42,
    "verbosity": -1,
}

# 阶段 5.1 分数阶微分阶数 d* 搜索网格
D_MIN_SEARCH_SPACE = [0.1, 0.3, 0.5, 0.7, 0.9]
CV_FOLDS = 3

# 阶段 5.3 特征过滤统计学超参
VIF_THRESHOLD = 30.0
CLUSTER_SELECT_RATIO = 0.8

# 阶段 5.5 双轨标定与 Conformal 阈值配置
GAMMA_GRID = np.linspace(0.3, 0.7, 9)
ERROR_THRESHOLD_WINDOW = 252
ERROR_MIN_SAMPLES = 50
TAU_BL = 0.02

# 联邦迁移学习与领域对抗高级控制参数
TRANSFER_WEIGHT_INITIAL = 1.0
NEGATIVE_TRANSFER_PATIENCE = 3
MMD_ALPHA = 0.1
TOP_K_GRADIENT_RATIO = 0.1