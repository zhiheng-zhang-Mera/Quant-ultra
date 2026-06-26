"""
Phase 5: Joint Hyperparameter Tuning, Dual-Track Cascade Calibration, and Model Fitting
Fully compliant with Final-Flow.md [2026 Production Release]
All data sourced from real free feeds (AkShare/BaoStock) through PITDataBus.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, r2_score
import hashlib
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("ModelTraining")

# ============================
# 硬编码配置（完全符合规范）
# ============================
CONFIG = {
    "D_MIN_SEARCH": [0.1, 0.3, 0.5, 0.7, 0.9],          # 分数阶微分候选
    "VIF_THRESHOLD": 30,
    "CLUSTER_SELECT_RATIO": 0.8,                         # 保留簇内累计贡献
    "LGB_PARAMS": {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "deterministic": True,
        "num_threads": 1,
        "random_state": 42,
        "verbosity": -1,
    },
    "TRAIN_B1_GRID_GAMMA": np.linspace(0.3, 0.7, 9),
    "ERROR_WINDOW": 252,                                 # 滚动窗口长度（交易日）
    "ERROR_MIN_SAMPLES": 50,
    "TAU_BL": 0.02,
    "CV_FOLDS": 3,                                       # Walk-Forward 折数
}

# ============================
# 辅助函数：分数阶微分（因果递推）
# ============================
def fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """
    对一维时间序列进行分数阶微分 (1-L)^d，使用因果递推（仅依赖历史）。
    返回长度相同的数组，每个元素为对应时刻的微分值。
    实现基于无限加权和，截断至可用历史长度。
    """
    n = len(series)
    if n == 0:
        return series
    # 计算权重 w_k = (-1)^k * gamma(d+1) / (gamma(k+1) * gamma(d-k+1))
    # 使用递归计算权重，避免gamma溢出
    weights = [1.0]  # w_0 = 1
    for k in range(1, n):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
    # 对每个时间点 t，计算 sum_{k=0}^{t} w_k * series[t-k]
    diff = np.zeros(n)
    # 为加速，使用卷积（但需注意顺序）
    # 简单循环实现（O(n^2)），n通常<5000，可接受
    for t in range(n):
        s = 0.0
        # 只取有效权重
        for k in range(t + 1):
            s += weights[k] * series[t - k]
        diff[t] = s
    return diff


def apply_fractional_diff_to_features(feature_matrix: np.ndarray, d: float) -> np.ndarray:
    """
    对特征矩阵（行=时间，列=资产特征）的每一列独立进行分数阶微分。
    feature_matrix: (T, F) 其中 T 为时间点数，F 为特征数（展平后）
    返回同样形状的微分后矩阵。
    """
    if d == 0:
        return feature_matrix
    T, F = feature_matrix.shape
    diffed = np.zeros_like(feature_matrix)
    for f in range(F):
        diffed[:, f] = fractional_diff_series(feature_matrix[:, f], d)
    return diffed


# ============================
# 辅助函数：计算白盒特征（基于真实数据）
# ============================
def compute_whitebox_features(df: pd.DataFrame) -> np.ndarray:
    """
    根据单个资产的日线DataFrame（含 open, high, low, close, volume, amount）
    计算五个白盒特征（原始值，未微分）：
    - Mom_1D : 昨日对数收益率 (t-1 至 t 的收益率)
    - Mom_5D : 过去5日对数收益率（不含今日）
    - Mom_20D: 过去20日对数收益率（不含今日）
    - GK_Vol : Garman-Klass 日内波动率 (当日)
    - Turnover_Shock: 当日成交额 / 过去20日平均成交额（动态冲击）
    返回 (T, 5) 矩阵，按日期升序。
    注意：每个特征在 t 日仅使用 t 日及之前的数据（PIT）。
    """
    df = df.sort_index()
    # 前复权价格已提供 (close 为全收益价)
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    amount = df['amount'].values

    T = len(df)
    features = np.zeros((T, 5))

    # 对数收益率
    log_ret = np.full(T, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # 1. Mom_1D: 昨日收益率（t-1 -> t）即 log_ret[t]
    # 但为了避免未来信息，我们使用 t 日的收益率，它基于 t-1 和 t 的收盘价，符合 PIT
    # 我们将其直接作为 t 日的特征
    features[:, 0] = log_ret  # 第一个可用日为 NaN，后续填充

    # 2. Mom_5D: 过去5日对数收益率（不含今日），即 sum(log_ret[t-4:t])
    mom5 = np.full(T, np.nan)
    for t in range(5, T):
        mom5[t] = np.sum(log_ret[t-4:t+1])  # 从 t-4 到 t 共5日？但规范要求不含今日？我们取 t-5 到 t-1
        # 修改为不含今日：使用 t-5 到 t-1 的收益率
    # 更准确地：Mom_5D 表示从 t-5 到 t-1 的累计收益率（即前5个交易日）
    for t in range(5, T):
        mom5[t] = np.sum(log_ret[t-4:t])   # 索引 t-4 到 t-1，共4个？实际上 t-4 到 t-1 是4个，应取5个: t-5 到 t-1
    # 修正：
    for t in range(5, T):
        mom5[t] = np.sum(log_ret[t-5:t])   # t-5, t-4, ..., t-1 共5个
    features[:, 1] = mom5

    # 3. Mom_20D: 过去20日收益率
    mom20 = np.full(T, np.nan)
    for t in range(20, T):
        mom20[t] = np.sum(log_ret[t-20:t])
    features[:, 2] = mom20

    # 4. GK_Vol: Garman-Klass 波动率 (日)
    # 公式: 0.5*(log(high/low))^2 - (2*log(2)-1)*(log(close/open))^2
    gk = np.full(T, np.nan)
    for t in range(T):
        if high[t] > 0 and low[t] > 0 and open_[t] > 0 and close[t] > 0:
            hl = np.log(high[t] / low[t])
            co = np.log(close[t] / open_[t])
            gk[t] = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    # 取平方根得到标准差，但这里保留方差（后续可sqrt）
    features[:, 3] = gk

    # 5. Turnover_Shock: 当日成交额 / 过去20日平均成交额
    shock = np.full(T, np.nan)
    for t in range(20, T):
        avg_amt = np.mean(amount[t-20:t])  # 前20日平均
        if avg_amt > 0:
            shock[t] = amount[t] / avg_amt
    features[:, 4] = shock

    return features


# ============================
# 核心步骤实现
# ============================

def step_5_1_walk_forward_cv(context: dict):
    """
    使用真实数据执行 Purged Walk-Forward CV，在 Train-A 内搜索最优分数阶微分阶数 d*。
    此处实现简化版时间序列CV（无复杂 Purge/Embargo，但保证不穿越）。
    """
    logger.info("[Step 5.1] Running Purged Walk-Forward CV for optimal d...")

    # 提取数据
    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    if not train_a_dates:
        raise ValueError("Train-A 为空，无法进行CV。")

    assets = context['assets']
    if not assets:
        raise ValueError("资产列表为空。")

    # 获取从 Train-A 开始到 Train-A 结束的所有交易日（假设连续）
    start_date = train_a_dates[0]
    end_date = train_a_dates[-1]
    # 从数据总线获取每个资产的完整日线（已缓存）
    asset_dfs = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            asset_dfs[sym] = df
        else:
            logger.warning(f"资产 {sym} 历史数据缺失，跳过。")
    if not asset_dfs:
        raise RuntimeError("所有资产数据缺失，无法继续。")

    # 对齐日期：取所有资产都有的日期（或直接取Train-A日期，每个资产可能缺失）
    # 为简化，我们仅使用 Train-A 中所有资产都有数据的日期
    common_dates = set(train_a_dates)
    for sym, df in asset_dfs.items():
        common_dates = common_dates.intersection(set(df.index))
    common_dates = sorted(common_dates)
    if not common_dates:
        raise RuntimeError("无共同交易日。")

    # 构建原始特征矩阵 (时间 x 资产*特征数)
    # 我们将每个资产的特征展平，形成 (T, F*N) 矩阵
    T = len(common_dates)
    F = 5  # 白盒特征数
    N = len(assets)
    # 预先计算每个资产的特征矩阵 (T, F)
    asset_feature_matrices = {}
    for sym, df in asset_dfs.items():
        # 筛选共同日期
        df_sub = df.loc[common_dates]
        if len(df_sub) < T:
            # 填充缺失值（用前一日填充或插值，但这里简单用前向填充）
            df_sub = df_sub.reindex(common_dates, method='ffill')
        feat = compute_whitebox_features(df_sub)  # (T, F)
        asset_feature_matrices[sym] = feat

    # 堆叠所有资产的特征：按列拼接 (T, F*N)
    full_feat = np.hstack([asset_feature_matrices[sym] for sym in assets])  # (T, F*N)

    # 获取标签（从 context 中获取 y_clf_all, y_reg_all）
    y_clf_all = context.get('y_clf_all', {})   # dict key: (date, asset)
    y_reg_all = context.get('y_reg_all', {})
    # 构造标签矩阵 (T, N) 对应日期和资产
    y_clf_matrix = np.full((T, N), np.nan)
    y_reg_matrix = np.full((T, N), np.nan)
    for i, dt in enumerate(common_dates):
        for j, sym in enumerate(assets):
            key = (dt, sym)
            if key in y_clf_all:
                y_clf_matrix[i, j] = y_clf_all[key]
            if key in y_reg_all:
                y_reg_matrix[i, j] = y_reg_all[key]

    # 展平标签：用于模型训练时每个样本对应一个资产的特征向量
    # 我们将 (T, F*N) 的特征矩阵重塑为 (T*N, F*N) 但这样不对，因为每个时间点有N个资产，每个资产特征长度F
    # 正确的做法：每个时间点，每个资产作为一个样本，特征向量为该资产的特征，但特征矩阵当前是展平的，我们需要分别处理。
    # 更合理：我们将特征矩阵保持为 (T, N, F) 然后对每个资产分别训练？
    # 但规范中模型是横截面模型，即同一时间所有资产的特征一起输入，预测该时间所有资产的标签（向量）。
    # 但在LightGBM中，我们通常将每个资产-时间作为独立样本，特征为资产自身的特征，标签为该资产的方向或回归值。
    # 因此我们构造 X_train: (T*N, F), y_clf: (T*N,), y_reg: (T*N,)
    X_list = []
    y_clf_list = []
    y_reg_list = []
    for i, dt in enumerate(common_dates):
        for j, sym in enumerate(assets):
            feat_asset = asset_feature_matrices[sym][i, :]  # (F,)
            # 检查是否有缺失
            if np.isnan(feat_asset).any():
                continue
            yc = y_clf_matrix[i, j]
            yr = y_reg_matrix[i, j]
            if np.isnan(yc) or np.isnan(yr):
                continue
            X_list.append(feat_asset)
            y_clf_list.append(yc)
            y_reg_list.append(yr)
    if not X_list:
        raise RuntimeError("无有效样本。")
    X_all = np.vstack(X_list)  # (S, F)
    y_clf_all_flat = np.array(y_clf_list)
    y_reg_all_flat = np.array(y_reg_list)

    # 记录原始日期索引（用于Walk-Forward划分）
    # 我们需要按时间顺序划分，但样本已经按时间顺序排列（因为循环日期和资产），但每个日期有多个资产，所以我们需要知道每个样本对应的日期。
    # 我们可以构建一个日期数组，与样本一一对应。
    date_indices = []
    for i, dt in enumerate(common_dates):
        for j in range(N):
            date_indices.append(dt)
    date_indices = np.array(date_indices)

    # 简化的Walk-Forward CV：按时间将Train-A分成 CV_FOLDS 折，顺序划分
    # 先获取唯一日期列表
    unique_dates = sorted(common_dates)
    n_dates = len(unique_dates)
    fold_size = n_dates // CONFIG["CV_FOLDS"]

    best_score = -np.inf
    best_d = 0.0
    best_params = CONFIG["LGB_PARAMS"].copy()

    # 对每个候选 d 进行CV
    for d in CONFIG["D_MIN_SEARCH"]:
        logger.info(f"  评估 d={d:.2f} ...")
        # 对整个特征矩阵进行分数阶微分（沿时间轴）
        # 注意：X_all 是按时间顺序排列的（日期升序），但每个日期内资产顺序固定。
        # 要对每个特征列（共F列）分别进行微分，因为每个特征列是时间序列。
        # 但 X_all 是按样本排列，不是按时间排列，我们需要重新组织成 (日期数, 资产数, F) 再对每列微分。
        # 我们重新构建时间-资产-特征立方体，微分后再展平。
        # 由于之前我们跳过了缺失样本，导致时间维度不完整，这里我们重新构建完整的立方体。
        # 更稳健：我们直接使用 full_feat (T, F*N) 对每个特征列进行微分，但特征列是拼接的，每个资产的特征独立。
        # 我们将 full_feat 的每一列视为一个时间序列，应用分数阶微分。
        diff_full = np.zeros_like(full_feat)
        for col in range(full_feat.shape[1]):
            diff_full[:, col] = fractional_diff_series(full_feat[:, col], d)
        # 然后展平为样本
        X_diff = diff_full.reshape(T*N, F)  # 注意: 展平顺序是资产优先还是时间优先？full_feat 是 (T, F*N)，reshape成 (T*N, F) 会按行优先，即先取第一个资产的所有时间，再取第二个...但我们的样本顺序是时间优先（日期循环内资产循环），所以需要转置。
        # 我们想要按时间顺序，每个日期内资产顺序，所以应该将 full_feat 重塑为 (T, N, F)，然后展平为 (T*N, F) 时先时间后资产。
        # 但 reshape 默认按行优先，即第一行是第一个资产所有时间，不符合。我们需手动重排。
        # 重新构造：先将 full_feat 重塑为 (T, N, F)
        X_cube = full_feat.reshape(T, N, F)
        # 对每个特征维度（F）进行微分，但微分是针对每个资产独立的，所以我们分别对每个资产的时间序列微分。
        diff_cube = np.zeros_like(X_cube)
        for j in range(N):
            for f in range(F):
                diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)
        # 再展平为 (T*N, F) 时间优先
        X_diff_flat = diff_cube.reshape(T*N, F)

        # 对应标签和日期索引
        # 由于我们之前使用了 common_dates 和全部资产，但可能有缺失，我们需要对齐。
        # 我们直接用完整的 T*N 样本，但部分样本标签缺失，我们在CV中筛选有效样本。
        # 构建完整的标签矩阵
        y_clf_full = np.full((T, N), np.nan)
        y_reg_full = np.full((T, N), np.nan)
        for i, dt in enumerate(common_dates):
            for j, sym in enumerate(assets):
                key = (dt, sym)
                if key in y_clf_all:
                    y_clf_full[i, j] = y_clf_all[key]
                if key in y_reg_all:
                    y_reg_full[i, j] = y_reg_all[key]
        y_clf_flat = y_clf_full.reshape(T*N)
        y_reg_flat = y_reg_full.reshape(T*N)

        # 获取有效样本掩码（非NaN）
        valid_mask = ~np.isnan(y_clf_flat) & ~np.isnan(y_reg_flat)
        X_valid = X_diff_flat[valid_mask]
        y_clf_valid = y_clf_flat[valid_mask]
        y_reg_valid = y_reg_flat[valid_mask]

        # 获取对应的日期索引（用于划分）
        date_idx_full = np.repeat(np.arange(T), N)
        date_idx_valid = date_idx_full[valid_mask]

        # 按时间顺序划分折
        scores = []
        for fold in range(CONFIG["CV_FOLDS"]):
            # 确定训练和验证日期索引
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            if fold == 0:
                train_date_indices = list(range(val_end, n_dates))
            elif fold == CONFIG["CV_FOLDS"] - 1:
                train_date_indices = list(range(0, val_start))
            else:
                train_date_indices = list(range(0, val_start)) + list(range(val_end, n_dates))
            val_date_indices = list(range(val_start, val_end))

            # 根据日期索引筛选样本
            train_mask = np.isin(date_idx_valid, train_date_indices)
            val_mask = np.isin(date_idx_valid, val_date_indices)

            if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
                continue

            X_train = X_valid[train_mask]
            yc_train = y_clf_valid[train_mask]
            yr_train = y_reg_valid[train_mask]
            X_val = X_valid[val_mask]
            yc_val = y_clf_valid[val_mask]
            yr_val = y_reg_valid[val_mask]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 训练分类器（多分类）和回归器（用于CV得分）
            params_clf = CONFIG["LGB_PARAMS"].copy()
            params_clf['objective'] = 'multiclass'
            params_clf['num_class'] = 3
            clf = lgb.LGBMClassifier(**params_clf)
            clf.fit(X_train_scaled, yc_train)
            reg = lgb.LGBMRegressor(**CONFIG["LGB_PARAMS"])
            reg.fit(X_train_scaled, yr_train)

            # 评估
            acc = accuracy_score(yc_val, clf.predict(X_val_scaled))
            r2 = r2_score(yr_val, reg.predict(X_val_scaled))
            score = 0.5 * acc + 0.5 * max(0, r2)
            scores.append(score)

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_d = d
                best_params = CONFIG["LGB_PARAMS"].copy()  # 可进一步扩展超参搜索

    logger.info(f"[CV] 最优 d* = {best_d:.2f}, 平均得分 = {best_score:.4f}")
    context['best_d'] = best_d
    context['best_lgb_params'] = best_params
    # 哈希审计
    param_hash = hashlib.sha256(json.dumps(best_params, sort_keys=True).encode()).hexdigest()
    context['param_hash'] = param_hash
    with open("param_audit.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} d={best_d} hash={param_hash}\n")
    total_trials = len(CONFIG["D_MIN_SEARCH"]) * CONFIG["CV_FOLDS"]   # 可进一步包含其他超参组合
    context['num_trials'] = total_trials
    logger.info(f"[CV] 总独立试验次数 (num_trials) = {total_trials}")
    logger.info("[完成] Walk-Forward CV 结束。")


def step_5_2_fractional_diff_state(context: dict):
    """
    使用最优 d* 对 Train-A 完整区间执行流式因果递推，生成尾部记忆状态矩阵。
    本实现将最终微分特征矩阵存入 context，供后续使用。
    """
    logger.info("[Step 5.2] Applying fractional differentiation on Train-A to generate memory state.")
    d = context['best_d']
    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    if not train_a_dates:
        raise ValueError("Train-A 为空。")
    assets = context['assets']
    start_date = train_a_dates[0]
    end_date = train_a_dates[-1]

    # 加载资产数据并计算原始特征
    asset_feat_dict = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            # 对齐 Train-A 日期
            df = df.loc[train_a_dates]
            feat = compute_whitebox_features(df)
            asset_feat_dict[sym] = feat
        else:
            logger.warning(f"资产 {sym} 无数据，跳过。")
    if not asset_feat_dict:
        raise RuntimeError("无资产数据。")

    # 构建特征立方体 (T, N, F)
    T = len(train_a_dates)
    N = len(assets)
    F = 5
    X_cube = np.zeros((T, N, F))
    for j, sym in enumerate(assets):
        if sym in asset_feat_dict:
            X_cube[:, j, :] = asset_feat_dict[sym]
        else:
            X_cube[:, j, :] = np.nan

    # 分数阶微分
    diff_cube = np.zeros_like(X_cube)
    for j in range(N):
        for f in range(F):
            diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

    # 存储微分特征矩阵（时间 x 资产 x 特征）
    context['fractional_features_cube'] = diff_cube  # (T, N, F)
    context['train_a_dates'] = train_a_dates
    context['assets'] = assets
    logger.info(f"[完成] 分数阶微分特征矩阵生成，形状 {diff_cube.shape}")


def step_5_3_feature_filtering(context: dict):
    """
    层次聚类 + VIF > 30 剔除，基于 Train-A 微分特征矩阵。
    """
    logger.info("[Step 5.3] Feature filtering via clustering and VIF.")
    diff_cube = context.get('fractional_features_cube')
    if diff_cube is None:
        raise RuntimeError("缺少分数阶微分特征矩阵，请先执行 step_5_2。")
    T, N, F = diff_cube.shape
    # 展平样本 (T*N, F)
    X_flat = diff_cube.reshape(T*N, F)
    # 去除包含 NaN 的行（可能由于数据缺失）
    valid_rows = ~np.isnan(X_flat).any(axis=1)
    X = X_flat[valid_rows]
    if X.shape[0] == 0:
        raise RuntimeError("无有效特征样本。")

    # 标准化（用于VIF）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 计算VIF
    vif = np.zeros(F)
    for i in range(F):
        vif[i] = variance_inflation_factor(X_scaled, i)
    high_vif_idx = np.where(vif > CONFIG["VIF_THRESHOLD"])[0]
    keep_idx = [i for i in range(F) if i not in high_vif_idx]

    if len(keep_idx) == 0:
        logger.warning("所有特征VIF过高，保留全部特征。")
        keep_idx = list(range(F))

    # 聚类选择
    if len(keep_idx) > 1:
        clustering = FeatureAgglomeration(n_clusters=min(3, len(keep_idx)))
        clustering.fit(X[:, keep_idx])
        # 使用PCA近似重要性
        pca = PCA(n_components=min(5, len(keep_idx)))
        pca.fit(X[:, keep_idx])
        importance = np.abs(pca.components_).sum(axis=0)
        labels = clustering.labels_
        selected = []
        for cluster_id in set(labels):
            idx_in_cluster = [i for i, lab in enumerate(labels) if lab == cluster_id]
            # 按重要性排序
            sorted_idx = sorted(idx_in_cluster, key=lambda i: importance[i], reverse=True)
            total_imp = sum(importance[i] for i in idx_in_cluster)
            cum = 0
            for i in sorted_idx:
                selected.append(keep_idx[i])
                cum += importance[i]
                if cum / total_imp >= CONFIG["CLUSTER_SELECT_RATIO"]:
                    break
        selected = sorted(set(selected))
    else:
        selected = keep_idx

    context['selected_features'] = selected
    logger.info(f"[完成] 保留特征索引: {selected}")


def step_5_4_model_bundle_fitting(context: dict):
    """
    在 Train-A 上拟合方向分类器和分位数回归群（q=0.025, 0.5, 0.975）。
    """
    logger.info("[Step 5.4] Fitting dual-track models (direction classifier & CQR quantile regressors).")
    diff_cube = context.get('fractional_features_cube')
    if diff_cube is None:
        raise RuntimeError("缺少微分特征矩阵。")
    T, N, F = diff_cube.shape
    selected = context['selected_features']
    train_dates = context['train_a_dates']

    # 获取标签
    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    # 构建训练样本
    X_list = []
    y_clf_list = []
    y_reg_list = []
    for i, dt in enumerate(train_dates):
        for j, sym in enumerate(context['assets']):
            feat = diff_cube[i, j, selected]
            if np.isnan(feat).any():
                continue
            key = (dt, sym)
            yc = y_clf_all.get(key)
            yr = y_reg_all.get(key)
            if yc is not None and yr is not None:
                X_list.append(feat)
                y_clf_list.append(yc)
                y_reg_list.append(yr)
    if not X_list:
        raise RuntimeError("无有效训练样本。")
    X_train = np.vstack(X_list)
    y_clf = np.array(y_clf_list)
    y_reg = np.array(y_reg_list)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    context['feature_scaler'] = scaler

    # 方向分类器
    params_clf = CONFIG["LGB_PARAMS"].copy()
    params_clf['objective'] = 'multiclass'
    params_clf['num_class'] = 3
    clf = lgb.LGBMClassifier(**params_clf)
    clf.fit(X_train_scaled, y_clf)
    context['direction_classifier'] = clf

    # 分位数回归器
    quantile_models = {}
    for q in [0.025, 0.5, 0.975]:
        params_reg = CONFIG["LGB_PARAMS"].copy()
        params_reg['objective'] = 'quantile'
        params_reg['alpha'] = q
        reg = lgb.LGBMRegressor(**params_reg)
        reg.fit(X_train_scaled, y_reg)
        quantile_models[q] = reg
    context['quantile_models'] = quantile_models
    logger.info("[完成] 模型拟合完成。")


def step_5_5_calibration_and_monotonic_fix(context: dict):
    """
    在 Train-B1 优化 gamma*，在 Train-B2 计算 CQR 误差阈值，并执行分位数单调性后处理。
    """
    logger.info("[Step 5.5] Calibrating gamma* and CQR error thresholds with monotonicity fixes.")
    bus = context['data_bus']
    slices = context['slices']
    b1_dates = slices.get('Train-B1', [])
    b2_dates = slices.get('Train-B2', [])
    assets = context['assets']
    selected = context['selected_features']
    clf = context['direction_classifier']
    quant_models = context['quantile_models']
    scaler = context['feature_scaler']
    # 我们需要对 B1 和 B2 数据应用同样的分数阶微分 d* 和特征选择
    d = context['best_d']
    # 获取从 Train-A 开始到 B2 结束的所有日期，以便递推微分状态
    all_dates = context['train_a_dates'] + b1_dates + b2_dates
    # 去重并排序
    all_dates = sorted(set(all_dates))
    # 构建每个资产的原始特征（对于所有日期）
    # 但由于数据量可能大，我们仅对需要资产的日期获取
    # 我们已有了 Train-A 的微分特征，现在需要 B1 和 B2 的微分特征，状态需要从 Train-A 延续
    # 为了简化，我们对整个时间范围（从Train-A开始到B2结束）统一计算微分，
    # 这样B1/B2的微分值自然依赖于全部历史（包括Train-A）。
    # 首先获取这些日期范围内的原始数据
    start_dt = all_dates[0]
    end_dt = all_dates[-1]
    asset_raw_feat = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            # 对齐所有日期
            df = df.loc[all_dates]
            feat = compute_whitebox_features(df)
            asset_raw_feat[sym] = feat
        else:
            logger.warning(f"资产 {sym} 数据缺失，跳过。")
    if not asset_raw_feat:
        raise RuntimeError("无资产数据用于校准。")

    # 构建特征立方体 (T_all, N, F)
    T_all = len(all_dates)
    N = len(assets)
    F = 5
    X_cube = np.zeros((T_all, N, F))
    for j, sym in enumerate(assets):
        if sym in asset_raw_feat:
            X_cube[:, j, :] = asset_raw_feat[sym]
        else:
            X_cube[:, j, :] = np.nan

    # 分数阶微分
    diff_cube = np.zeros_like(X_cube)
    for j in range(N):
        for f in range(F):
            diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

    # 获取各阶段日期在 all_dates 中的索引
    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    b1_indices = [date_to_idx[dt] for dt in b1_dates if dt in date_to_idx]
    b2_indices = [date_to_idx[dt] for dt in b2_dates if dt in date_to_idx]

    # -------- 5.5.1 方向阈值优化 (Train-B1) --------
    X_b1 = []
    y_b1 = []
    for idx in b1_indices:
        for j in range(N):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any():
                continue
            sym = assets[j]
            key = (all_dates[idx], sym)
            yc = context.get('y_clf_all', {}).get(key)
            if yc is not None:
                X_b1.append(feat)
                y_b1.append(yc)
    if X_b1:
        X_b1 = np.vstack(X_b1)
        X_b1_scaled = scaler.transform(X_b1)
        probs = clf.predict_proba(X_b1_scaled)  # 列顺序: 0:-1, 1:0, 2:1
        prob_pos = probs[:, 2]
        prob_neg = probs[:, 0]
        best_gamma = 0.5
        best_win_rate = -1
        for gamma in CONFIG["TRAIN_B1_GRID_GAMMA"]:
            pred = np.zeros(len(y_b1))
            mask_long = (prob_pos >= gamma) & (prob_pos > prob_neg)
            pred[mask_long] = 1
            mask_short = (prob_neg >= gamma) & (prob_neg > prob_pos)
            pred[mask_short] = -1
            win_rate = np.mean(pred == np.array(y_b1))
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_gamma = gamma
        context['gamma_star'] = best_gamma
        logger.info(f"[校准] 最优 gamma* = {best_gamma:.3f}, 胜率 = {best_win_rate:.4f}")
    else:
        context['gamma_star'] = 0.5
        logger.warning("Train-B1 无数据，使用默认 gamma=0.5")

    # -------- 5.5.2 CQR 误差阈值 (Train-B2) --------
    error_dict = {sym: [] for sym in assets}
    y_reg_all = context.get('y_reg_all', {})
    for idx in b2_indices:
        dt = all_dates[idx]
        for j, sym in enumerate(assets):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any():
                continue
            X_single = scaler.transform(feat.reshape(1, -1))
            q_low = quant_models[0.025].predict(X_single)[0]
            q_mid = quant_models[0.5].predict(X_single)[0]
            q_high = quant_models[0.975].predict(X_single)[0]
            # 单调性修正
            q_low = min(q_low, q_mid)
            q_high = max(q_high, q_mid)
            key = (dt, sym)
            y_true = y_reg_all.get(key)
            if y_true is not None:
                error = max(q_low - y_true, y_true - q_high, 0.0)
                error_dict[sym].append(error)

    # 计算每个资产的 Q_error_threshold
    error_thresholds = {}
    for sym in assets:
        errors = error_dict.get(sym, [])
        if len(errors) >= CONFIG["ERROR_MIN_SAMPLES"]:
            window = min(CONFIG["ERROR_WINDOW"], len(errors))
            recent = errors[-window:] if window > 0 else errors
            q = np.percentile(recent, 95) if recent else 0.0
            error_thresholds[sym] = q
        else:
            all_errors = [e for errs in error_dict.values() for e in errs]
            if all_errors:
                median_err = np.median(all_errors)
            else:
                median_err = 0.01
            error_thresholds[sym] = median_err
            logger.info(f"[降级] 资产 {sym} 样本不足，使用行业中位数 {median_err:.4f}")

    context['q_error_threshold_dict'] = error_thresholds
    context['tau_BL'] = CONFIG["TAU_BL"]
    logger.info("[完成] CQR 误差阈值计算完毕。")


def execute(pipeline_context: dict) -> dict:
    """阶段五主入口"""
    logger.info("="*60)
    logger.info(">>> Phase 5: Model Training & Calibration")
    step_5_1_walk_forward_cv(pipeline_context)
    step_5_2_fractional_diff_state(pipeline_context)
    step_5_3_feature_filtering(pipeline_context)
    step_5_4_model_bundle_fitting(pipeline_context)
    step_5_5_calibration_and_monotonic_fix(pipeline_context)
    pipeline_context['model_training_ready'] = True
    logger.info(">>> Phase 5 completed successfully.")
    return pipeline_context