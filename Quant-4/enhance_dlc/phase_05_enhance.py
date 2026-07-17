# -*- coding: utf-8 -*-
"""
Phase 05 增强模块：模型概率校准引擎
设计模式：后置校准拦截器
功能：
1. 对训练好的分类器进行 Platt Scaling 或 Isotonic Regression 校准
2. 使用独立的验证集（如 Train-B1）拟合校准器
3. 计算并记录 Brier Score，评估校准质量
4. 输出校准后的概率矩阵，供 Phase-6 使用
"""
import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.base import clone

logger = logging.getLogger("ModelTraining.Calibration")

class ProbabilityCalibrator:
    """
    概率校准器，支持 Platt Scaling 和 Isotonic Regression。
    使用 CalibratedClassifierCV 以 'prefit' 模式进行校准。
    """
    def __init__(self, method='isotonic', cv='prefit'):
        """
        Parameters
        ----------
        method : str, {'platt', 'isotonic'}
            校准方法，'platt' 对应 Platt Scaling (sigmoid)，'isotonic' 对应保序回归。
        cv : int or 'prefit'
            若为 'prefit'，则基模型已拟合，仅在校准集上学习映射；
            若为整数，则执行交叉验证校准（但会重新拟合基模型，可能不适合现有流程）。
        """
        self.method = method
        self.cv = cv
        self.calibrator = None
        self.base_estimator = None
        self.is_fitted = False

    def fit(self, base_estimator, X_cal, y_cal):
        """
        在校准集上拟合校准器。

        Parameters
        ----------
        base_estimator : 已训练的分类器（必须具有 predict_proba）
        X_cal : array-like, shape (n_cal_samples, n_features)
        y_cal : array-like, shape (n_cal_samples,)
            校准集的真实标签（多分类需为整数编码）
        """
        if not hasattr(base_estimator, "predict_proba"):
            raise ValueError("Base estimator must have predict_proba method.")

        # 检查基模型是否已拟合（粗略检测）
        try:
            base_estimator.predict_proba(X_cal[:1])
        except:
            raise RuntimeError("Base estimator appears not fitted. Please fit before calibration.")

        # 使用 CalibratedClassifierCV 进行校准，使用 'prefit' 模式
        calibrator = CalibratedClassifierCV(
            base_estimator=clone(base_estimator),
            method=self.method,
            cv=self.cv
        )
        # 由于 cv='prefit'，fit 方法不会重新训练基模型，仅拟合校准映射
        calibrator.fit(X_cal, y_cal)

        self.calibrator = calibrator
        self.base_estimator = base_estimator
        self.is_fitted = True

        # 计算校准集上的 Brier Score（多分类需平均）
        proba_cal = calibrator.predict_proba(X_cal)
        if proba_cal.shape[1] == 2:  # 二分类
            brier = brier_score_loss(y_cal, proba_cal[:, 1])
        else:  # 多分类，使用 one-vs-rest 平均
            n_classes = proba_cal.shape[1]
            brier = 0.0
            for i in range(n_classes):
                y_bin = (y_cal == i).astype(int)
                brier += brier_score_loss(y_bin, proba_cal[:, i])
            brier /= n_classes
        logger.info(f"Calibrated Brier Score (on calibration set): {brier:.4f}")

        # 若未校准前的概率存在，可计算对比
        if hasattr(base_estimator, "predict_proba"):
            proba_raw = base_estimator.predict_proba(X_cal)
            if proba_raw.shape[1] == 2:
                brier_raw = brier_score_loss(y_cal, proba_raw[:, 1])
            else:
                brier_raw = 0.0
                for i in range(n_classes):
                    y_bin = (y_cal == i).astype(int)
                    brier_raw += brier_score_loss(y_bin, proba_raw[:, i])
                brier_raw /= n_classes
            logger.info(f"Uncalibrated Brier Score (on calibration set): {brier_raw:.4f}")
            if brier < brier_raw:
                logger.info("Calibration improved Brier Score.")
            else:
                logger.warning("Calibration did not improve Brier Score; consider different method.")

        return self

    def predict_proba(self, X):
        """返回校准后的概率"""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.calibrator.predict_proba(X)

    def predict(self, X):
        """返回预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted.")
        return self.calibrator.predict(X)


def calibrate_trained_model(context: dict, method='isotonic', cal_partition='Train-B1'):
    """
    从上下文中提取训练好的分类器和校准数据，执行概率校准，并将校准器存入 context。

    Parameters
    ----------
    context : dict
        包含 'direction_classifier'（已拟合）、'feature_scaler'、'selected_features'、
        'slices'、'fractional_features_cube'、'trading_days_dt'、'assets'、'y_clf_all' 等。
    method : str, {'platt', 'isotonic'}
    cal_partition : str
        用于校准的切片名称，如 'Train-B1' 或 'Validation'。
    """
    clf = context.get('direction_classifier')
    if clf is None:
        logger.error("No classifier found in context. Run fit_model_bundle first.")
        return

    # 准备校准数据集（使用与 build_partition_dataset 相似的逻辑，但此处简化）
    from Phase_5.dataset_utils import build_partition_dataset
    X_cal, y_cal, _ = build_partition_dataset(context, cal_partition)
    if X_cal is None or len(X_cal) < 50:
        logger.warning(f"Calibration partition {cal_partition} has insufficient samples. Try falling back to Train-B1.")
        # 尝试使用 Train-B1 或 Validation
        for alt in ['Train-B1', 'Validation']:
            if alt != cal_partition:
                X_cal, y_cal, _ = build_partition_dataset(context, alt)
                if X_cal is not None and len(X_cal) >= 50:
                    logger.info(f"Using {alt} for calibration instead.")
                    break
        if X_cal is None or len(X_cal) < 50:
            logger.error("No suitable calibration data found. Skipping calibration.")
            return

    # 应用特征选择
    selected = context.get('selected_features', list(range(X_cal.shape[1])))
    X_cal_sub = X_cal[:, selected]

    # 标准化（使用训练时的 scaler）
    scaler = context.get('feature_scaler')
    if scaler is not None:
        X_cal_scaled = scaler.transform(X_cal_sub)
    else:
        logger.warning("No scaler found in context, using raw features.")
        X_cal_scaled = X_cal_sub

    # 确保标签类型正确（多分类整型）
    y_cal = np.asarray(y_cal, dtype=int)

    # 初始化校准器
    calibrator = ProbabilityCalibrator(method=method, cv='prefit')
    calibrator.fit(clf, X_cal_scaled, y_cal)

    # 将校准器存入 context
    context['calibrated_classifier'] = calibrator
    context['calibration_method'] = method

    # 可选：输出校准后的概率矩阵（用于 Phase-6）
    # 可在此处对全量数据（如 Test）预测并保存，但为保持灵活性，仅存储校准器。

    logger.info(f"Probability calibration ({method}) completed and stored in context['calibrated_classifier'].")
    return calibrator


def calibration_audit_hook(trainer_func):
    """
    装饰器（保留原始用途，但可扩展）：对训练函数包装，在训练后执行校准审计。
    现已被更完整的类取代，可保留向后兼容。
    """
    def wrapper(X, y, *args, **kwargs):
        model = trainer_func(X, y, *args, **kwargs)
        logger.info("增强注入：评估模型预测概率的置信度校准(Calibration)质量...")
        
        if hasattr(model, 'predict_proba'):
            try:
                from sklearn.metrics import brier_score_loss
                probs = model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1]==2 else model.predict_proba(X)
                # 对于多分类，简化处理（仅警告）
                if model.predict_proba(X).shape[1] > 2:
                    logger.warning("多分类模型，Brier Score 需按类别平均，此处仅打印原始警告。")
                else:
                    brier = brier_score_loss(y, probs)
                    if brier > 0.25:
                        logger.warning(f"增强警告：Brier Score ({brier:.4f}) 偏高，分类器过度自信，建议执行 Isotonic Regression 重校准。")
            except ImportError:
                pass
        return model
    return wrapper