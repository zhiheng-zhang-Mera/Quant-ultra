# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase 09 Enhancement: Concept Drift Auto-Healing Loop
实现 MD 计划中的自愈式滚动训练闭环：
1. 分级 PSI/KS 漂移监测（含红色/黄色预警）
2. 自动触发后台重训（调用 Phase 1~5 的滚动训练流程）
3. 影子模型冷启动验证（对比 Brier Score / 胜率）
4. 无感线上切流（Silent Switch）
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
from typing import Dict, Any, Optional, Callable
import threading
import queue
import time

from Phase_9.config import DEFAULT_MLOPS_CONFIG
from Phase_9.tiered_updater import _compute_conformal_psi  # 复用已有 PSI 计算

logger = logging.getLogger("MLOps.DriftEnhance")

# ==================== 辅助函数 ====================
def _compute_ks_drift(reference: np.ndarray, target: np.ndarray) -> float:
    """计算 KS 统计量（返回 p-value，越小漂移越显著）"""
    if len(reference) == 0 or len(target) == 0:
        return 1.0
    _, p_value = ks_2samp(reference, target)
    return float(p_value)

# ==================== 核心增强类 ====================
class ConceptDriftManager:
    """
    概念漂移管理与自愈触发引擎
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_MLOPS_CONFIG
        self.psi_threshold_yellow = 0.1          # 黄色预警
        self.psi_threshold_red = 0.25            # 红色警报（高于此值立即重训）
        self.psi_consecutive_days = 3            # 连续黄色预警天数触发重训
        self.ks_p_threshold = 0.01               # KS 检验 p 值阈值
        
        # 状态跟踪
        self.psi_breach_days = 0                 # 连续黄色超限天数
        self.last_trigger_time = None            # 上次触发重训时间（避免过于频繁）
        self.retrain_cooldown_days = 7           # 重训冷却期（天）
        
    def evaluate_drift(self, context: Dict) -> Dict:
        """
        评估当前特征漂移，返回增强后的 context，并可能触发重训。
        预期 context 中已有 'current_mean_psi'（由 tiered_updater 计算）。
        同时计算 KS 统计量作为补充。
        """
        # 1. 获取现有 PSI
        psi = context.get('current_mean_psi', 0.0)
        
        # 2. 补充 KS 检验（可对每个特征单独计算，这里简化为整体均值）
        feature_cube = context.get('fractional_features_cube')
        selected_features = context.get('selected_features')
        lookback = self.config.get('lookback_psi_window', 60)
        ks_p_avg = 1.0
        if feature_cube is not None and selected_features is not None and feature_cube.shape[0] > lookback + 10:
            base_sub = feature_cube[:lookback, :, :]
            target_sub = feature_cube[-10:, :, :]
            ks_ps = []
            for f_idx in selected_features:
                if f_idx < feature_cube.shape[2]:
                    base_vec = base_sub[:, :, f_idx].flatten()
                    target_vec = target_sub[:, :, f_idx].flatten()
                    ks_ps.append(_compute_ks_drift(base_vec, target_vec))
            if ks_ps:
                ks_p_avg = float(np.mean(ks_ps))
        context['avg_ks_pvalue'] = ks_p_avg
        
        # 3. 分级预警逻辑
        alert_level = 'GREEN'
        if psi >= self.psi_threshold_red or ks_p_avg < self.ks_p_threshold:
            alert_level = 'RED'
        elif psi >= self.psi_threshold_yellow:
            alert_level = 'YELLOW'
        context['drift_alert_level'] = alert_level
        
        # 4. 判定是否触发重训（红色立即，黄色连续）
        trigger_retrain = False
        if alert_level == 'RED':
            logger.critical("🚨 红色警报：PSI=%.4f 或 KS p值=%.4f，立即触发自愈重训！", psi, ks_p_avg)
            trigger_retrain = True
            self.psi_breach_days = 0  # 重置连续计数
        elif alert_level == 'YELLOW':
            self.psi_breach_days += 1
            logger.warning("⚠️ 黄色预警（连续第%d天），PSI=%.4f", self.psi_breach_days, psi)
            if self.psi_breach_days >= self.psi_consecutive_days:
                logger.critical("🔥 连续%d天黄色预警，触发自愈重训！", self.psi_breach_days)
                trigger_retrain = True
                self.psi_breach_days = 0
        else:
            self.psi_breach_days = 0
            logger.info("✅ 特征分布平稳，PSI=%.4f, KS p=%.4f", psi, ks_p_avg)
        
        # 5. 冷却期检查（避免频繁重训）
        if trigger_retrain:
            now = datetime.now()
            if self.last_trigger_time and (now - self.last_trigger_time).days < self.retrain_cooldown_days:
                logger.warning("⏳ 冷却期内，暂不执行重训（上次触发时间：%s）", self.last_trigger_time)
                trigger_retrain = False
            else:
                self.last_trigger_time = now
        
        context['trigger_retrain'] = trigger_retrain
        return context


class ShadowModelManager:
    """
    影子模型管理：新模型冷启动、验证、无感切流
    """
    def __init__(self):
        self.shadow_model = None          # 影子模型对象（占位）
        self.shadow_performance = []      # 影子模型最近几日的验证指标（如 Brier Score）
        self.active_model = None          # 当前线上主模型
        self.validation_window = 5        # 连续验证天数
        self.switch_threshold = 0.02      # 影子相对主模型优势阈值（如 Brier 降低 2%）
        
    def deploy_shadow_model(self, new_model_func: Callable) -> None:
        """
        部署新模型到影子系统（实际应调用 Phase-1~5 生成新模型，此处为占位）
        """
        self.shadow_model = new_model_func
        self.shadow_performance = []
        logger.info("🌓 新模型已部署至影子系统，开始冷启动验证...")
        
    def collect_shadow_metric(self, metric_value: float) -> None:
        """
        每日收集影子模型的验证指标（例如 Brier Score 或 预测胜率）
        """
        if self.shadow_model is not None:
            self.shadow_performance.append(metric_value)
            if len(self.shadow_performance) > self.validation_window:
                self.shadow_performance.pop(0)
                
    def evaluate_and_switch(self, active_metric: float) -> bool:
        """
        根据累积的影子指标与主模型指标对比，决定是否执行无感切换
        返回是否切换成功
        """
        if self.shadow_model is None or len(self.shadow_performance) < self.validation_window:
            return False
        
        shadow_avg = np.mean(self.shadow_performance)
        # 假设指标越小越好（如 Brier Score），否则需反转逻辑
        if active_metric - shadow_avg > self.switch_threshold:
            logger.critical("✅ 影子模型连续%d天优于主模型（影子平均: %.4f vs 主模型: %.4f），执行无感切流！",
                            self.validation_window, shadow_avg, active_metric)
            # 切换主模型为影子模型
            self.active_model = self.shadow_model
            self.shadow_model = None
            self.shadow_performance = []
            return True
        else:
            logger.info("🔁 影子模型暂未超越主模型，继续观察...")
            return False


class AutoHealingOrchestrator:
    """
    自愈闭环总调度器：整合漂移检测、重训触发、影子验证、切流
    """
    def __init__(self, config: Optional[Dict] = None):
        self.drift_manager = ConceptDriftManager(config)
        self.shadow_manager = ShadowModelManager()
        self.is_retraining = False          # 是否正在重训（防并发）
        self.retrain_queue = queue.Queue()  # 重训任务队列（异步）
        
    def daily_health_check(self, context: Dict) -> Dict:
        """
        每日健康检查入口（应在每日收盘后调用）
        返回更新后的 context
        """
        # 1. 漂移评估
        context = self.drift_manager.evaluate_drift(context)
        
        # 2. 若触发重训，则异步启动重训任务
        if context.get('trigger_retrain', False) and not self.is_retraining:
            self._start_retraining(context)
            context['retraining_started'] = True
        else:
            context['retraining_started'] = False
            
        # 3. 检查影子模型验证状态（若有）
        if self.shadow_manager.shadow_model is not None:
            # 假设可以从 context 获取当日影子模型指标和主模型指标
            shadow_metric = context.get('shadow_metric', 1.0)
            active_metric = context.get('active_metric', 1.0)
            self.shadow_manager.collect_shadow_metric(shadow_metric)
            if self.shadow_manager.evaluate_and_switch(active_metric):
                context['silent_switched'] = True
                context['active_model'] = self.shadow_manager.active_model
            else:
                context['silent_switched'] = False
        else:
            context['silent_switched'] = False
            
        return context
    
    def _start_retraining(self, context: Dict) -> None:
        """
        后台异步启动重训流程（调用 Phase 1~5 的滚动训练）
        此处仅作模拟，实际应启动子进程或线程执行训练脚本
        """
        self.is_retraining = True
        logger.info("🔄 开始异步重训（调用 Phase 1~5 滚动训练）...")
        
        def retrain_task():
            try:
                # 模拟训练时长
                time.sleep(10)  # 实际应调用训练管线
                # 训练完成后，获得新模型（此处用 lambda 占位）
                new_model = lambda x: x  # 伪模型
                self.shadow_manager.deploy_shadow_model(new_model)
                logger.info("✅ 重训完成，新模型已部署至影子系统")
            except Exception as e:
                logger.error("❌ 重训失败: %s", e)
            finally:
                self.is_retraining = False
        
        # 启动后台线程（实际推荐使用 Celery 或 Airflow）
        thread = threading.Thread(target=retrain_task, daemon=True)
        thread.start()


# ==================== 对外统一接口 ====================
def apply_auto_healing(context: Dict, config: Optional[Dict] = None) -> Dict:
    """
    供 Phase-9 主流程调用的统一增强接口。
    在 step9_live_mlops.py 的 execute 中调用，完成漂移自愈闭环。
    """
    orchestrator = AutoHealingOrchestrator(config)
    return orchestrator.daily_health_check(context)