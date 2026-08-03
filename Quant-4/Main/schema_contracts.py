import logging
from typing import Dict, Any, Set

logger = logging.getLogger("SchemaGuard")

PHASE_MODULES = [
    "Phase_1.step1_data_foundation",
    "Phase_2.step2_data_slicing",
    "Phase_3.step3_pit_setup",
    "Phase_4.step4_labeling_weighting",
    "Phase_5.step5_model_training_calibration",
    "Phase_6.step6_position_sizing",
    "Phase_7.step7_fsm_backtest",
    "Phase_8.step8_audit_stress_test",
    "Phase_9.step9_live_mlops",
    "Phase_10.step10_cio_reporting",
    "Phase_11.step11_interactive_advisor"
]

PHASE_INPUT_SCHEMA: Dict[str, Set[str]] = {
    "Phase_1.step1_data_foundation": set(),
    "Phase_2.step2_data_slicing": {"assets"},
    "Phase_3.step3_pit_setup": {"slices"},
    "Phase_4.step4_labeling_weighting": {"feature_panel_shared", "slices"},
    "Phase_5.step5_model_training_calibration": {"feature_panel_shared", "y_clf_all", "sample_weights", "calendar_alignment"},
    "Phase_6.step6_position_sizing": {"fractional_features_cube", "direction_classifier", "quantile_models"},
    "Phase_7.step7_fsm_backtest": {"daily_weights", "calendar_alignment"},
    "Phase_8.step8_audit_stress_test": {"daily_weights", "daily_adv20", "daily_nav", "violations", "audit_logger"},
    "Phase_9.step9_live_mlops": {"nav_history", "audit_summary", "config"}
    ,"Phase_10.step10_cio_reporting": {"audit_summary", "audit_passed", "recon_passed"}
    ,"Phase_11.step11_interactive_advisor": {"phase10_ready", "daily_weights", "data_manager"}
}

PHASE_OUTPUT_SCHEMA: Dict[str, Set[str]] = {
    "Phase_1.step1_data_foundation": {"assets", "adv_data", "theoretical_aum_limit"},
    "Phase_2.step2_data_slicing": {"slices", "embargo_window"},
    "Phase_3.step3_pit_setup": {"feature_panel_shared", "feature_panel_private_a", "feature_panel_private_us", "online_regime_state"},
    "Phase_4.step4_labeling_weighting": {"y_clf_all", "y_reg_all", "sample_weights"},
    "Phase_5.step5_model_training_calibration": {"direction_classifier", "quantile_models", "gamma_star", "q_error_threshold_dict", "selected_features", "fractional_features_cube"},
    "Phase_6.step6_position_sizing": {"daily_weights", "daily_intervals", "daily_adv20"},
    "Phase_7.step7_fsm_backtest": {"daily_nav", "daily_returns", "violations", "final_nav", "nav_history"},
    "Phase_8.step8_audit_stress_test": {"audit_passed", "audit_summary"},
    "Phase_9.step9_live_mlops": {"reconciliation_mae", "recon_passed", "psi_consecutive_breaches", "enforce_crowded_allocation_cap"}
    ,"Phase_10.step10_cio_reporting": {"cio_decision", "cio_evidence", "cio_report_path", "phase10_ready"}
    ,"Phase_11.step11_interactive_advisor": {"investment_candidates", "phase11_report_path", "phase11_csv_path", "phase11_ready"}
}

PHASE_DEPENDENCIES: Dict[str, Set[str]] = {
    "Phase_1.step1_data_foundation": set(),
    "Phase_2.step2_data_slicing": {"Phase_1.step1_data_foundation"},
    "Phase_3.step3_pit_setup": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing"},
    "Phase_4.step4_labeling_weighting": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup"},
    "Phase_5.step5_model_training_calibration": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup", "Phase_4.step4_labeling_weighting"},
    "Phase_6.step6_position_sizing": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup", "Phase_4.step4_labeling_weighting", "Phase_5.step5_model_training_calibration"},
    "Phase_7.step7_fsm_backtest": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup", "Phase_4.step4_labeling_weighting", "Phase_5.step5_model_training_calibration", "Phase_6.step6_position_sizing"},
    "Phase_8.step8_audit_stress_test": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup", "Phase_4.step4_labeling_weighting", "Phase_5.step5_model_training_calibration", "Phase_6.step6_position_sizing", "Phase_7.step7_fsm_backtest"},
    "Phase_9.step9_live_mlops": {"Phase_1.step1_data_foundation", "Phase_2.step2_data_slicing", "Phase_3.step3_pit_setup", "Phase_4.step4_labeling_weighting", "Phase_5.step5_model_training_calibration", "Phase_6.step6_position_sizing", "Phase_7.step7_fsm_backtest", "Phase_8.step8_audit_stress_test"},
    "Phase_10.step10_cio_reporting": {"Phase_9.step9_live_mlops"},
    "Phase_11.step11_interactive_advisor": {"Phase_10.step10_cio_reporting"},
}

def validate_phase_contract(phase_name: str, context: Dict[str, Any], stage: str = "output") -> bool:
    schema_dict = PHASE_INPUT_SCHEMA if stage == "input" else PHASE_OUTPUT_SCHEMA
    required_keys = schema_dict.get(phase_name, set())
    # logger.info("[OP] Scan Context Keys | [SOURCE] Runtime Pipeline State Map | [RESULT] Targets to verify: %s | [SIGNIFICANCE] Commences rigid validation of data element boundaries", list(required_keys))
    # logger.info("[操作] 扫描上下文键值 | [来源] 运行时流水线状态图 | [结果] 待验证目标群: %s | [意义] 开始对数据要素边界执行刚性验证")
    logger.info("由运行流水线扫描上下文键值以完成对数据要素边界执行刚性验证 | 当前阶段: %s | 待验证目标群: %s", stage, list(required_keys))
    missing_keys = [k for k in required_keys if k not in context or context[k] is None]
    if missing_keys:
        # logger.critical("[OP] Intercept Invalid Flow State | [SOURCE] Schema Structural Inspector | [RESULT] Validation failed! Missing tokens: %s | [SIGNIFICANCE] Hard break prevents downstream error propagation or feature poison", missing_keys)
        # logger.critical("[操作] 拦截非法流动状态 | [来源] Schema结构检查器 | [结果] 验证失败！缺失令牌: %s | [意义] 强行熔断防止下游错误级联或特征污染")
        logger.critical("由运行流水线拦截非法流动状态以防止下游错误级联或特征污染 | 当前阶段: %s | 缺失令牌: %s", stage, missing_keys)
        return False
    # logger.info("[OP] Validate Structural Contract Pass | [SOURCE] Integrity Assertion Evaluator | [RESULT] Node %s (%s verification) matches specification | [SIGNIFICANCE] Eliminates silent type corruption or matrix alignment failures", phase_name, stage)
    # logger.info("[操作] 验证结构化契约通过 | [来源] 完整性断言评估器 | [结果] 节点 %s (%s 校验) 与设计规范完全吻合 | [意义] 彻底消灭隐性类型损坏或矩阵对齐失效问题")
    logger.info("由运行流水线验证结构化契约通过 | 当前阶段: %s | 节点 %s (%s 校验) 与设计规范完全吻合", stage, phase_name, stage)
    return True
