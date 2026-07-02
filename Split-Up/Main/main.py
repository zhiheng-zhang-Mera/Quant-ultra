# -*- coding: utf-8 -*-
"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Main Orchestrator [2026 Production Release - Fully Federated Base Edition]
修复版：四重维度刚性金身单例生命周期红线，彻底消灭全缓存击中时对 DataBus 的字符串毒化覆盖。
"""
import sys
import logging
import argparse
import json
import traceback
import time
import warnings
import importlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import numpy as np
import pandas as pd
import pytz

# ========================================================
# 核心路径注入与原子解耦导入
# ========================================================
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入打散后的各自独立组件
from Main.env_config import get_git_hash, get_git_status
from Main.datasource_manager import FreeDataSourceManager
from Main.audit_logger import AuditLogger
from Main.data_bus import PITDataBus

warnings.filterwarnings("ignore")

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

RUN_TIMESTAMP = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")[:-3]
MAIN_LOG_FILE = LOG_DIR / f"orchestrator_{RUN_TIMESTAMP}.log"

# 阶段模块列表（严格按顺序依赖）
PHASE_MODULES = [
    "step1.step1_data_foundation",
    "step2.step2_data_slicing",
    "step3.step3_pit_setup",
    "step4.step4_labeling_weighting",
    "step5.step5_model_training_calibration",
    "step6.step6_position_sizing",
    "step7.step7_fsm_backtest",
    "step8.step8_audit_stress_test",
    "step9.step9_live_mlops",
]

PHASE_DEPENDENCIES: Dict[str, Set[str]] = {
    "step1.step1_data_foundation": set(),
    "step2.step2_data_slicing": {"step1.step1_data_foundation"},
    "step3.step3_pit_setup": {"step1.step1_data_foundation", "step2.step2_data_slicing"},
    "step4.step4_labeling_weighting": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup"},
    "step5.step5_model_training_calibration": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup", "step4.step4_labeling_weighting"},
    "step6.step6_position_sizing": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup", "step4.step4_labeling_weighting", "step5.step5_model_training_calibration"},
    "step7.step7_fsm_backtest": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup", "step4.step4_labeling_weighting", "step5.step5_model_training_calibration", "step6.step6_position_sizing"},
    "step8.step8_audit_stress_test": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup", "step4.step4_labeling_weighting", "step5.step5_model_training_calibration", "step6.step6_position_sizing", "step7.step7_fsm_backtest"},
    "step9.step9_live_mlops": {"step1.step1_data_foundation", "step2.step2_data_slicing", "step3.step3_pit_setup", "step4.step4_labeling_weighting", "step5.step5_model_training_calibration", "step6.step6_position_sizing", "step7.step7_fsm_backtest", "step8.step8_audit_stress_test"},
}

# ========================================================
# 对齐审查报告 4.1：全局显式数据契约生命周期定义 (Schema Guard)
# ========================================================
PHASE_INPUT_SCHEMA: Dict[str, Set[str]] = {
    "step1.step1_data_foundation": set(),
    "step2.step2_data_slicing": {"assets"},
    "step3.step3_pit_setup": {"slices"},
    "step4.step4_labeling_weighting": {"feature_panel_shared", "slices"},
    "step5.step5_model_training_calibration": {"feature_panel_shared", "y_clf_all", "sample_weights", "calendar_alignment"},
    "step6.step6_position_sizing": {"fractional_features_cube", "direction_classifier", "quantile_models"},
    "step7.step7_fsm_backtest": {"daily_weights", "calendar_alignment"},
    "step8.step8_audit_stress_test": {"daily_weights", "daily_adv20", "daily_nav", "violations", "audit_logger"},
    "step9.step9_live_mlops": {"nav_history", "audit_summary", "config"}
}

PHASE_OUTPUT_SCHEMA: Dict[str, Set[str]] = {
    "step1.step1_data_foundation": {"assets", "adv_data", "theoretical_aum_limit"},
    "step2.step2_data_slicing": {"slices", "embargo_window"},
    "step3.step3_pit_setup": {"feature_panel_shared", "feature_panel_private_a", "feature_panel_private_us", "online_regime_state"},
    "step4.step4_labeling_weighting": {"y_clf_all", "y_reg_all", "sample_weights"},
    "step5.step5_model_training_calibration": {"direction_classifier", "quantile_models", "gamma_star", "q_error_threshold_dict", "selected_features", "fractional_features_cube"},
    "step6.step6_position_sizing": {"daily_weights", "daily_intervals", "daily_adv20"},
    "step7.step7_fsm_backtest": {"daily_nav", "daily_returns", "violations", "final_nav", "nav_history"},
    "step8.step8_audit_stress_test": {"audit_passed", "audit_summary"},
    "step9.step9_live_mlops": {"reconciliation_mae", "recon_passed", "psi_consecutive_breaches", "enforce_crowded_allocation_cap"}
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(MAIN_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestrator")

ENV_FINGERPRINT = {
    "git_commit_hash": get_git_hash(),
    "git_status": get_git_status(),
    "python_version": sys.version,
    "run_timestamp": RUN_TIMESTAMP,
    "working_dir": str(PROJECT_ROOT)
}

def validate_phase_contract(phase_name: str, context: Dict[str, Any], stage: str = "output") -> bool:
    schema_dict = PHASE_INPUT_SCHEMA if stage == "input" else PHASE_OUTPUT_SCHEMA
    required_keys = schema_dict.get(phase_name, set())
    missing_keys = [key for key in required_keys if key not in context or context[key] is None]
    
    if missing_keys:
        diag_report = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "validation_stage": stage,
            "status": "CRITICAL_CONTRACT_BREAK",
            "missing_data_assets": missing_keys,
            "current_context_available_keys": list([k for k in context.keys() if not k.startswith("_")]),
            "remediation_guidance": f"请检查该阶段子模块的 execute 返回字典，确保显式包含且不为None: {missing_keys}"
        }
        logger.critical(f"🚨 [CONTRACT VIOLATION] 阶段 {phase_name} {stage} 数据契约断裂！诊断详情如下:\n"
                        f"{json.dumps(diag_report, indent=2, ensure_ascii=False)}")
        return False
    return True

# ========================================================
# 阶段结果稳健缓存层（核心组件红线硬阻断扩展版）
# ========================================================
CACHE_ROOT = PROJECT_ROOT / "Phase_Result"
PARQUET_CACHE = CACHE_ROOT / "parquet"
FEATHER_CACHE = CACHE_ROOT / "feather"

def get_phase_cache_dir(phase_name: str, format_type: str) -> Path:
    if phase_name not in PHASE_MODULES:
        raise ValueError(f"未知阶段: {phase_name}")
    idx = PHASE_MODULES.index(phase_name) + 1
    base = CACHE_ROOT / format_type / f"Phase_{idx}"
    base.mkdir(parents=True, exist_ok=True)
    return base

def save_phase_result(phase_name: str, result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        logger.warning(f"阶段 {phase_name} 返回结果不是字典，跳过缓存")
        return

    parquet_dir = get_phase_cache_dir(phase_name, "parquet")
    feather_dir = get_phase_cache_dir(phase_name, "feather")

    metadata = {}
    for key, value in result.items():
        # 🛡️ 防御第一层：写盘硬隔离。刚性剥离任何夹杂在产出字典中的核心业务单例
        if key.startswith("_") or key in ["data_bus", "data_manager", "audit_logger"]:  
            continue

        metadata[key] = {}
        if isinstance(value, (pd.DataFrame, pd.Series)):
            parquet_path = parquet_dir / f"{key}.parquet"
            feather_path = feather_dir / f"{key}.feather"
            try:
                value.to_parquet(parquet_path, index=True)
                if isinstance(value, pd.DataFrame):
                    metadata[key]["index_names"] = list(value.index.names)
                    value.reset_index().to_feather(feather_path)
                else:
                    metadata[key]["index_names"] = [value.index.name or "index"]
                    value.to_frame().reset_index().to_feather(feather_path)
                
                metadata[key]["type"] = "dataframe"
                metadata[key]["parquet"] = str(parquet_path.relative_to(CACHE_ROOT))
                metadata[key]["feather"] = str(feather_path.relative_to(CACHE_ROOT))
            except Exception as e:
                logger.error(f"❌ 固化数据资产键 [{key}] 遭遇异常: {e}")
                metadata[key]["type"] = "unsupported"
                
        elif isinstance(value, dict) and any(isinstance(k, tuple) for k in value.keys()):
            # 高维复合时空元组键高保真转化落盘
            json_path = parquet_dir / f"{key}.json"
            try:
                serialized_list = []
                for k, v in value.items():
                    k_elements = []
                    k_types = []
                    for sub_k in k:
                        k_types.append(type(sub_k).__name__)
                        if hasattr(sub_k, "strftime"):
                            k_elements.append(sub_k.strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            k_elements.append(str(sub_k))
                    serialized_list.append({"k": k_elements, "t": k_types, "v": v})
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_list, f, ensure_ascii=False, indent=2, default=str)
                metadata[key]["type"] = "tuple_dict"
                metadata[key]["json"] = str(json_path.relative_to(CACHE_ROOT))
            except Exception as e:
                logger.error(f"❌ 固化高维复合元组资产键 [{key}] 遭遇异常: {e}")
                metadata[key]["type"] = "unsupported"
        else:
            json_path = parquet_dir / f"{key}.json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2, default=str)
                metadata[key]["type"] = "json"
                metadata[key]["json"] = str(json_path.relative_to(CACHE_ROOT))
            except Exception as e:
                logger.warning(f"键 {key} 无法序列化为 JSON: {e}")
                metadata[key]["type"] = "unsupported"

    meta_path = parquet_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    shutil.copy(meta_path, feather_dir / "metadata.json")
    logger.info(f"✅ 阶段 {phase_name} 数据及元数据已稳健写盘")

def load_phase_result(phase_name: str) -> Optional[Dict[str, Any]]:
    parquet_dir = get_phase_cache_dir(phase_name, "parquet")
    meta_path = parquet_dir / "metadata.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning(f"阶段 {phase_name} 缓存元数据解析失败: {e}")
        return None

    result = {}
    for key, meta in metadata.items():
        # 🛡️ 防御第二层：读盘硬隔离。绝不允许任何历史同名毒化字段流入反序列化载具
        if key in ["data_bus", "data_manager", "audit_logger"]:
            continue
            
        if meta.get("type") == "dataframe":
            parquet_path = parquet_dir / f"{key}.parquet"
            if parquet_path.exists():
                try:
                    result[key] = pd.read_parquet(parquet_path)
                    continue
                except Exception as e:
                    logger.warning(f"读取 Parquet 主轨失败，切往 Feather: {e}")
            
            feather_dir = CACHE_ROOT / "feather" / f"Phase_{PHASE_MODULES.index(phase_name)+1}"
            feather_path = feather_dir / f"{key}.feather"
            if feather_path.exists():
                try:
                    df_raw = pd.read_feather(feather_path)
                    idx_names = meta.get("index_names", [])
                    valid_idx = [col for col in idx_names if col in df_raw.columns]
                    if valid_idx:
                        df_raw.set_index(valid_idx, inplace=True)
                    result[key] = df_raw
                    continue
                except Exception as e:
                    logger.warning(f"Feather 备用轨恢复失败: {e}")
        elif meta.get("type") == "tuple_dict":
            json_path = parquet_dir / f"{key}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        serialized_list = json.load(f)
                    reconstructed_dict = {}
                    for item in serialized_list:
                        k_elements = item["k"]
                        k_types = item["t"]
                        v = item["v"]
                        
                        reconstructed_k = []
                        for sub_k, t_name in zip(k_elements, k_types):
                            if t_name in ["Timestamp", "DatetimeIndex", "datetime", "Timestamp__"]:
                                reconstructed_k.append(pd.Timestamp(sub_k))
                            elif t_name in ["int", "int64"]:
                                reconstructed_k.append(int(sub_k))
                            elif t_name in ["float", "float64"]:
                                reconstructed_k.append(float(sub_k))
                            else:
                                reconstructed_k.append(str(sub_k))
                        reconstructed_dict[tuple(reconstructed_k)] = v
                    result[key] = reconstructed_dict
                except Exception as e:
                    logger.warning(f"读取元组键字典 {key} 失败，触发无损自愈: {e}")
        elif meta.get("type") == "json":
            json_path = parquet_dir / f"{key}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        result[key] = json.load(f)
                except Exception as e:
                    logger.warning(f"读取 JSON 键 {key} 失败: {e}")
    return result if result else None

def load_phase_module(phase_name: str):
    try:
        mod = importlib.import_module(phase_name)
        if hasattr(mod, "execute") and callable(mod.execute):
            return mod.execute
        return None
    except ImportError as e:
        logger.warning(f"模块 {phase_name} 分离式导入失败: {e}")
        return None

def check_dependencies(phase: str, completed_phases: Set[str]) -> bool:
    deps = PHASE_DEPENDENCIES.get(phase, set())
    missing = deps - completed_phases
    if missing:
        logger.error(f"❌ 阶段 {phase} 拦截: 缺少必要前置数据原子: {missing}")
        return False
    return True

def save_context_snapshot(context: Dict, phase_name: str):
    snapshot_path = LOG_DIR / f"context_{phase_name.replace('.', '_')}_{RUN_TIMESTAMP}.json"
    try:
        safe_ctx = {}
        for k, v in context.items():
            if k in ["data_bus", "audit_logger", "data_manager"]:
                safe_ctx[k] = f"<{type(v).__name__} object>"
            elif isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                safe_ctx[k] = f"<{type(v).__name__} shape={v.shape if hasattr(v, 'shape') else 'N/A'}>"
            elif isinstance(v, dict):
                safe_ctx[k] = {str(dk): dv for dk, dv in v.items()}
            else:
                safe_ctx[k] = v
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(safe_ctx, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"调试上下文影子快照输出失败: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Quant-Ultra + Conformal-BL 生产级主控分配内核")
    parser.add_argument("--config", type=str, default="./config.yaml", help="外部 YAML 配置文件路径")
    parser.add_argument("--skip-phases", type=str, default="", help="跳过指定阶段(逗号隔离)")
    parser.add_argument("--only-phase", type=str, default=None, help="约束仅执行指定 independent 阶段")
    parser.add_argument("--resume-from", type=str, default=None, help="自断点指定阶段恢复流水线")
    parser.add_argument("--no-git-check", action="store_true", help="强制关闭 Git 脏工作区校验硬红线")
    parser.add_argument("--offline", action="store_true", help="激活全离线调试模式，阻断一切外部 network 请求")
    parser.add_argument("--force-recompute", action="store_true", help="降级全量缓存，强制执行边缘计算")
    return parser.parse_args()

def run_pipeline(args):
    logger.info("=" * 80)
    logger.info("🚀 LAUNCHING QUANT-ULTRA + CONFORMAL-BL PRODUCTION FLOW (FULLY FEDERATED)")
    logger.info("=" * 80)

    if ENV_FINGERPRINT["git_status"] == "DIRTY" and not args.no_git_check:
        logger.critical("🚨 FATAL: 检测到生产工作区存留未提交修改，刚性熔断禁止启动回测！")
        sys.exit(1)

    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    FEATHER_CACHE.mkdir(parents=True, exist_ok=True)

    skip_set = {p.strip() for p in args.skip_phases.split(",")} if args.skip_phases else set()
    skip_set = {p for p in skip_set if p}

    if args.only_phase:
        target = args.only_phase if "." in args.only_phase else f"{args.only_phase.split('_')[0]}.{args.only_phase}"
        if target not in PHASE_MODULES: sys.exit(1)
        phases_to_run = []
        def collect_deps(p):
            for dep in PHASE_DEPENDENCIES.get(p, set()):
                if dep not in phases_to_run: collect_deps(dep)
            if p not in phases_to_run: phases_to_run.append(p)
        collect_deps(target)
        phases_to_run = [p for p in PHASE_MODULES if p in phases_to_run and p not in skip_set]
    elif args.resume_from:
        target = args.resume_from if "." in args.resume_from else f"{args.resume_from.split('_')[0]}.{args.resume_from}"
        phases_to_run = [p for p in PHASE_MODULES[PHASE_MODULES.index(target):] if p not in skip_set]
    else:
        phases_to_run = [p for p in PHASE_MODULES if p not in skip_set]

    try:
        data_manager = FreeDataSourceManager(offline_debug=args.offline)
        audit_logger = AuditLogger(LOG_DIR, RUN_TIMESTAMP)
        data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)
    except Exception as e:
        logger.critical(f"❌ 核心底层总线基础设施原子组装崩溃: {e}"); sys.exit(1)

    config = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f: config = yaml.safe_load(f) or {}
        except Exception as e: logger.warning(f"外部配置加载失败: {e}")

    np.random.seed(42)
    default_config = {
        "adv_window": 20, "min_adv_threshold": 1e7, "ipo_safety_days": 20, "max_participation_rate": 0.05,
        "expected_turnover": 0.05, "max_single_stock_weight": 0.05, "default_residual_rate": 0.0,
        "impact_alpha": 0.5, "impact_kappa_base": 0.05, "spread_lookback_days": 60, "stock_cap_pct": 0.045,
        "total_shares_source": "free_float", "short_rate_default": 0.08/252, "short_rate_source": "fixed",
        "tau_BL": 0.02, "omega_min": 1e-8, "omega_max": 0.01, "gamma_risk_initial": 2.5, "sector_limit": 0.3,
        "epsilon": 0.001, "transaction_cost_coeff": 0.0003, "lambda_decay": 0.01, "vol_window": 20,
        "threshold_multiplier": 0.5, "min_vol_obs": 5, "error_threshold_window": 252, "embargo_min": 5,
        "holding_period": 5, "max_leverage": 2.0, "d_min_search": [0.1, 0.3, 0.5, 0.7, 0.9], "vif_threshold": 30,
        "cluster_select_ratio": 0.8, "lgb_params": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05, "deterministic": True, "num_threads": 1, "random_state": 42, "verbosity": -1},
        "train_b1_grid_gamma": np.linspace(0.3, 0.7, 9).tolist(), "error_min_samples": 50, "cv_folds": 3,
        "psi_lookback_days": 60, "volatility_window": 20, "crowded_corr_threshold": 0.95, "vol_compress_quantile": 0.1,
        "mae_threshold": 1e-5, "watchdog_timeout": 30, "psi_threshold": 0.25, "psi_window": 5, "max_incremental_trees": 2000,
        "max_model_size": 2e9, "smoothing_period": 25,
        "federated_nodes": ["A_share_node", "US_share_node"],   
        "negative_transfer_patience": 3,                        
        "domain_adaptation_alpha": 0.1,                         
        "gradient_compression_top_k": 0.1,                      
        "domain_adaptation_loss_type": "MMD",                   
        "pure_ashare_baseline_loss": None,                      
        "negative_transfer_rollback_flag": False,               
    }
    for k, v in default_config.items():
        if k not in config: config[k] = v

    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": ENV_FINGERPRINT["git_commit_hash"], "args": vars(args)},
        "config": config, "data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger,
        "trading_days_dt": None, "slices": {}, "assets": [], "embargo_window": None, "holding_period": config.get("holding_period", 5),
        "_completed_phases": set(), "_phase_status": {}, "_phase_timings": {},
        "federated_status": {"current_round": 0, "node_sync_tokens": {}},
        "negative_transfer_monitor": {"consecutive_violation_count": 0, "triggered_melt": False}
    }

    try:
        cal_cn = data_manager.fetch_trading_calendar(2010, 2026)
        pipeline_context["trading_days_dt_cn"] = cal_cn.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()
        
        if hasattr(data_manager, "_ak") and not args.offline:
            try:
                df_us = data_manager._ak.index_us_stock_sina(symbol=".INX")
                if df_us is not None and not df_us.empty:
                    cal_us = pd.DatetimeIndex(pd.to_datetime(df_us["date"])).sort_values()
                    cal_us = cal_us[(cal_us.year >= 2010) & (cal_us.year <= 2026)]
                else: raise ValueError("日历抓取返回集为空")
            except Exception as e:
                logger.warning(f"⚠️ 跨市场交易日历启动弹保障轨：全同步对齐兜底")
                cal_us = cal_cn
        else: cal_us = cal_cn
            
        pipeline_context["trading_days_dt_us"] = cal_us.tz_localize(pytz.timezone("America/New_York")).tolist()
        cn_str_list = [d.strftime("%Y-%m-%d") for d in pipeline_context["trading_days_dt_cn"]]
        us_str_list = [d.strftime("%Y-%m-%d") for d in pipeline_context["trading_days_dt_us"]]
        min_len = min(len(cn_str_list), len(us_str_list))
        
        alignment_table = pd.DataFrame({
            "sequence_token": range(min_len), "ashare_date": cn_str_list[:min_len], "usshare_date": us_str_list[:min_len]
        })
        
        pipeline_context["calendar_alignment"] = {
            "alignment_table": alignment_table,
            "date_to_seq_cn": {d: i for i, d in enumerate(cn_str_list)},
            "date_to_seq_us": {d: i for i, d in enumerate(us_str_list)},
            "seq_to_date_cn": {i: d for i, d in enumerate(cn_str_list)},
            "seq_to_date_us": {i: d for i, d in enumerate(us_str_list)},
        }
        pipeline_context["trading_days_dt"] = pipeline_context["trading_days_dt_cn"]
        logger.info(f"✅ 跨市场双历对齐服务启动完毕。硬映射代数序号资产深度: {min_len} 个步进令牌。")
    except Exception as e:
        logger.critical(f"❌ 独立双系统交易日历装配阻断崩溃: {e}"); sys.exit(1)

    # 主循环体
    for phase_name in phases_to_run:
        if not check_dependencies(phase_name, pipeline_context["_completed_phases"]):
            logger.critical(f"❌ 依赖异常：阶段 {phase_name} 被物理拦截熔断"); sys.exit(1)

        cache_hit = False
        if not args.force_recompute:
            cached_data = load_phase_result(phase_name)
            if cached_data is not None:
                logger.info(f"✅ [CACHE HIT] 检测到阶段 {phase_name} 完整缓存，尝试执行热注入验证...")
                
                if validate_phase_contract(phase_name, cached_data, stage="output"):
                    pipeline_context.update(cached_data)
                    
                    # 🛡️ 防御第三层：热注入拨乱反正。强制洗净缓存可能带回的污染，死锁健康的活体内存单例
                    pipeline_context["data_bus"] = data_bus
                    pipeline_context["data_manager"] = data_manager
                    pipeline_context["audit_logger"] = audit_logger
                    
                    pipeline_context["_completed_phases"].add(phase_name)
                    pipeline_context["_phase_status"][phase_name] = "cached"
                    cache_hit = True
                    continue
                else:
                    logger.warning(f"⚠️ 阶段 {phase_name} 磁盘缓存未通过输出契约验证，降级为强制重算轨道...")

        if not cache_hit:
            if args.force_recompute:
                logger.info(f"⚡ 外部指令强制覆盖缓存，重新解算阶段: {phase_name} (--force-recompute)")
            else:
                logger.info(f"⏳ 阶段 {phase_name} 无可用存盘缓存，正常切入计算流...")

        # 🛡️ 刚性时空金身看门狗防线
        current_slices = pipeline_context.get("slices", {})
        if not current_slices or not any(current_slices.values()):
            if "calendar_alignment" in pipeline_context:
                align_table = pipeline_context["calendar_alignment"].get("alignment_table", pd.DataFrame())
                if not align_table.empty and "ashare_date" in align_table.columns:
                    full_timeline = align_table["ashare_date"].tolist()
                    idx = pd.DatetimeIndex(full_timeline).tz_localize(None)
                    pipeline_context["slices"] = {
                        "Train-A": idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist(),
                        "Train-B1": idx[(idx >= "2018-07-10") & (idx <= "2020-03-05")].strftime("%Y-%m-%d").tolist(),
                        "Train-B2": idx[(idx >= "2020-03-20") & (idx <= "2021-11-16")].strftime("%Y-%m-%d").tolist(),
                        "Validation": idx[(idx >= "2021-12-01") & (idx <= "2024-06-06")].strftime("%Y-%m-%d").tolist(),
                        "Test": idx[(idx >= "2024-06-24") & (idx <= "2026-12-31")].strftime("%Y-%m-%d").tolist()
                    }
                    logger.info(f"🛡️ [Orchestrator 时空看门狗] 全量分区日期已秒级自愈复原！")

        # 🛡️ 防御第四层：冷启动流物理锁死。在流入任何子模块计算核心前，强行将活体中枢单例注入上下文
        pipeline_context["data_bus"] = data_bus
        pipeline_context["data_manager"] = data_manager
        pipeline_context["audit_logger"] = audit_logger

        if not validate_phase_contract(phase_name, pipeline_context, stage="input"):
            logger.critical(f"❌ [INPUT BLOCK] 阶段 {phase_name} 上游输入要素不完整，刚性终止执行！")
            sys.exit(1)

        func = load_phase_module(phase_name)
        if func is None:
            pipeline_context["_phase_status"][phase_name] = "skipped"
            continue

        logger.info(f">>> 开始推进核心阶段: {phase_name} <<<")
        start_time = time.time()
        try:
            pipeline_context["_current_phase"] = phase_name
            result = func(pipeline_context)
            if not isinstance(result, dict):
                raise TypeError(f"规范违背错误: 阶段 {phase_name} 必须向总线返回 dict 对象")

            if not validate_phase_contract(phase_name, result, stage="output"):
                logger.critical(f"❌ [OUTPUT BLOCK] 阶段 {phase_name} 业务产出物与设计契约不吻合，拒绝合并进全局上下文！")
                sys.exit(1)

            pipeline_context.update(result)
            
            # 执行后再次确保活体单例安全
            pipeline_context["data_bus"] = data_bus
            pipeline_context["data_manager"] = data_manager
            pipeline_context["audit_logger"] = audit_logger

            elapsed = time.time() - start_time
            pipeline_context["_completed_phases"].add(phase_name)
            pipeline_context["_phase_status"][phase_name] = "success"
            pipeline_context["_phase_timings"][phase_name] = elapsed
            audit_logger.log_event("PHASE_SUCCESS", {"phase": phase_name, "elapsed_seconds": elapsed})

            save_phase_result(phase_name, result)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.critical(f"❌ CRITICAL FATAL: 阶段 {phase_name} 遭遇核心业务崩溃\n{traceback.format_exc()}")
            audit_logger.log_event("PHASE_FATAL", {"phase": phase_name, "exception": str(e)})
            audit_logger.flush()
            sys.exit(1)

        save_context_snapshot(pipeline_context, phase_name)

    audit_logger.flush()
    logger.info("🏁 QUANT-ULTRA ALL AGENTS WORKFLOW REPLAY COMPLETED WITH CONTRACT ASSURANCES")

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)