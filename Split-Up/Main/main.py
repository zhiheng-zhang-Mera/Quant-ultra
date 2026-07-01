"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Main Orchestrator [2026 Production Release]
"""
import sys
import logging
import argparse
import json
import traceback
import time
import warnings
import importlib
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

def load_phase_module(phase_name: str):
    try:
        mod = importlib.import_module(phase_name)
        if hasattr(mod, "execute") and callable(mod.execute):
            return mod.execute
        logger.warning(f"模块 {phase_name} 缺少可调用 execute 函数")
        return None
    except ImportError as e:
        logger.warning(f"模块 {phase_name} 导入失败: {e}")
        return None

def check_dependencies(phase: str, completed_phases: Set[str]) -> bool:
    deps = PHASE_DEPENDENCIES.get(phase, set())
    missing = deps - completed_phases
    if missing:
        logger.error(f"阶段 {phase} 缺少前置依赖: {missing}")
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
            else:
                safe_ctx[k] = v
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(safe_ctx, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"快照保存失败: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Quant-Ultra + Conformal-BL 量化投资工作流主控器")
    parser.add_argument("--config", type=str, default="./config.yaml", help="外部 YAML 配置文件路径")
    parser.add_argument("--skip-phases", type=str, default="", help="跳过指定阶段")
    parser.add_argument("--only-phase", type=str, default=None, help="仅执行指定阶段")
    parser.add_argument("--resume-from", type=str, default=None, help="从指定阶段恢复执行")
    parser.add_argument("--no-git-check", action="store_true", help="跳过 Git 脏工作区检查")
    return parser.parse_args()

def run_pipeline(args):
    logger.info("=" * 80)
    logger.info("🚀 LAUNCHING QUANT-ULTRA + CONFORMAL-BL PRODUCTION FLOW (FULLY ATOMIC)")
    logger.info("=" * 80)

    if ENV_FINGERPRINT["git_status"] == "DIRTY" and not args.no_git_check:
        logger.critical("FATAL: Git 工作区存在未提交修改，禁止启动生产级回测！")
        sys.exit(1)

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
        data_manager = FreeDataSourceManager()
        audit_logger = AuditLogger(LOG_DIR, RUN_TIMESTAMP)
        data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)
    except Exception as e:
        logger.critical(f"❌ 基础设施原子装配失败: {e}"); sys.exit(1)

    config = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f: config = yaml.safe_load(f) or {}
        except Exception as e: logger.warning(f"配置加载失败: {e}")

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
    }
    for k, v in default_config.items():
        if k not in config: config[k] = v

    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": ENV_FINGERPRINT["git_commit_hash"], "args": vars(args)},
        "config": config, "data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger,
        "trading_days_dt": None, "slices": {}, "assets": [], "embargo_window": None, "holding_period": config.get("holding_period", 5),
        "_completed_phases": set(), "_phase_status": {}, "_phase_timings": {},
    }

    try:
        cal = data_manager.fetch_trading_calendar(2010, 2026)
        pipeline_context["trading_days_dt"] = cal.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()
    except Exception as e:
        logger.critical(f"❌ 交易日历加载失败: {e}"); sys.exit(1)

    for phase_name in phases_to_run:
        if not check_dependencies(phase_name, pipeline_context["_completed_phases"]):
            logger.critical(f"❌ 阶段 {phase_name} 前置依赖未满足"); sys.exit(1)

        func = load_phase_module(phase_name)
        if func is None:
            pipeline_context["_phase_status"][phase_name] = "skipped"
            continue

        logger.info(f">>> 开始执行阶段: {phase_name} <<<")
        start_time = time.time()
        try:
            pipeline_context["_current_phase"] = phase_name
            result = func(pipeline_context)
            if not isinstance(result, dict): raise TypeError(f"阶段 {phase_name} 必须返回 dict 字典对象")
            
            pipeline_context.update(result)
            elapsed = time.time() - start_time
            pipeline_context["_completed_phases"].add(phase_name)
            pipeline_context["_phase_status"][phase_name] = "success"
            pipeline_context["_phase_timings"][phase_name] = elapsed
            audit_logger.log_event("PHASE_SUCCESS", {"phase": phase_name, "elapsed_seconds": elapsed})
        except Exception as e:
            elapsed = time.time() - start_time
            logger.critical(f"❌ 阶段 {phase_name} 崩溃\n{traceback.format_exc()}")
            audit_logger.log_event("PHASE_FATAL", {"phase": phase_name, "exception": str(e)})
            audit_logger.flush()
            sys.exit(1)
        save_context_snapshot(pipeline_context, phase_name)

    audit_logger.flush()
    logger.info("🏁 QUANT-ULTRA WORKFLOW COMPLETED")

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)