"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Main Orchestrator [2026 Production Release - Fully Federated Base Edition]
Enhanced with:
1. Phase Result Caching (Parquet + Symmetric Feather Index Resetting)
2. Dual-Market Trading Calendar Alignment Service (Flow-Pro 1.4 Hard Synchronization)
3. Federated Learning & Negative Transfer Melt Control Infrastructure (Flow-Pro 5.1)
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

# ========================================================
# 阶段结果稳健缓存层（Parquet + Feather 修复版）
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
    """保存阶段结果（修复 Feather 无法序列化非默认索引的致命 Bug）"""
    if not isinstance(result, dict):
        logger.warning(f"阶段 {phase_name} 返回结果不是字典，跳过缓存")
        return

    parquet_dir = get_phase_cache_dir(phase_name, "parquet")
    feather_dir = get_phase_cache_dir(phase_name, "feather")

    metadata = {}
    for key, value in result.items():
        if key.startswith("_"):  
            continue

        metadata[key] = {}
        if isinstance(value, (pd.DataFrame, pd.Series)):
            parquet_path = parquet_dir / f"{key}.parquet"
            feather_path = feather_dir / f"{key}.feather"
            try:
                # 1. Parquet 原生支持多重/具名索引，直接保留 Index 固化
                value.to_parquet(parquet_path, index=True)
                
                # 2. Feather 刚性严禁具名索引。在此处执行重置转换为纯 Arrow 列存储（C语言消费规范对齐）
                if isinstance(value, pd.DataFrame):
                    # 记录原始 index 名字，以便加载时精准还原
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
        else:
            # 基础配置项标量、拓扑字典使用高容错 JSON 固化
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
        if meta.get("type") == "dataframe":
            parquet_path = parquet_dir / f"{key}.parquet"
            if parquet_path.exists():
                try:
                    result[key] = pd.read_parquet(parquet_path)
                    continue
                except Exception as e:
                    logger.warning(f"读取 Parquet 主轨失败，切往 Feather 灾备灾备: {e}")
            
            # Feather 灾备读取逻辑与还原
            feather_dir = CACHE_ROOT / "feather" / f"Phase_{PHASE_MODULES.index(phase_name)+1}"
            feather_path = feather_dir / f"{key}.feather"
            if feather_path.exists():
                try:
                    df_raw = pd.read_feather(feather_path)
                    idx_names = meta.get("index_names", [])
                    # 智能还原被剥离的具名索引
                    valid_idx = [col for col in idx_names if col in df_raw.columns]
                    if valid_idx:
                        df_raw.set_index(valid_idx, inplace=True)
                    result[key] = df_raw
                    continue
                except Exception as e:
                    logger.warning(f"Feather 备用轨恢复失败: {e}")
        elif meta.get("type") == "json":
            json_path = parquet_dir / f"{key}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        result[key] = json.load(f)
                except Exception as e:
                    logger.warning(f"读取 JSON 键 {key} 失败: {e}")
        else:
            logger.debug(f"键 {key} 为不受支持类型，略过")

    return result if result else None

# ========================================================
# 基础设施模块动态加载与依赖校验
# ========================================================
def load_phase_module(phase_name: str):
    try:
        mod = importlib.import_module(phase_name)
        if hasattr(mod, "execute") and callable(mod.execute):
            return mod.execute
        logger.warning(f"模块 {phase_name} 缺失标准的可调用 execute 入口")
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
    parser.add_argument("--only-phase", type=str, default=None, help="约束仅执行指定独立阶段")
    parser.add_argument("--resume-from", type=str, default=None, help="自断点指定阶段恢复流水线")
    parser.add_argument("--no-git-check", action="store_true", help="强制关闭 Git 脏工作区校验红线")
    parser.add_argument("--offline", action="store_true", help="激活全离线调试模式，阻断一切外部网络下载请求")
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
        except Exception as e: logger.warning(f"外部高级配置加载失败，转为默认策略: {e}")

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
        
        # ====================================================
        # 跨市场联邦迁移学习专属控制硬红线配置 (Flow-Pro 5.1/5.4 规范落地)
        # ====================================================
        "federated_nodes": ["A_share_node", "US_share_node"],   # 分布式独立私有数据特征节点映射
        "negative_transfer_patience": 3,                        # 触发微调回退的负迁移红线连续轮数上限
        "domain_adaptation_alpha": 0.1,                         # DANN架构最大均值差异（MMD）损失协同项平衡系数
        "gradient_compression_top_k": 0.1,                      # 隐私边界通信数据压缩比例约束
        "domain_adaptation_loss_type": "MMD",                   # 跨地理区域底层规律提炼对抗对齐损失函数类型
        "pure_ashare_baseline_loss": None,                      # 纯本地轨独立拟合性能参照系Loss锚点
        "negative_transfer_rollback_flag": False,               # 负迁移检测自适应熔断一键物理回归本地模型硬开关
    }
    for k, v in default_config.items():
        if k not in config: config[k] = v

    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": ENV_FINGERPRINT["git_commit_hash"], "args": vars(args)},
        "config": config, "data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger,
        "trading_days_dt": None, "slices": {}, "assets": [], "embargo_window": None, "holding_period": config.get("holding_period", 5),
        "_completed_phases": set(), "_phase_status": {}, "_phase_timings": {},
        
        # 联邦底层通信信道和防劣化监控变量初始化
        "federated_status": {"current_round": 0, "node_sync_tokens": {}},
        "negative_transfer_monitor": {"consecutive_violation_count": 0, "triggered_melt": False}
    }

    # ====================================================
    # Flow-Pro 1.4: 独立双市场交易日历服务与交易日序号硬对齐机制
    # ====================================================
    try:
        # 1. 组装中国交易日历独立服务核心时序
        cal_cn = data_manager.fetch_trading_calendar(2010, 2026)
        pipeline_context["trading_days_dt_cn"] = cal_cn.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()
        
        # 2. 组装美股交易日历独立服务核心时序
        if hasattr(data_manager, "_ak") and not args.offline:
            try:
                # 在线模式下通过获取标普500指数底层实际标的交易日期作为真实美股历轨
                df_us = data_manager._ak.index_us_stock_sina(symbol=".INX")
                if df_us is not None and not df_us.empty:
                    cal_us = pd.DatetimeIndex(pd.to_datetime(df_us["date"])).sort_values()
                    cal_us = cal_us[(cal_us.year >= 2010) & (cal_us.year <= 2026)]
                else:
                    raise ValueError("日历抓取返回集为空")
            except Exception as e:
                logger.warning(f"⚠️ 动态抓取美股实时交易日历异常({e})，系统启动弹保障轨：全同步对齐兜底")
                cal_us = cal_cn
        else:
            # 离线模式及无网络环境默认采用同步镜像序列对齐兜底
            cal_us = cal_cn
            
        pipeline_context["trading_days_dt_us"] = cal_us.tz_localize(pytz.timezone("America/New_York")).tolist()
        
        # 3. 核心升级：废弃自然日对齐，全量构建交易日序号双向映射（美股第 N 个交易日映射 A股第 N 个交易日）
        cn_str_list = [d.strftime("%Y-%m-%d") for d in pipeline_context["trading_days_dt_cn"]]
        us_str_list = [d.strftime("%Y-%m-%d") for d in pipeline_context["trading_days_dt_us"]]
        min_len = min(len(cn_str_list), len(us_str_list))
        
        alignment_table = pd.DataFrame({
            "sequence_token": range(min_len),
            "ashare_date": cn_str_list[:min_len],
            "usshare_date": us_str_list[:min_len]
        })
        
        pipeline_context["calendar_alignment"] = {
            "alignment_table": alignment_table,
            "date_to_seq_cn": {d: i for i, d in enumerate(cn_str_list)},
            "date_to_seq_us": {d: i for i, d in enumerate(us_str_list)},
            "seq_to_date_cn": {i: d for i, d in enumerate(cn_str_list)},
            "seq_to_date_us": {i: d for i, d in enumerate(us_str_list)},
        }
        
        # 默认回测引擎时序推进轴绑定 A 股独立底层
        pipeline_context["trading_days_dt"] = pipeline_context["trading_days_dt_cn"]
        logger.info(f"✅ 跨市场双历对齐服务启动完毕。硬映射代数序号资产深度: {min_len} 个步进令牌。")
    except Exception as e:
        logger.critical(f"❌ 中国/美股独立双系统交易日历装配阻断崩溃: {e}"); sys.exit(1)

    # ====================================================
    # 主循环体：支持级联缓存回溯的流程调度控制核
    # ====================================================
    for phase_name in phases_to_run:
        if not check_dependencies(phase_name, pipeline_context["_completed_phases"]):
            logger.critical(f"❌ 依赖异常：阶段 {phase_name} 被物理拦截熔断"); sys.exit(1)

        # 1. 尝试触发会话状态高速恢复（若未开启强制重算）
        if not args.force_recompute:
            cached_data = load_phase_result(phase_name)
            if cached_data is not None:
                logger.info(f"✅ [CACHE HIT] 检测到阶段 {phase_name} 完整缓存，正在热注入进程...")
                pipeline_context.update(cached_data)
                pipeline_context["_completed_phases"].add(phase_name)
                pipeline_context["_phase_status"][phase_name] = "cached"
                continue
            else:
                logger.info(f"⏳ 阶段 {phase_name} 无可用存盘缓存，正常切入计算流...")
        else:
            logger.info(f"⚡ 外部指令强制覆盖缓存，重新解算阶段: {phase_name} (--force-recompute)")

        # 2. 调度执行特定数据段/回测执行算子
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
                raise TypeError(f"开发接口规范违背错误: 阶段 {phase_name} 必须向总线返回 dict 字典对象")

            pipeline_context.update(result)
            elapsed = time.time() - start_time
            pipeline_context["_completed_phases"].add(phase_name)
            pipeline_context["_phase_status"][phase_name] = "success"
            pipeline_context["_phase_timings"][phase_name] = elapsed
            audit_logger.log_event("PHASE_SUCCESS", {"phase": phase_name, "elapsed_seconds": elapsed})

            # 3. 稳健持久化，同时生成未来 C 语言消费级的 Feather 文件及 Python Parquet 矩阵
            save_phase_result(phase_name, result)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.critical(f"❌ CRITICAL FATAL: 阶段 {phase_name} 遭遇核心业务崩溃\n{traceback.format_exc()}")
            audit_logger.log_event("PHASE_FATAL", {"phase": phase_name, "exception": str(e)})
            audit_logger.flush()
            sys.exit(1)

        save_context_snapshot(pipeline_context, phase_name)

    audit_logger.flush()
    logger.info("🏁 QUANT-ULTRA ALL AGENTS WORKFLOW REPLAY COMPLETED")

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)