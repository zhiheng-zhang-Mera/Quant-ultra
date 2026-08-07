# -*- coding: utf-8 -*-
import sys
import logging
import argparse
import traceback
import importlib
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main.env_config import get_git_hash, get_git_status, LOG_DIR
from Main.datasource_manager import FreeDataSourceManager
from Main.audit_logger import AuditLogger
from Main.data_bus import PITDataBus
from Main.schema_contracts import PHASE_MODULES, PHASE_DEPENDENCIES, validate_phase_contract
from Main.context_io import save_phase_result, load_phase_result, save_context_snapshot
from Main.stage_reporter import StageReporter
from Main.orchestration_guard import build_run_fingerprint, validate_orchestration, write_startup_manifest
from Main.schema_contracts import PHASE_INPUT_SCHEMA, PHASE_OUTPUT_SCHEMA, resolve_phase_name
from Main.distributed_compute import apply_resource_plan, initialize_distributed_compute
from Main.execute_report import generate_execute_report
from Main.universe_rules import symbol_purchase_eligibility

RUN_TIMESTAMP = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")[:-3]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_DIR / f"orchestrator_{RUN_TIMESTAMP}.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Orchestrator")

def parse_args():
    parser = argparse.ArgumentParser(description="Quant-Ultra Workflow Engine Core")
    parser.add_argument("--config", type=str, default="./config.yaml", help="外部 YAML 配置文件路径")
    parser.add_argument("--skip-phases", type=str, default="", help="跳过指定阶段(逗号隔离)")
    parser.add_argument("--only-phase", type=str, default=None, help="约束仅执行指定独立阶段")
    parser.add_argument("--resume-from", type=str, default=None, help="自断点指定阶段恢复流水线")
    parser.add_argument("--no-git-check", action="store_true", help="强制关闭 Git 脏工作区校验硬红线")
    parser.add_argument("--offline", action="store_true", help="激活全离线调试模式")
    parser.add_argument("--force-recompute", action="store_true", help="降级全量缓存强制执行")
    parser.add_argument("--non-interactive", action="store_true", help="阶段11只生成报告，不进入交互查询")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols for a bounded real-data run")
    parser.add_argument("--download-workers", type=int, default=None, help="Override Phase 1 market-data worker count")
    return parser.parse_args()

def run_pipeline(args):
    # logger.info("[OP] Boot Pipeline Framework | [SOURCE] Command Line Args Parse Node | [RESULT] System settings initialized | [SIGNIFICANCE] Entering master orchestrator deployment lifecycle", vars(args))
    # logger.info("[操作] 引导流水线框架启动 | [来源] 命令行参数解析节点 | [结果] 系统底层配置就绪 | [意义] 进入主控编排器的核心部署生命周期")
    print("[操作] 引导流水线框架启动 | [来源] 命令行参数解析节点 | [结果] 系统底层配置就绪 | [意义] 进入主控编排器的核心部署生命周期")

    if get_git_status() == "DIRTY" and not args.no_git_check:
        logger.critical("🚨 检测到生产工作区存留未提交修改，刚性熔断禁止启动回测！ | Git Dirty Check Failed")
        sys.exit(1)

    # ---- 完整默认配置 ----
    default_config = {
        "analysis_only": True,
        "adv_window": 20, "min_adv_threshold": 1e7, "ipo_safety_days": 20, "max_participation_rate": 0.05,
        "expected_turnover": 0.05, "max_single_stock_weight": 0.05, "default_residual_rate": 0.0,
        "impact_alpha": 0.5, "impact_kappa_base": 0.05, "spread_lookback_days": 60, "stock_cap_pct": 0.045,
        "total_shares_source": "free_float", "short_rate_default": 0.08/252, "short_rate_source": "fixed",
        "tau_BL": 0.02, "omega_min": 1e-8, "omega_max": 0.01, "gamma_risk_initial": 2.5, "sector_limit": 0.3,
        "epsilon": 0.001, "transaction_cost_coeff": 0.0003, "lambda_decay": 0.01, "vol_window": 20,
        "commission_rate": 0.00025, "minimum_commission": 5.0, "exchange_fee_rate": 0.0000341,
        "regulatory_fee_rate": 0.00002, "stamp_tax": 0.0005, "slippage_rate": 0.0002,
        "etf_annual_management_fee": 0.005, "cash_buffer_weight": 0.05,
        "minimum_invested_weight": 0.10, "max_daily_turnover": 0.25, "concentration_penalty": 0.02,
        "threshold_multiplier": 0.5, "min_vol_obs": 5, "error_threshold_window": 252, "embargo_min": 5,
        "holding_period": 5, "max_leverage": 2.0, "d_min_search": [0.1, 0.3, 0.5, 0.7, 0.9], "vif_threshold": 30,
        "cluster_select_ratio": 0.8, "lgb_params": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05, "deterministic": True, "num_threads": 1, "random_state": 42, "verbosity": -1},
        "train_b1_grid_gamma": np.linspace(0.3, 0.7, 9).tolist(), "error_min_samples": 50, "cv_folds": 3,
        "psi_lookback_days": 60, "volatility_window": 20, "crowded_corr_threshold": 0.95, "vol_compress_quantile": 0.1,
        "mae_threshold": 1e-5, "watchdog_timeout": 30, "psi_threshold": 0.25, "psi_window": 5, "max_incremental_trees": 2000,
        "max_model_size": 2e9, "smoothing_period": 25,
        "federated_nodes": ["A_share_node", "US_share_node"], "negative_transfer_patience": 3,
        "domain_adaptation_alpha": 0.1, "gradient_compression_top_k": 0.1,
        "domain_adaptation_loss_type": "MMD", "pure_ashare_baseline_loss": None, "negative_transfer_rollback_flag": False,
        "distributed_cpu_worker_cap": 16, "distributed_io_worker_cap": 24,
        "distributed_allow_gpu": True, "distributed_gpu_backend_ready": False, "connectivity_probe_timeout": 1.0,
        "news_input_path": None, "forum_input_path": None, "local_llm_sentiment_enabled": True,
        "local_llm_model": "qwen3-coder:30b", "local_llm_base_url": "http://127.0.0.1:11434",
        "local_llm_timeout_seconds": 20, "local_llm_max_records_total": 6,
        "local_llm_max_records_per_symbol": 2, "local_llm_max_chars_per_record": 300,
        "rotation_mode": "FULL_MARKET_DAILY_GUERRILLA", "rotation_rebalance_days": 1,
        "rotation_minimum_market_coverage": 50,
        "rotation_minimum_stock_coverage": 100,
        "market_source_timeout_seconds": 90.0,
        "market_source_cooldown_seconds": 60.0, "market_source_max_cooldown_seconds": 600.0,
    }
    config = default_config.copy()
    config["phase11_interactive"] = not args.non_interactive
    config["bounded_universe"] = bool(args.symbols.strip())
    if args.download_workers is not None:
        if args.download_workers < 1:
            raise ValueError("--download-workers must be at least 1")
        config["download_workers"] = args.download_workers
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f) or {}
                config.update(user_cfg)
            # logger.info("[OP] Load External Config | [SOURCE] YAML File Parser | [RESULT] Merged custom parameters successfully | [SIGNIFICANCE] Overrides default kernel hyperparameters")
            print("[操作] 加载外部配置 | [来源] YAML文件解析器 | [结果] 成功合并自定义参数 | [意义] 覆盖默认内核超参数")
        except Exception as e:
            logger.warning(f"外部配置加载失败: {e}")
    if config.get("analysis_only") is not True:
        raise ValueError("Quant-4 is an analysis-only engine; analysis_only must remain true")

    # The run fingerprint must be stable across hardware states: it feeds the
    # phase cache. Hardware-derived resource-plan keys change every run and
    # would otherwise invalidate all cached phase outputs.
    run_fingerprint, fingerprint_material = build_run_fingerprint(get_git_hash(), config, PHASE_MODULES)
    compute_audit = initialize_distributed_compute(config, PROJECT_ROOT)
    config = apply_resource_plan(config, compute_audit)
    logger.info("Distributed compute initialized | hardware=%s | connectivity=%s | plan=%s", compute_audit["hardware"], compute_audit["connectivity"], compute_audit["resource_plan"])
    dag_audit = validate_orchestration(PHASE_MODULES, PHASE_DEPENDENCIES, PHASE_INPUT_SCHEMA, PHASE_OUTPUT_SCHEMA)
    startup_manifest = write_startup_manifest(PROJECT_ROOT / "reports", RUN_TIMESTAMP, run_fingerprint, fingerprint_material, dag_audit)

    # ---- 核心组件 ----
    data_manager = FreeDataSourceManager(offline_debug=args.offline, source_timeout_seconds=float(config.get("market_source_timeout_seconds", 30.0)), source_cooldown_seconds=float(config.get("market_source_cooldown_seconds", 60.0)), source_max_cooldown_seconds=float(config.get("market_source_max_cooldown_seconds", 600.0)))
    audit_logger = AuditLogger(LOG_DIR, RUN_TIMESTAMP)
    stage_reporter = StageReporter(PROJECT_ROOT / "reports", RUN_TIMESTAMP, get_git_hash())
    data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)
    requested_symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if requested_symbols:
        # Bounded verification runs: use exactly the requested symbols and skip
        # the full-market rotation path. Production default (no --symbols)
        # continues to use full-market sector rotation.
        logger.info("Bounded universe requested: %s symbols", len(requested_symbols))
        full_market = requested_symbols
    else:
        full_market = data_manager.fetch_full_market_list(include_delisted=True)
        full_market = [symbol for symbol in full_market if symbol_purchase_eligibility(symbol)[0]]
        minimum_coverage = int(config.get("rotation_minimum_market_coverage", 50))
        minimum_stocks = int(config.get("rotation_minimum_stock_coverage", 100))
        stock_count = sum(not symbol.split(".")[0].startswith(("15", "16", "50", "51", "56", "58")) for symbol in full_market)
        if not args.offline and (len(full_market) < minimum_coverage or stock_count < minimum_stocks):
            raise RuntimeError(f"full-market rotation requires at least {minimum_coverage} securities and {minimum_stocks} stocks; found {len(full_market)} and {stock_count}")
    config["bounded_universe"] = bool(requested_symbols)
    data_bus.set_universe(full_market)

    # ---- 双市场日历对齐 ----
    sh_tz = pytz.timezone("Asia/Shanghai")
    ny_tz = pytz.timezone("America/New_York")
    cal_cn = data_manager.fetch_trading_calendar(2010, datetime.now().year)
    cn_str_list = [d.strftime("%Y-%m-%d") for d in cal_cn]
    trading_days_dt_cn = cal_cn.tz_localize(sh_tz).tolist() if cal_cn.tz is None else cal_cn.tz_convert(sh_tz).tolist()
    try:
        cal_us = data_manager.fetch_us_trading_calendar(2010, datetime.now().year)
        us_str_list = [d.strftime("%Y-%m-%d") for d in cal_us]
        trading_days_dt_us = cal_us.tz_localize(ny_tz).tolist() if cal_us.tz is None else cal_us.tz_convert(ny_tz).tolist()
    except Exception as e:
        logger.warning(f"[WARNING] 美股日历获取失败，使用A股日历对齐兜底: {e}")
        us_str_list = cn_str_list.copy()
        trading_days_dt_us = trading_days_dt_cn.copy()
    min_len = min(len(cn_str_list), len(us_str_list))
    alignment_table = pd.DataFrame({
        "sequence_token": range(min_len),
        "ashare_date": cn_str_list[:min_len],
        "usshare_date": us_str_list[:min_len]
    })
    calendar_alignment = {
        "alignment_table": alignment_table,
        "date_to_seq_cn": {d: i for i, d in enumerate(cn_str_list)},
        "date_to_seq_us": {d: i for i, d in enumerate(us_str_list)},
        "seq_to_date_cn": {i: d for i, d in enumerate(cn_str_list)},
        "seq_to_date_us": {i: d for i, d in enumerate(us_str_list)},
    }
    trading_days_dt = trading_days_dt_cn

    # ---- 切片看门狗 ----
    full_timeline = cn_str_list
    idx = pd.DatetimeIndex(full_timeline).tz_localize(None)
    slices = {
        "Train-A": idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist(),
        "Train-B1": idx[(idx >= "2018-07-10") & (idx <= "2020-03-05")].strftime("%Y-%m-%d").tolist(),
        "Train-B2": idx[(idx >= "2020-03-20") & (idx <= "2021-11-16")].strftime("%Y-%m-%d").tolist(),
        "Validation": idx[(idx >= "2021-12-01") & (idx <= "2024-06-06")].strftime("%Y-%m-%d").tolist(),
        "Test": idx[(idx >= "2024-06-24") & (idx <= f"{datetime.now().strftime('%Y-%m-%d')}")].strftime("%Y-%m-%d").tolist()
    }

    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": get_git_hash(), "run_fingerprint": run_fingerprint, "startup_manifest": str(startup_manifest)},
        "config": config,
        "compute_audit": compute_audit,
        "data_bus": data_bus,
        "data_manager": data_manager,
        "audit_logger": audit_logger,
        "assets": data_bus.get_universe(),
        "trading_days_dt": trading_days_dt,
        "trading_days_dt_cn": trading_days_dt_cn,
        "trading_days_dt_us": trading_days_dt_us,
        "calendar_alignment": calendar_alignment,
        "slices": slices,
        "_completed_phases": set(),
    }

    # ---- 阶段调度 ----
    skips = {x.strip() for x in args.skip_phases.split(",") if x.strip()}
    if args.only_phase:
        target = resolve_phase_name(args.only_phase)
        phases_to_run = []
        def collect_deps(p):
            for dep in PHASE_DEPENDENCIES.get(p, set()):
                if dep not in phases_to_run:
                    collect_deps(dep)
            if p not in phases_to_run:
                phases_to_run.append(p)
        collect_deps(target)
        phases_to_run = [p for p in PHASE_MODULES if p in phases_to_run and p not in skips]
    elif args.resume_from:
        target = resolve_phase_name(args.resume_from)
        # Resume must still traverse every phase so the cache loader can hydrate
        # the completed predecessors; execution only starts at the target.
        phases_to_run = [p for p in PHASE_MODULES if p not in skips]
        logger.info("Resume requested from %s; preceding phases will load from cache", target)
    else:
        phases_to_run = [p for p in PHASE_MODULES if p not in skips]

    for phase in phases_to_run:
        stage_reporter.start(phase)
        cache_hit = False
        if not args.force_recompute:
            cached = load_phase_result(phase, PHASE_MODULES, expected_fingerprint=run_fingerprint)
            if cached and validate_phase_contract(phase, cached, "output"):
                pipeline_context.update(cached)
                pipeline_context.update({"data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger})
                pipeline_context["_completed_phases"].add(phase)
                cache_hit = True
                stage_reporter.finish(phase, cached, True, cache_hit=True)
                # logger.info("[OP] Trigger Hot Cache Injection | [SOURCE] Binary Serialization Node | [RESULT] Phase memory fully restored | [SIGNIFICANCE] Enforces rigid singleton safety locks: %s", phase)
                logger.info("[操作] 触发热缓存注入 | [来源] 二进制序列化节点 | [结果] 阶段内存完全还原 | [意义] 强制执行严格的单例安全锁: %s", phase)
                continue

        if not validate_phase_contract(phase, pipeline_context, "input"):
            sys.exit(1)

        logger.info("[RUNNING] Executing computational block: %s", phase)
        try:
            mod = importlib.import_module(phase)
            res = mod.execute(pipeline_context)
            if not isinstance(res, dict):
                raise TypeError("Output must return a dictionary mapping object.")
            if validate_phase_contract(phase, res, "output"):
                pipeline_context.update(res)
                pipeline_context.update({"data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger})
                pipeline_context["_completed_phases"].add(phase)
                save_phase_result(phase, res, PHASE_MODULES, run_fingerprint=run_fingerprint)
                stage_reporter.finish(phase, res, True, cache_hit=False)
            else:
                raise ValueError(f"Output contract validation failed for {phase}")
        except Exception as e:
            stage_reporter.finish(phase, {"exception": repr(e), "traceback": traceback.format_exc()}, False, cache_hit=False)
            logger.critical(f"[CRITICAL] Pipeline failed at step [{phase}]: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        save_context_snapshot(pipeline_context, phase, RUN_TIMESTAMP, LOG_DIR)

    execute_report_path, execute_report_data_path = generate_execute_report(stage_reporter.root, pipeline_context, phases_to_run)
    pipeline_context["execute_report_path"] = str(execute_report_path)
    pipeline_context["execute_report_data_path"] = str(execute_report_data_path)
    logger.info("Execute report generated | HTML=%s | data=%s", execute_report_path, execute_report_data_path)
    logger.info("请求的 %d 个阶段已按合约执行完成", len(phases_to_run))
    return pipeline_context

if __name__ == '__main__':
    run_pipeline(parse_args())
