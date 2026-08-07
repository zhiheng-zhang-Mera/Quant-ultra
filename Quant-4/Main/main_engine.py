# -*- coding: utf-8 -*-
"""DEPRECATED - legacy orchestrator copy.

``Main/main.py`` is the canonical pipeline orchestrator. This module is kept
only for historical reference and is not imported by any pipeline component.
New features (bounded runs, resilient data-source fallback, multi-level
reports) are implemented in ``Main/main.py`` and will not be backported here.
"""
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

RUN_TIMESTAMP = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")[:-3]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_DIR / f"orchestrator_{RUN_TIMESTAMP}.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Orchestrator")

def parse_args():
    parser = argparse.ArgumentParser(description="Quant-Ultra Workflow Engine Core (Production Ready)")
    # [新增] 运行模式：实盘/增量 vs 回测/重算
    parser.add_argument("--mode", type=str, choices=["backtest", "live"], default="backtest", help="系统运行模式：回测(backtest)或实盘增量(live)")
    parser.add_argument("--config", type=str, default="./config.yaml", help="外部 YAML 配置文件路径 (优先级最高)")
    parser.add_argument("--skip-phases", type=str, default="", help="跳过指定阶段(逗号隔离)")
    parser.add_argument("--only-phase", type=str, default=None, help="约束仅执行指定独立阶段")
    parser.add_argument("--resume-from", type=str, default=None, help="自断点指定阶段恢复流水线")
    parser.add_argument("--no-git-check", action="store_true", help="强制关闭 Git 脏工作区校验硬红线")
    parser.add_argument("--offline", action="store_true", help="激活全离线调试模式")
    parser.add_argument("--force-recompute", action="store_true", help="降级全量缓存强制执行")
    return parser.parse_args()

def run_pipeline(args):
    logger.info(f"[OP] Boot Pipeline Framework | Mode: {args.mode.upper()} | [RESULT] System initialized | [SIGNIFICANCE] Entering master orchestrator deployment lifecycle")

    if get_git_status() == "DIRTY" and not args.no_git_check:
        logger.critical("🚨 检测到生产工作区存留未提交修改，刚性熔断禁止启动回测/实盘！ | Git Dirty Check Failed")
        sys.exit(1)

    # [调整6] 配置彻底解耦，从外部 yaml 读取默认配置
    config = {}
    default_cfg_path = CURRENT_DIR / "default_param.yaml"
    if default_cfg_path.exists():
        with open(default_cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"⚠️ 找不到基础配置文件 {default_cfg_path}，请确保 default_param.yaml 存在。")

    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f) or {}
                config.update(user_cfg)
            logger.info("[OP] Load External Config | [RESULT] Merged custom parameters successfully")
        except Exception as e:
            logger.warning(f"外部配置加载失败: {e}")

    data_manager = FreeDataSourceManager(offline_debug=args.offline)
    audit_logger = AuditLogger(LOG_DIR, RUN_TIMESTAMP)
    data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)

    # [调整1] 拆除2026年硬编码时间炸弹：动态获取当前年份
    current_year = datetime.now().year
    safety_end_year = current_year + 1

    sh_tz = pytz.timezone("Asia/Shanghai")
    ny_tz = pytz.timezone("America/New_York")
    
    cal_cn = data_manager.fetch_trading_calendar(2010, safety_end_year)
    cn_str_list = [d.strftime("%Y-%m-%d") for d in cal_cn]
    trading_days_dt_cn = cal_cn.tz_localize(sh_tz).tolist() if cal_cn.tz is None else cal_cn.tz_convert(sh_tz).tolist()
    
    try:
        cal_us = data_manager.fetch_us_trading_calendar(2010, safety_end_year)
        us_str_list = [d.strftime("%Y-%m-%d") for d in cal_us]
        trading_days_dt_us = cal_us.tz_localize(ny_tz).tolist() if cal_us.tz is None else cal_us.tz_convert(ny_tz).tolist()
    except Exception as e:
        logger.warning(f"⚠️ 美股日历获取失败，使用A股日历对齐兜底: {e}")
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

    # [调整1 & 5] 切片看门狗修改：实盘模式下动态截止到“今天”
    full_timeline = cn_str_list
    idx = pd.DatetimeIndex(full_timeline).tz_localize(None)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    test_slice_end = today_str if args.mode == "live" else f"{safety_end_year}-12-31"

    slices = {
        "Train-A": idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist(),
        "Train-B1": idx[(idx >= "2018-07-10") & (idx <= "2020-03-05")].strftime("%Y-%m-%d").tolist(),
        "Train-B2": idx[(idx >= "2020-03-20") & (idx <= "2021-11-16")].strftime("%Y-%m-%d").tolist(),
        "Validation": idx[(idx >= "2021-12-01") & (idx <= "2024-06-06")].strftime("%Y-%m-%d").tolist(),
        "Test": idx[(idx >= "2024-06-24") & (idx <= test_slice_end)].strftime("%Y-%m-%d").tolist()
    }

    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": get_git_hash(), "mode": args.mode},
        "config": config,
        "data_bus": data_bus,
        "data_manager": data_manager,
        "audit_logger": audit_logger,
        "assets": data_bus.get_universe(),
        "trading_days_dt": trading_days_dt,
        "calendar_alignment": calendar_alignment,
        "slices": slices,
        "_completed_phases": set(),
    }

    # [调整5] 增量实盘阶段跳过：实盘不跑重型训练、回测和压力测试
    skips = {x.strip() for x in args.skip_phases.split(",") if x.strip()}
    if args.mode == "live":
        live_skips = {
            "Phase_2.step2_1_slicing", "Phase_2.step2_2_validation",
            "Phase_5.step_5_1_cv", "Phase_5.step_5_4_fitting", "Phase_5.step_5_5_calibration",
            "Phase_7.step7_fsm_backtest", "Phase_8.step8_audit_stress_test"
        }
        skips.update(live_skips)
        logger.info(f"🚀 实盘模式激活！已自动静默跳过历史重算阶段: {live_skips}")

    if args.only_phase:
        target = args.only_phase if "." in args.only_phase else f"Phase_{args.only_phase.split('_')[0]}.{args.only_phase}"
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
        target = args.resume_from if "." in args.resume_from else f"Phase_{args.resume_from.split('_')[0]}.{args.resume_from}"
        phases_to_run = [p for p in PHASE_MODULES[PHASE_MODULES.index(target):] if p not in skips]
    else:
        phases_to_run = [p for p in PHASE_MODULES if p not in skips]

    for phase in phases_to_run:
        cache_hit = False
        if not args.force_recompute:
            cached = load_phase_result(phase, PHASE_MODULES)
            if cached and validate_phase_contract(phase, cached, "output"):
                pipeline_context.update(cached)
                pipeline_context.update({"data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger})
                pipeline_context["_completed_phases"].add(phase)
                cache_hit = True
                logger.info(f"[OP] Trigger Hot Cache Injection | [RESULT] {phase} memory fully restored")
                continue

        if not validate_phase_contract(phase, pipeline_context, "input"):
            sys.exit(1)

        logger.info(f"⏳ Executing computational block: {phase}")
        try:
            mod = importlib.import_module(phase)
            res = mod.execute(pipeline_context)
            if not isinstance(res, dict):
                raise TypeError("Output must return a dictionary mapping object.")
            if validate_phase_contract(phase, res, "output"):
                pipeline_context.update(res)
                pipeline_context.update({"data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger})
                pipeline_context["_completed_phases"].add(phase)
                save_phase_result(phase, res, PHASE_MODULES)
        except Exception as e:
            logger.critical(f"🚨 CRITICAL COLLAPSE at step [{phase}]: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        save_context_snapshot(pipeline_context, phase, RUN_TIMESTAMP, LOG_DIR)

    logger.info("🏁 ALL QUANT AGENTS EXECUTED SUCCESSFULLY WITH CONTRACT ASSURANCES")

if __name__ == '__main__':
    run_pipeline(parse_args())
