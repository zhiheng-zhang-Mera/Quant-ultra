# -*- coding: utf-8 -*-
import sys
import logging
import argparse
import traceback
import importlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from Main.env_config import get_git_hash, get_git_status, LOG_DIR
from Main.datasource_manager import FreeDataSourceManager
from Main.audit_logger import AuditLogger
from Main.data_bus import PITDataBus
from Main.schema_contracts import PHASE_MODULES, validate_phase_contract
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
    parser = argparse.ArgumentParser(description="Quant-Ultra Workflow Engine Core")
    parser.add_argument("--skip-phases", type=str, default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()

def run_pipeline(args):
    logger.info("[OP] Boot Pipeline Framework | [SOURCE] Command Line Args Parse Node | [RESULT] System settings initialized | [SIGNIFICANCE] Entering master orchestrator deployment lifecycle", vars(args))
    logger.info("[操作] 引导流水线框架启动 | [来源] 命令行参数解析节点 | [结果] 系统底层配置就绪 | [意义] 进入主控编排器的核心部署生命周期")
    
    if get_git_status() == "DIRTY":
        logger.critical("🚨 Hot-halt: Git dirty check failed.")
        sys.exit(1)

    data_manager = FreeDataSourceManager(offline_debug=args.offline)
    audit_logger = AuditLogger(LOG_DIR, RUN_TIMESTAMP)
    data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)
    
    cal_cn = data_manager.fetch_trading_calendar(2010, 2026)
    cn_str_list = [d.strftime("%Y-%m-%d") for d in cal_cn]
    
    pipeline_context = {
        "run_metadata": {"timestamp": RUN_TIMESTAMP, "git_hash": get_git_hash()},
        "config": {"holding_period": 5}, "data_bus": data_bus, "data_manager": data_manager, "audit_logger": audit_logger,
        "assets": data_bus.get_universe(), "_completed_phases": set(),
        "calendar_alignment": {"alignment_table": pd.DataFrame({"ashare_date": cn_str_list}), "date_to_seq_cn": {d: i for i, d in enumerate(cn_str_list)}}
    }
    pipeline_context["slices"] = {"Train-A": cn_str_list[:1000], "Validation": cn_str_list[1000:1500], "Test": cn_str_list[1500:]}

    skips = {x.strip() for x in args.skip_phases.split(",") if x.strip()}
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
                logger.info("[OP] Trigger Hot Cache Injection | [SOURCE] Binary Serialization Node | [RESULT] Phase memory fully restored | [SIGNIFICANCE] Enforces rigid singleton safety locks to overwrite toxic cache dictionaries", phase)
                logger.info("[操作] 触发热缓存注入 | [来源] 二进制反序列化节点 | [结果] 阶段数据内存恢复完成 | [意义] 强行应用刚性单例安全锁，清洗恶意或失效的历史毒化字典")
                continue

        if not validate_phase_contract(phase, pipeline_context, "input"): sys.exit(1)
        
        logger.info(f"⏳ Executing computational block: {phase}")
        try:
            mod = importlib.import_module(phase)
            res = mod.execute(pipeline_context)
            if not isinstance(res, dict): raise TypeError("Output must return a dictionary mapping object.")
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