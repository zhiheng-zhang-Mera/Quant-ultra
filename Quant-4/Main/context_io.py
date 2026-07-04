import json
import pickle
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("ContextIO")
CACHE_ROOT = Path(__file__).parent.parent.resolve() / "Phase_Result"

def get_phase_cache_dir(phase_name: str, format_type: str, modules_list: list) -> Path:
    base = CACHE_ROOT / format_type / f"Phase_{modules_list.index(phase_name) + 1}"
    base.mkdir(parents=True, exist_ok=True)
    return base

def save_phase_result(phase_name: str, result: Dict[str, Any], modules_list: list) -> None:
    if not isinstance(result, dict): return
    p_dir = get_phase_cache_dir(phase_name, "parquet", modules_list)
    f_dir = get_phase_cache_dir(phase_name, "feather", modules_list)
    metadata = {}
    for k, v in result.items():
        if k.startswith("_") or k in ["data_bus", "data_manager", "audit_logger"]: continue
        if isinstance(v, (pd.DataFrame, pd.Series)):
            p_path, f_path = p_dir / f"{k}.parquet", f_dir / f"{k}.feather"
            v.to_parquet(p_path, index=True)
            if isinstance(v, pd.DataFrame):
                metadata[k] = {"type": "dataframe", "index_names": list(v.index.names)}
                v.reset_index().to_feather(f_path)
            else:
                metadata[k] = {"type": "dataframe", "index_names": [v.index.name or "index"]}
                v.to_frame().reset_index().to_feather(f_path)
            logger.info("[OP] Write Tabular Engine State | [SOURCE] Memory Matrix Block | [RESULT] Persisted to Parquet: %s | [SIGNIFICANCE] Fast disk backup of analytical results", p_path.name)
            logger.info("[操作] 写入表格式引擎状态 | [来源] 内存矩阵块 | [结果] 已固化至Parquet: %s | [意义] 分析结果的盘片高弹性备份")
        else:
            try:
                json.dumps(v)
                with open(p_dir / f"{k}.json", "w") as f: json.dump(v, f, indent=2)
                metadata[k] = {"type": "json"}
            except Exception:
                with open(p_dir / f"{k}.pkl", "wb") as f: pickle.dump(v, f)
                metadata[k] = {"type": "pickle"}
                logger.info("[OP] Serialize Dynamic Model Entity | [SOURCE] Live Mathematical Kernel | [RESULT] Output generated: %s.pkl | [SIGNIFICANCE] Captures trainable parameters and structural graphs", k)
                logger.info("[操作] 序列化动态模型实体 | [来源] 活体数学内核 | [结果] 产出生成: %s.pkl | [意义] 捕获可训练参数及结构图谱")
    with open(p_dir / "metadata.json", "w") as f: json.dump(metadata, f, indent=2)
    shutil.copy(p_dir / "metadata.json", f_dir / "metadata.json")

def load_phase_result(phase_name: str, modules_list: list) -> Optional[Dict[str, Any]]:
    p_dir = get_phase_cache_dir(phase_name, "parquet", modules_list)
    meta_file = p_dir / "metadata.json"
    if not meta_file.exists():
        logger.warning("[OP] Probe Checkpoint Cache File | [SOURCE] Disk Directory Auditor | [RESULT] Missed! File path absent: %s | [SIGNIFICANCE] Signals cold start execution trajectory required", meta_file.name)
        logger.warning("[操作] 探测检查点缓存文件 | [来源] 磁盘目录审计器 | [结果] 未命中！物理路径不存在: %s | [意义] 标志着系统必须切入冷启动计算轨迹")
        return None
    with open(meta_file, "r") as f: metadata = json.load(f)
    res = {}
    for k, m in metadata.items():
        t = m.get("type")
        if t == "dataframe" and (p_dir / f"{k}.parquet").exists():
            res[k] = pd.read_parquet(p_dir / f"{k}.parquet")
        elif t == "json" and (p_dir / f"{k}.json").exists():
            with open(p_dir / f"{k}.json", "r") as f: res[k] = json.load(f)
        elif t == "pickle" and (p_dir / f"{k}.pkl").exists():
            with open(p_dir / f"{k}.pkl", "rb") as f: res[k] = pickle.load(f)
    logger.info("[OP] Hydrate Phase Context Cache | [SOURCE] Validated Cold Disk Blocks | [RESULT] Successfully inflated assets: %s | [SIGNIFICANCE] Hot injection bypasses heavy re-computation steps entirely", list(res.keys()))
    logger.info("[操作] 还原阶段上下文缓存 | [来源] 经验证的冷磁盘块 | [结果] 成功充能资产键: %s | [意义] 热注入技术完全绕过了沉重的重算步骤")
    return res if res else None

def save_context_snapshot(context: Dict, phase_name: str, run_ts: str, log_dir: Path):
    out = {k: f"<{type(v).__name__} shape={getattr(v, 'shape', 'N/A')}>" if isinstance(v, (pd.DataFrame, pd.Series)) else v for k, v in context.items() if k not in ["data_bus", "data_manager", "audit_logger"]}
    snap_file = log_dir / f"ctx_snap_{phase_name.replace('.', '_')}_{run_ts}.json"
    with open(snap_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)