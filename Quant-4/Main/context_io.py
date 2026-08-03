import json
import pickle
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("ContextIO")
CACHE_ROOT = Path(__file__).parent.parent.resolve() / "Phase_Result"

# 需要跳过保存的键（因为它们包含不可序列化的对象或不需要持久化）
IGNORED_KEYS = {
    "data_bus", "data_manager", "audit_logger",
    "fsm_engine", "context", "_yf", "manager",
    "calendar_alignment"  # 可能包含大对象
}

def get_phase_cache_dir(phase_name: str, format_type: str, modules_list: list) -> Path:
    base = CACHE_ROOT / format_type / f"Phase_{modules_list.index(phase_name) + 1}"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _is_picklable(obj) -> bool:
    """检测对象是否可以被 pickle 序列化"""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, TypeError, AttributeError, RecursionError):
        return False

def save_phase_result(phase_name: str, result: Dict[str, Any], modules_list: list, run_fingerprint: str = None) -> None:
    if not isinstance(result, dict):
        return
    p_dir = get_phase_cache_dir(phase_name, "parquet", modules_list)
    f_dir = get_phase_cache_dir(phase_name, "feather", modules_list)
    metadata = {}

    for k, v in result.items():
        # 跳过特定键
        if k.startswith("_") or k in IGNORED_KEYS:
            logger.debug("[SKIP] Key '%s' is ignored (not saved)", k)
            continue

        # ---- 处理 DataFrame / Series ----
        if isinstance(v, (pd.DataFrame, pd.Series)):
            try:
                p_path = p_dir / f"{k}.parquet"
                f_path = f_dir / f"{k}.feather"
                if isinstance(v, pd.DataFrame):
                    v.to_parquet(p_path, index=True)
                    metadata[k] = {"type": "dataframe", "index_names": list(v.index.names)}
                    v.reset_index().to_feather(f_path)
                else:  # Series
                    df = v.to_frame()
                    df.to_parquet(p_path, index=True)
                    metadata[k] = {"type": "dataframe", "index_names": [v.index.name or "index"]}
                    df.reset_index().to_feather(f_path)
                # logger.info("[OP] Write Tabular Engine | [SOURCE] Memory Matrix | [RESULT] Parquet/Feather saved: %s", p_path.name)
                logger.info("由上下文I/O模块保存表格引擎数据 | 来源: 内存矩阵 | 结果: Parquet/Feather已保存: %s", p_path.name)
            except Exception as e:
                logger.warning("[FAIL] Failed to save %s as DataFrame/Series: %s", k, e)
            continue

        # ---- 处理元组键的字典 ----
        if isinstance(v, dict) and any(isinstance(key, tuple) for key in v.keys()):
            json_path = p_dir / f"{k}.json"
            try:
                serialized_list = []
                for key_tuple, val in v.items():
                    k_elements = []
                    k_types = []
                    for sub_k in key_tuple:
                        k_types.append(type(sub_k).__name__)
                        if hasattr(sub_k, "strftime"):
                            k_elements.append(sub_k.strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            k_elements.append(str(sub_k))
                    serialized_list.append({"k": k_elements, "t": k_types, "v": val})
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_list, f, ensure_ascii=False, indent=2, default=str)
                metadata[k] = {"type": "tuple_dict"}
                # logger.info("[OP] Write Tuple-Dict JSON | [SOURCE] Memory Mapping | [RESULT] Saved: %s", json_path.name)
                logger.info("由上下文I/O模块保存元组键字典 | 来源: 内存映射 | 结果: 已保存: %s", json_path.name)
            except Exception as e:
                # logger.warning("[FAIL] Failed to save tuple-dict %s: %s", k, e)
                logger.warning("由上下文I/O模块保存元组键字典失败 | 键: %s | 错误: %s", k, e)
            continue

        # ---- 通用序列化：优先 JSON，失败则尝试 Pickle ----
        # 先检测是否可 pickle，若不可 pickle 则跳过
        if not _is_picklable(v):
            # logger.warning("[SKIP] Value for key '%s' is not picklable, skipping", k)
            logger.warning("由上下文I/O模块跳过不可序列化对象 | 键: %s | 原因: 无法被Pickle序列化", k)
            continue

        # 尝试 JSON
        try:
            json.dumps(v)
            with open(p_dir / f"{k}.json", "w") as f:
                json.dump(v, f, indent=2)
            metadata[k] = {"type": "json"}
            # logger.info("[OP] Write JSON | [SOURCE] Memory Atom | [RESULT] Saved: %s.json", k)
            logger.info("由上下文I/O模块保存JSON数据 | 结果: 已保存: %s.json", k)
        except Exception:
            # JSON 失败则使用 Pickle
            try:
                with open(p_dir / f"{k}.pkl", "wb") as f:
                    pickle.dump(v, f)
                metadata[k] = {"type": "pickle"}
                # logger.info("[OP] Serialize Dynamic Model | [SOURCE] Live Kernel | [RESULT] Pickle generated: %s.pkl", k)
                logger.info("由上下文I/O模块序列化动态模型 | 结果: Pickle已生成: %s.pkl", k)
            except Exception as e:
                logger.warning("由上下文I/O模块序列化动态模型失败 | 键: %s | 错误: %s", k, e)
                # logger.warning("[FAIL] Failed to pickle key '%s': %s", k, e)

    # 写入元数据
    try:
        with open(p_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        shutil.copy(p_dir / "metadata.json", f_dir / "metadata.json")
        if run_fingerprint:
            manifest = {"run_fingerprint": run_fingerprint, "phase": phase_name, "schema_version": 1}
            with open(p_dir / "cache_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.warning("[FAIL] Failed to write metadata: %s", e)

def load_phase_result(phase_name: str, modules_list: list, expected_fingerprint: str = None) -> Optional[Dict[str, Any]]:
    p_dir = get_phase_cache_dir(phase_name, "parquet", modules_list)
    meta_file = p_dir / "metadata.json"
    if expected_fingerprint:
        cache_manifest = p_dir / "cache_manifest.json"
        if not cache_manifest.exists():
            logger.warning("拒绝无运行指纹的旧缓存: %s", phase_name)
            return None
        with open(cache_manifest, "r", encoding="utf-8") as f:
            cached_fingerprint = json.load(f).get("run_fingerprint")
        if cached_fingerprint != expected_fingerprint:
            logger.warning("拒绝运行指纹不匹配的缓存: %s", phase_name)
            return None
    if not meta_file.exists():
        # logger.warning("[OP] Probe Checkpoint | [SOURCE] Disk Auditor | [RESULT] Missed! Cold start required | [SIGNIFICANCE] Cache absent")
        logger.warning("由上下文I/O模块探测检查点 | 结果: 缺失！需要冷启动 | 意义: 缓存不存在")
        return None
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    res = {}
    for k, m in metadata.items():
        t = m.get("type")
        if t == "dataframe":
            p_path = p_dir / f"{k}.parquet"
            if p_path.exists():
                res[k] = pd.read_parquet(p_path)
            else:
                f_dir = CACHE_ROOT / "feather" / f"Phase_{modules_list.index(phase_name)+1}"
                f_path = f_dir / f"{k}.feather"
                if f_path.exists():
                    df_raw = pd.read_feather(f_path)
                    idx_names = m.get("index_names", [])
                    valid_idx = [col for col in idx_names if col in df_raw.columns]
                    if valid_idx:
                        df_raw.set_index(valid_idx, inplace=True)
                    res[k] = df_raw
            continue
        if t == "tuple_dict":
            json_path = p_dir / f"{k}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    serialized_list = json.load(f)
                reconstructed = {}
                for item in serialized_list:
                    k_elements = item["k"]
                    k_types = item["t"]
                    v = item["v"]
                    reconstructed_k = []
                    for sub_k, type_name in zip(k_elements, k_types):
                        if type_name in ["Timestamp", "datetime"]:
                            reconstructed_k.append(pd.Timestamp(sub_k))
                        elif type_name in ["int", "int64"]:
                            reconstructed_k.append(int(sub_k))
                        elif type_name in ["float", "float64"]:
                            reconstructed_k.append(float(sub_k))
                        else:
                            reconstructed_k.append(str(sub_k))
                    reconstructed[tuple(reconstructed_k)] = v
                res[k] = reconstructed
            continue
        if t == "json" and (p_dir / f"{k}.json").exists():
            with open(p_dir / f"{k}.json", "r") as f:
                res[k] = json.load(f)
        elif t == "pickle" and (p_dir / f"{k}.pkl").exists():
            with open(p_dir / f"{k}.pkl", "rb") as f:
                res[k] = pickle.load(f)
    # logger.info("[OP] Hydrate Phase Context | [SOURCE] Disk Blocks | [RESULT] Inflated assets: %s | [SIGNIFICANCE] Hot injection bypasses re-computation", list(res.keys()))
    logger.info("由上下文I/O模块恢复阶段上下文 | 来源: 磁盘块 | 结果: 已膨胀资产: %s | 意义: 热注入绕过重新计算", list(res.keys()))
    return res if res else None

def save_context_snapshot(context: Dict, phase_name: str, run_ts: str, log_dir: Path):
    def _stringify_keys(obj: Any) -> Any:
        """递归将对象中所有非标准JSON字典键（如元组键）转换为字符串"""
        if isinstance(obj, dict):
            return {str(k): _stringify_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_stringify_keys(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(_stringify_keys(i) for i in obj)
        return obj

    out = {}
    for k, v in context.items():
        if k in IGNORED_KEYS:
            continue
        if isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = f"<{type(v).__name__} shape={getattr(v, 'shape', 'N/A')}>"
        else:
            out[k] = _stringify_keys(v)
    snap_file = log_dir / f"ctx_snap_{phase_name.replace('.', '_')}_{run_ts}.json"
    with open(snap_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
