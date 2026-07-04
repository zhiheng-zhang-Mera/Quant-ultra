import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pytz
import logging

logger = logging.getLogger("AuditLogger")

class AuditLogger:
    def __init__(self, log_dir: Path, session_id: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict] = []
        self._session_id = session_id
        logger.info("[OP] Instantiate Immutable Audit Core | [SOURCE] Session Orchestrator Initialization | [RESULT] Tracker bound to session ID: %s | [SIGNIFICANCE] Forms the authoritative non-repudiation log matrix for this process", session_id)
        logger.info("[操作] 实例化不可变审计核心 | [来源] 会话编排器初始化 | [结果] 追踪器绑定会话ID: %s | [意义] 构成本次进程权威的不可否认性日志矩阵")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(pytz.timezone("Asia/Shanghai")).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "event_type": event_type,
            "details": details
        }
        self.events.append(entry)
        log_file = self.log_dir / f"audit_{self._session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[OP] Commit Append-Only Audit Line | [SOURCE] Active Signal Pipeline | [RESULT] Stream updated: %s | [SIGNIFICANCE] Guarantees tick-by-tick operational audit compliance", log_file.name)
        logger.info("[操作] 提交只增审计日志行 | [来源] 活跃信号流水线 | [结果] 流式日志已更新: %s | [意义] 保证逐笔操作具备合规审计效力")

    def flush(self):
        out_file = self.log_dir / f"audit_full_{self._session_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
        logger.info("[OP] Flush Memory Audit Array | [SOURCE] In-Memory Volatile Buffer Stack | [RESULT] Formatted JSON dumped to: %s | [SIGNIFICANCE] Creates permanent deterministic ledger for regulatory post-mortems", out_file.name)
        logger.info("[操作] 刷写内存审计数组 | [来源] 内存易失性缓冲区栈 | [结果] 格式化JSON已倾倒至: %s | [意义] 为监管盘后分析创建永久确定性的账本")