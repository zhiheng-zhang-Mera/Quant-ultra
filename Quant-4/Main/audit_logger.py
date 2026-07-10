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
        self._log_file_handle = None
        self._event_count = 0
        self._console_interval = 1000  # 每隔 1000 条高频事件降噪打印一次终端，消除终端滚动带来的I/O卡顿
        logger.info("由会话流水线初始化审计日志 | 当前会话ID: %s", session_id)
        # logger.info("[OP] Instantiate Immutable Audit Core | [SOURCE] Session Orchestrator Initialization | [RESULT] Tracker bound to session ID: %s | [SIGNIFICANCE] Forms the authoritative non-repudiation log matrix for this process", session_id)
        # logger.info("[操作] 实例化不可变审计核心 | [来源] 会话编排器初始化 | [结果] 追踪器绑定会话ID: %s | [意义] 构成本次进程权威的不可否认性日志矩阵")

    def _get_file_handle(self):
        """惰性初始化并复用持久化文件句柄，避免频繁 open/close 触发操作系统磁盘系统调用瓶颈"""
        if self._log_file_handle is None:
            log_file = self.log_dir / f"audit_{self._session_id}.jsonl"
            # 采用 64KB 较大的缓冲区，进行块级增量写入
            self._log_file_handle = open(log_file, "a", encoding="utf-8", buffering=64 * 1024)
        return self._log_file_handle

    def log_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(pytz.timezone("Asia/Shanghai")).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "event_type": event_type,
            "details": details
        }
        self.events.append(entry)
        self._event_count += 1
        
        # 极限性能：直接向复用的文件句柄执行块缓冲写入
        f = self._get_file_handle()
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # 终端降噪：避免百万次级标准输出（stdout）渲染阻塞 CPU 主线程
        if self._event_count == 1 or self._event_count % self._console_interval == 0:
            logger.info("由信号流水线提交只增审计日志行 | 当前累计总行数: %d", self._event_count)
            # logger.info("[OP] Commit Append-Only Audit Line | [SOURCE] Active Signal Pipeline | [RESULT] Stream updated (Total entries: %d) | [SIGNIFICANCE] Guarantees tick-by-tick operational audit compliance with optimized I/O", self._event_count)
            # logger.info("[操作] 提交只增审计日志行 | [来源] 活跃信号流水线 | [结果] 流式日志已更新 (当前累计总行数: %d) | [意义] 在极限优化I/O的条件下保证逐笔操作具备合规审计效力", self._event_count)

    def flush(self):
        """显式强制刷写，关闭持久化文件句柄并输出全量格式化总账本"""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.flush()
            self._log_file_handle.close()
            self._log_file_handle = None
            
        out_file = self.log_dir / f"audit_full_{self._session_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
        # logger.info("[OP] Flush Memory Audit Array | [SOURCE] In-Memory Volatile Buffer Stack | [RESULT] Formatted JSON dumped to: %s | [SIGNIFICANCE] Creates permanent deterministic ledger for regulatory post-mortems", out_file.name)
        # logger.info("[操作] 刷写内存审计数组 | [来源] 内存易失性缓冲区栈 | [结果] 格式化JSON已倾倒至: %s | [意义] 为监管盘后分析创建永久确定性的账本")
        logger.info("审计日志已刷新至磁盘 | 当前会话ID: %s | 输出文件: %s", self._session_id, out_file.name)
        
    def __del__(self):
        """看门狗析构，确保在对象销毁时句柄安全关闭"""
        if hasattr(self, '_log_file_handle') and self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.close()