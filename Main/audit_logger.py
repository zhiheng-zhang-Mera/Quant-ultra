"""
Quant-Ultra Flow - Audit Logger
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pytz

class AuditLogger:
    def __init__(self, log_dir: Path, session_id: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict] = []
        self._session_id = session_id

    def log_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(pytz.timezone("Asia/Shanghai")).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "event_type": event_type,
            "details": details
        }
        self.events.append(entry)
        with open(self.log_dir / f"audit_{self._session_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def flush(self):
        with open(self.log_dir / f"audit_full_{self._session_id}.json", "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)