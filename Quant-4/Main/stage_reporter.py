"""Notebook-like, self-verifying report emitted for every pipeline phase."""
from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any
import pandas as pd

class StageReporter:
    def __init__(self, root: Path, run_id: str, git_hash: str):
        self.root = Path(root) / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.git_hash = git_hash
        self.started = {}

    def start(self, phase: str): self.started[phase] = time.perf_counter()

    def finish(self, phase: str, result: dict, contract_valid: bool, cache_hit: bool = False) -> Path:
        elapsed = time.perf_counter() - self.started.get(phase, time.perf_counter())
        summary = {k: self._describe(v) for k, v in result.items() if not k.startswith("_")}
        proof = {"phase": phase, "status": "PASS" if contract_valid else "FAIL", "contract_valid": contract_valid, "cache_hit": cache_hit, "elapsed_seconds": round(elapsed, 6), "git_hash": self.git_hash, "python": platform.python_version(), "outputs": summary}
        payload = json.dumps(proof, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        proof["report_sha256"] = hashlib.sha256(payload).hexdigest()
        slug = phase.replace(".", "_")
        json_path = self.root / f"{slug}.json"
        json_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"# {phase} 阶段运行报告", "", f"- 状态：**{proof['status']}**", f"- 合约验证：`{contract_valid}`", f"- 缓存命中：`{cache_hit}`", f"- 耗时：`{proof['elapsed_seconds']}` 秒", f"- Git：`{self.git_hash}`", f"- 证据哈希：`{proof['report_sha256']}`", "", "## 输出摘要", "", "| 输出 | 类型 | 摘要 |", "|---|---|---|"]
        for key, item in summary.items(): lines.append(f"| `{key}` | {item['type']} | {item['summary']} |")
        md_path = self.root / f"{slug}.md"
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return md_path

    @staticmethod
    def _describe(value: Any) -> dict:
        if isinstance(value, pd.DataFrame):
            digest = hashlib.sha256(value.to_csv(index=True).encode("utf-8")).hexdigest()[:16]
            return {"type": "DataFrame", "summary": f"shape={value.shape}, columns={list(value.columns)[:8]}, sha256={digest}"}
        if isinstance(value, pd.Series): return {"type": "Series", "summary": f"length={len(value)}, nulls={int(value.isna().sum())}"}
        if isinstance(value, dict): return {"type": "dict", "summary": f"keys={list(value)[:12]}"}
        return {"type": type(value).__name__, "summary": str(value)[:160].replace("|", "\\|")}

