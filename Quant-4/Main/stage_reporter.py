"""Bilingual, self-verifying report emitted for every pipeline phase."""
from __future__ import annotations
import hashlib, json, platform, time
from pathlib import Path
from typing import Any
import pandas as pd

PHASE_GUIDE = {
    1: ("数据底座与流动性筛选", "Data foundation and liquidity screening", "确认标的、价格、存续状态和容量边界可用于后续计算。", "Confirms that universe, prices, survival status, and capacity limits are usable downstream."),
    2: ("时间切片与隔离", "Temporal slicing and isolation", "建立训练、验证、测试和禁运窗口，降低时间泄漏风险。", "Creates train, validation, test, and embargo windows to reduce temporal leakage."),
    3: ("PIT 特征与另类数据", "PIT features and alternative data", "将价格、新闻、论坛情绪与资金池变化按当时可见时间对齐。", "Aligns prices, news, forum sentiment, and capital-pool changes to information available at the time."),
    4: ("标签与样本权重", "Labels and sample weights", "生成预测目标并校正样本重要性，避免重复事件被过度计权。", "Builds prediction targets and adjusts sample importance so repeated events are not overweighted."),
    5: ("模型训练与校准", "Model training and calibration", "训练方向与区间模型，并输出校准误差和可复算参数。", "Trains direction and interval models and emits calibration errors and reproducible parameters."),
    6: ("风险预算与仓位优化", "Risk budgeting and position optimization", "在风险、成本、集中度、现金和换手约束下求解目标仓位。", "Solves target weights under risk, cost, concentration, cash, and turnover constraints."),
    7: ("交易状态机回测", "Execution FSM backtest", "模拟可交易性、整手成交、滑点、税费、管理费和账户净值。", "Simulates tradability, board lots, slippage, taxes, management fees, and account NAV."),
    8: ("审计与压力测试", "Audit and stress testing", "检验覆盖率、容量、流动性冲击、极端情景和统计稳健性。", "Tests coverage, capacity, liquidity impact, stress scenarios, and statistical robustness."),
    9: ("影子对账与 MLOps", "Shadow reconciliation and MLOps", "比较目标与执行仓位，监控漂移并触发交易门禁。", "Compares target and executed positions, monitors drift, and activates trading gates."),
    10: ("CIO 治理报告", "CIO governance report", "汇总证据并给出继续、观察或人工复核结论。", "Summarizes evidence and returns proceed, observe, or manual-review decisions."),
    11: ("交互式投顾观察", "Interactive advisory observation", "生成含买点、仓位、止盈止损和成本的候选清单；治理失败时禁止执行。", "Produces candidates with entries, weights, exits, and costs; execution remains blocked when governance fails."),
}

TERMS = {
    "PIT": "Point-in-Time / 时点可见信息",
    "NAV": "Net Asset Value / 账户净值",
    "MAE": "Mean Absolute Error / 平均绝对误差",
    "PSI": "Population Stability Index / 群体稳定性指数",
    "VaR": "Value at Risk / 风险价值",
    "CVaR": "Conditional Value at Risk / 条件风险价值",
    "ADV20": "20-day Average Daily Value / 20日平均成交额",
    "HOLD_FOR_REVIEW": "人工复核前禁止执行 / No execution before manual review",
}

class StageReporter:
    def __init__(self, root: Path, run_id: str, git_hash: str):
        self.root = Path(root) / "runs" / run_id; self.root.mkdir(parents=True, exist_ok=True)
        self.git_hash, self.started = git_hash, {}

    def start(self, phase: str): self.started[phase] = time.perf_counter()

    def finish(self, phase: str, result: dict, contract_valid: bool, cache_hit: bool = False) -> Path:
        elapsed = time.perf_counter() - self.started.get(phase, time.perf_counter())
        summary = {k: self._describe(v) for k, v in result.items() if not k.startswith("_")}
        number = int(phase.split(".")[0].split("_")[1]); zh, en, interpretation_zh, interpretation_en = PHASE_GUIDE[number]
        proof = {"phase": phase, "title_zh": zh, "title_en": en, "status": "PASS" if contract_valid else "FAIL", "contract_valid": contract_valid, "cache_hit": cache_hit, "elapsed_seconds": round(elapsed, 6), "git_hash": self.git_hash, "python": platform.python_version(), "interpretation_zh": interpretation_zh, "interpretation_en": interpretation_en, "outputs": summary, "terms": TERMS}
        payload = json.dumps(proof, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        proof["report_sha256"] = hashlib.sha256(payload).hexdigest()
        slug = phase.replace(".", "_"); json_path = self.root / f"{slug}.json"
        json_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"# Phase {number}: {zh} / {en}", "", "## 结论解读 / Conclusion", "", interpretation_zh, "", interpretation_en, "", "## 运行证据 / Run Evidence", "", f"- 状态 / Status: **{proof['status']}**", f"- 合约验证 / Contract valid: `{contract_valid}`", f"- 缓存命中 / Cache hit: `{cache_hit}`", f"- 耗时 / Elapsed: `{proof['elapsed_seconds']}` seconds", f"- Git: `{self.git_hash}`", f"- 证据哈希 / Evidence SHA-256: `{proof['report_sha256']}`", "", "## 输出摘要 / Output Summary", "", "| 输出 / Output | 类型 / Type | 摘要 / Summary |", "|---|---|---|"]
        lines.extend(f"| `{key}` | {item['type']} | {item['summary']} |" for key, item in summary.items())
        lines += ["", "## 术语 / Glossary", ""] + [f"- **{key}**: {value}" for key, value in TERMS.items()] + ["", "> 工程验证不等于投资收益保证。 / Engineering validation is not a guarantee of investment performance."]
        md_path = self.root / f"{slug}.md"; md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return md_path

    @staticmethod
    def _describe(value: Any) -> dict:
        if isinstance(value, pd.DataFrame):
            digest = hashlib.sha256(value.to_csv(index=True).encode("utf-8")).hexdigest()[:16]
            return {"type": "DataFrame", "summary": f"shape={value.shape}, columns={list(value.columns)[:8]}, sha256={digest}"}
        if isinstance(value, pd.Series): return {"type": "Series", "summary": f"length={len(value)}, nulls={int(value.isna().sum())}"}
        if isinstance(value, dict): return {"type": "dict", "summary": f"keys={list(value)[:12]}"}
        return {"type": type(value).__name__, "summary": str(value)[:160].replace("|", "\\|")}
