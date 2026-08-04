"""Self-contained bilingual HTML execution report assembled after a pipeline run."""
from __future__ import annotations
import hashlib, html, json
from datetime import datetime
from pathlib import Path
import pandas as pd

def _safe(value): return html.escape(str(value))

def _phase_proofs(run_dir):
    proofs = []
    for path in sorted(Path(run_dir).glob("Phase_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8")); item["source_file"] = path.name; proofs.append(item)
        except (OSError, json.JSONDecodeError): continue
    return sorted(proofs, key=lambda x: int(x.get("phase", "Phase_0").split(".")[0].split("_")[1]))

def _candidate_rows(context, limit=12):
    frame = context.get("investment_candidates")
    if not isinstance(frame, pd.DataFrame) or frame.empty: return []
    columns = ["symbol", "advisory_mode", "suggested_weight", "entry_price_low", "entry_price_high", "take_profit_pct", "stop_loss_pct", "dominant_selection_method", "dominant_entry_method", "dominant_holding_method", "dominant_take_profit_method"]
    return frame[[c for c in columns if c in frame]].head(limit).to_dict("records")

def _table(mapping):
    return "<table><tbody>" + "".join(f"<tr><th>{_safe(k)}</th><td>{_safe(v)}</td></tr>" for k, v in mapping.items()) + "</tbody></table>"

def generate_execute_report(run_dir, context, requested_phases):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    proofs = _phase_proofs(run_dir); passed = sum(p.get("status") == "PASS" for p in proofs)
    elapsed = sum(float(p.get("elapsed_seconds", 0)) for p in proofs)
    costs = context.get("transaction_costs", {}) if isinstance(context.get("transaction_costs"), dict) else {}
    compute = context.get("compute_audit", {}) if isinstance(context.get("compute_audit"), dict) else {}
    payload = {"generated_at": datetime.now().astimezone().isoformat(), "run_id": run_dir.name, "git_hash": context.get("run_metadata", {}).get("git_hash", "UNKNOWN"), "requested_phases": list(requested_phases), "phase_count": len(proofs), "passed_phases": passed, "elapsed_seconds": elapsed, "cio_decision": context.get("cio_decision", "NOT_RUN"), "audit_passed": context.get("audit_passed"), "recon_passed": context.get("recon_passed"), "reconciliation_mae": context.get("reconciliation_mae"), "final_nav": context.get("final_nav"), "transaction_costs": costs, "compute_audit": compute, "phases": proofs, "candidates": _candidate_rows(context), "limitations": ["Engineering completion does not establish future investment performance.", "A HOLD_FOR_REVIEW or OBSERVATION_ONLY result prohibits treating candidates as executable instructions.", "Third-party data availability and source authenticity remain run-specific evidence boundaries."]}
    data_path = run_dir / "execute_report_data.json"; data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    governance_class = "ok" if payload["cio_decision"] not in ("HOLD_FOR_REVIEW", "NOT_RUN") and payload["audit_passed"] is not False and payload["recon_passed"] is not False else "warn"
    phase_cards = []; max_elapsed = max([float(p.get("elapsed_seconds", 0)) for p in proofs] or [1])
    for proof in proofs:
        seconds = float(proof.get("elapsed_seconds", 0)); width = max(2, seconds / max_elapsed * 100)
        summary = "".join(f"<li><code>{_safe(k)}</code>: {_safe(v.get('summary', ''))}</li>" for k, v in list(proof.get("outputs", {}).items())[:12])
        phase_cards.append(f"<article class='phase'><header><span class='badge'>{_safe(proof.get('status'))}</span><h3>{_safe(proof.get('title_zh', proof.get('phase')))}<small>{_safe(proof.get('title_en',''))}</small></h3></header><p>{_safe(proof.get('interpretation_zh',''))}</p><p class='muted'>{_safe(proof.get('interpretation_en',''))}</p><div class='bar'><i style='width:{width:.1f}%'></i></div><p class='meta'>{seconds:.3f}s · cache={_safe(proof.get('cache_hit'))} · SHA-256 {_safe(proof.get('report_sha256','')[:16])}</p><details><summary>输出证据 / Output evidence</summary><ul>{summary}</ul></details></article>")
    candidates = payload["candidates"]; candidate_table = "<p>无候选或 Phase 11 未运行。 / No candidates or Phase 11 was not run.</p>"
    if candidates:
        columns = list(candidates[0]); candidate_table = "<table><thead><tr>" + "".join(f"<th>{_safe(c)}</th>" for c in columns) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_safe(row.get(c,''))}</td>" for c in columns) + "</tr>" for row in candidates) + "</tbody></table>"
    css = ":root{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#667085;--line:#d9e0ea;--accent:#2458d3;--ok:#087443;--warn:#a04400}@media(prefers-color-scheme:dark){:root{--bg:#0d1320;--card:#151d2d;--text:#ecf1fa;--muted:#a9b4c8;--line:#303b50;--accent:#7da2ff;--ok:#54d69a;--warn:#ffad66}}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:34px;margin:.2em 0}h2{margin-top:38px;border-bottom:1px solid var(--line);padding-bottom:8px}small,.muted,.meta{display:block;color:var(--muted)}.hero,.card,.phase{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 5px 20px #0000000b}.hero{border-left:7px solid var(--accent)}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}.metric strong{font-size:25px;display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.phase header{display:flex;gap:12px;align-items:start}.phase h3{margin:0}.badge{font-weight:700;color:var(--ok)}.bar{height:7px;background:var(--line);border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}table{width:100%;border-collapse:collapse;display:block;overflow:auto}th,td{border-bottom:1px solid var(--line);padding:9px;text-align:left;white-space:nowrap}code{font-size:12px}.warn{color:var(--warn);font-weight:700}details{margin-top:10px}footer{margin:40px 0;color:var(--muted)}"
    report_path = run_dir / "execute_report.html"
    report_path.write_text(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='light dark'><title>Quant-Ultra Execute Report {run_dir.name}</title><style>{css}</style></head><body><main><section class='hero'><p>Quant-Ultra · Run {_safe(run_dir.name)}</p><h1>执行报告 / Execute Report</h1><p class='{governance_class}'>技术结论 / Technical conclusion: {passed}/{len(proofs)} 阶段通过；CIO={_safe(payload['cio_decision'])}；Audit={_safe(payload['audit_passed'])}；Reconciliation={_safe(payload['recon_passed'])}。</p><p>阶段执行完成仅证明工程流程和合约状态；治理失败时不得执行候选交易。 / Phase completion proves engineering and contract status only; failed governance blocks execution.</p></section><section class='metrics'><div class='card metric'><span>阶段 / Phases</span><strong>{passed}/{len(proofs)}</strong></div><div class='card metric'><span>累计耗时 / Elapsed</span><strong>{elapsed:.1f}s</strong></div><div class='card metric'><span>最终净值 / Final NAV</span><strong>{_safe(payload['final_nav'])}</strong></div><div class='card metric'><span>交易磨损 / Friction</span><strong>{_safe(costs.get('total','N/A'))}</strong></div><div class='card metric'><span>对账 MAE</span><strong>{_safe(payload['reconciliation_mae'])}</strong></div></section><h2>计算环境与任务分配 / Compute Environment</h2><div class='grid'><div class='card'>{_table(compute.get('hardware',{}))}</div><div class='card'>{_table(compute.get('connectivity',{}))}{_table(compute.get('resource_plan',{}))}</div></div><h2>阶段证据与解释 / Phase Evidence</h2><div class='grid'>{''.join(phase_cards)}</div><h2>风险、治理与成本 / Risk, Governance and Costs</h2><div class='grid'><div class='card'>{_table({'CIO decision':payload['cio_decision'],'audit_passed':payload['audit_passed'],'recon_passed':payload['recon_passed'],'reconciliation_mae':payload['reconciliation_mae'],'final_nav':payload['final_nav']})}</div><div class='card'>{_table(costs or {'costs':'NOT_AVAILABLE'})}</div></div><h2>候选及四阶段主导方法 / Candidates and Dominant Methods</h2><div class='card'>{candidate_table}</div><h2>边界与后续动作 / Limitations and Next Steps</h2><div class='card'><ul>{''.join(f'<li>{_safe(x)}</li>' for x in payload['limitations'])}</ul><p>下一步：先解决未通过的审计或对账门禁，再评估参数或实盘准备度。 / Resolve failed audit or reconciliation gates before evaluating parameter changes or implementation readiness.</p></div><footer>Generated {_safe(payload['generated_at'])} · Git {_safe(payload['git_hash'])} · Data SHA-256 {payload_hash}</footer></main></body></html>", encoding="utf-8")
    return report_path, data_path
