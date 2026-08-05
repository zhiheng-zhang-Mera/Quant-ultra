"""Self-contained bilingual HTML execution report assembled after a pipeline run."""
from __future__ import annotations
import hashlib, html, json
from datetime import datetime
from pathlib import Path
import pandas as pd

FIELD_LABELS = {"symbol": "证券代码 / Symbol", "advisory_mode": "建议模式 / Advisory mode", "suggested_weight": "suggested weight / 建议仓位", "entry_price_low": "建议入场价下限 / Entry price low", "entry_price_high": "建议入场价上限 / Entry price high", "take_profit_pct": "目标止盈率(%) / Target take-profit (%)", "stop_loss_pct": "止损率(%) / Stop-loss (%)", "dominant_selection_method": "主导选股方法 / Selection method", "dominant_entry_method": "主导入场方法 / Entry method", "dominant_holding_method": "主导持有方法 / Holding method", "dominant_take_profit_method": "主导止盈方法 / Take-profit method", "audit_passed": "审计通过 / Audit passed", "recon_passed": "对账通过 / Reconciliation passed", "reconciliation_mae": "对账平均绝对误差 / Reconciliation MAE", "final_nav": "最终净值 / Final NAV"}
PERCENT_FIELDS = {"suggested_weight", "take_profit_pct", "stop_loss_pct", "reconciliation_mae"}
def _safe(x): return html.escape(str(x))
def _label(key): return FIELD_LABELS.get(key, str(key).replace("_", " "))
def _display(key, value):
    if key == "symbol": return value
    if key in PERCENT_FIELDS and isinstance(value, (int, float)): return f"{value * 100:.2f}%"
    return f"{value:.6g}" if isinstance(value, float) else value
def _proofs(run_dir):
    rows=[]
    for path in Path(run_dir).glob("Phase_*.json"):
        try: rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError,json.JSONDecodeError): pass
    return sorted(rows,key=lambda x:int(x.get("phase","Phase_0").split(".")[0].split("_")[1]))
def _candidates(context):
    frame=context.get("investment_candidates")
    cols=["symbol","advisory_mode","suggested_weight","entry_price_low","entry_price_high","take_profit_pct","stop_loss_pct","dominant_selection_method","dominant_entry_method","dominant_holding_method","dominant_take_profit_method"]
    if not isinstance(frame,pd.DataFrame) or frame.empty: return []
    rows=frame[[c for c in cols if c in frame]].head(12).to_dict("records")
    names=context.get("asset_names", {})
    for row in rows:
        symbol=str(row.get("symbol", "")).upper()
        row["symbol"] = f"{symbol} / {names.get(symbol, row.get('asset_name', '名称数据不可用 / Name data unavailable'))}"
    return rows
def _table(rows): return "<table>"+"".join(f"<tr><th>{_safe(_label(k))}</th><td>{_safe(_display(k,v))}</td></tr>" for k,v in rows.items())+"</table>"
def generate_execute_report(run_dir, context, requested_phases):
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True); proofs=_proofs(run_dir); candidates=_candidates(context); costs=context.get("transaction_costs",{}) if isinstance(context.get("transaction_costs"),dict) else {}; compute=context.get("compute_audit",{}) if isinstance(context.get("compute_audit"),dict) else {}
    payload={"generated_at":datetime.now().astimezone().isoformat(),"run_id":run_dir.name,"git_hash":context.get("run_metadata",{}).get("git_hash","UNKNOWN"),"requested_phases":list(requested_phases),"phase_count":len(proofs),"passed_phases":sum(p.get("status")=="PASS" for p in proofs),"elapsed_seconds":sum(float(p.get("elapsed_seconds",0)) for p in proofs),"cio_decision":context.get("cio_decision","NOT_RUN"),"audit_passed":context.get("audit_passed"),"recon_passed":context.get("recon_passed"),"reconciliation_mae":context.get("reconciliation_mae"),"final_nav":context.get("final_nav"),"transaction_costs":costs,"compute_audit":compute,"phases":proofs,"candidates":candidates,"limitations":["Engineering completion does not establish future investment performance.","A HOLD_FOR_REVIEW or OBSERVATION_ONLY result prohibits treating candidates as executable instructions.","Third-party data availability and source authenticity remain run-specific evidence boundaries."]}
    data_path=run_dir/"execute_report_data.json"; data_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); digest=hashlib.sha256(data_path.read_bytes()).hexdigest(); elapsed=payload["elapsed_seconds"]
    max_seconds=max([float(p.get("elapsed_seconds",0)) for p in proofs]or[1]); cards=[]
    for p in proofs:
        seconds=float(p.get("elapsed_seconds",0)); evidence="".join(f"<li><b>{_safe(_label(k))}</b>: {_safe(v.get('summary',''))}</li>" for k,v in list(p.get("outputs",{}).items())[:12]); cards.append(f"<article><h3>{_safe(p.get('title_zh',p.get('phase')))}<small>{_safe(p.get('title_en',''))}</small></h3><p>{_safe(p.get('interpretation_zh',''))}</p><p>{_safe(p.get('interpretation_en',''))}</p><div class='bar'><i style='width:{max(2,seconds/max_seconds*100):.1f}%'></i></div><small>{seconds:.3f}s · cache={_safe(p.get('cache_hit'))}</small><details><summary>输出证据 / Output evidence</summary><ul>{evidence}</ul></details></article>")
    candidate_table="<p>无候选，或 Phase 11 未运行。 / No candidates or Phase 11 was not run.</p>"
    if candidates:
        cols=list(candidates[0]); candidate_table="<table><thead><tr>"+"".join(f"<th>{_safe(_label(c))}</th>" for c in cols)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{_safe(_display(c,r.get(c,'')))}</td>" for c in cols)+"</tr>" for r in candidates)+"</tbody></table>"
    conclusion="可进入分析环节 / eligible for analysis" if payload["cio_decision"]=="ELIGIBLE_FOR_PHASE_11" and payload["audit_passed"] and payload["recon_passed"] else "治理门禁阻断执行 / governance blocks execution"
    html_text=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Quant-Ultra Execute Report</title><style>body{{background:#f5f7fb;color:#172033;font:15px system-ui;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}section,article{{background:#fff;border:1px solid #d9e0ea;border-radius:12px;padding:18px;margin:14px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}small{{display:block;color:#667085}}table{{width:100%;border-collapse:collapse;display:block;overflow:auto}}th,td{{padding:8px;border-bottom:1px solid #d9e0ea;text-align:left;white-space:nowrap}}.bar{{height:7px;background:#d9e0ea}}.bar i{{display:block;height:100%;background:#2458d3}}</style></head><body><main><section><h1>执行报告 / Execute Report</h1><p><b>技术结论 / Technical conclusion:</b> {payload['passed_phases']}/{len(proofs)} 阶段通过；{_safe(conclusion)}。</p><p>阶段完成只证明工程流程与合约状态；治理失败时不得执行候选交易。 / Phase completion proves engineering and contract status only; failed governance blocks execution.</p></section><section class='grid'><article>{_table({'final_nav':payload['final_nav'],'reconciliation_mae':payload['reconciliation_mae'],'audit_passed':payload['audit_passed'],'recon_passed':payload['recon_passed'],'cio_decision':payload['cio_decision'],'elapsed_seconds':elapsed})}</article><article>{_table(costs or {'costs':'NOT_AVAILABLE'})}</article></section><h2>计算环境 / Compute Environment</h2><section>{_table(compute.get('resource_plan',{}))}</section><h2>阶段证据 / Phase Evidence</h2><div class='grid'>{''.join(cards)}</div><h2>候选与四阶段主导方法 / Candidates and Dominant Methods</h2><section>{candidate_table}</section><section><h2>边界 / Limitations</h2><ul>{''.join(f'<li>{_safe(x)}</li>' for x in payload['limitations'])}</ul><small>Generated {_safe(payload['generated_at'])} · Git {_safe(payload['git_hash'])} · Data SHA-256 {digest}</small></section></main></body></html>"""
    report_path=run_dir/"execute_report.html"; report_path.write_text(html_text,encoding="utf-8"); return report_path,data_path
