import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT))
from Main.data_quality import validate_ohlcv
from Main.fast_math import log_returns
from Main.portfolio_analytics import holding_advice, metrics, nearest_psd, recommendation, risk_parity_weights, technical_snapshot
from analyze_cn_asset import normalize
from Main.investment_advisor import write_candidate_report
from Main.walk_forward_backtest import walk_forward_backtest
from Main.orchestration_guard import build_run_fingerprint, validate_orchestration
from Main.parameter_governance import validate_parameter_proposal
from Main.schema_contracts import PHASE_DEPENDENCIES, PHASE_INPUT_SCHEMA, PHASE_MODULES, PHASE_OUTPUT_SCHEMA, resolve_phase_name, validate_phase_contract
from Main.trading_costs import explicit_order_fees, round_trip_friction_rate
from Main.stage_reporter import StageReporter
from Phase_3.alternative_data import build_alternative_signals, enhance_sentiment_with_local_llm, score_text
from Main.decision_chain import STAGE_METHODS, TRANSITIONS, four_stage_decision_chain
from Main.distributed_compute import HardwareProfile, apply_resource_plan, build_resource_plan
from Main.execute_report import generate_execute_report
from Main.allocation_constraints import apply_allocation_cap

def test_data_quality_proves_valid_and_rejects_bad():
    good=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=3),"open":[1,2,3],"high":[2,3,4],"low":[.5,1,2],"close":[1.5,2.5,3.5],"volume":[1,2,3]})
    assert validate_ohlcv(good)["valid"]
    bad=good.copy(); bad.loc[0,"high"]=0
    assert not validate_ohlcv(bad)["valid"]

def test_fast_math_and_metrics():
    p=np.array([100.,110.,121.]); assert np.allclose(log_returns(p),np.log([1.1,1.1]))
    assert metrics(pd.Series(np.linspace(100,130,100)))["observations"]==99

def test_risk_parity_invariants():
    cov=nearest_psd(np.array([[.04,.01],[.01,.09]])); w=risk_parity_weights(cov)
    assert np.isclose(w.sum(),1) and np.all(w>=0)

def test_stock_and_etf_normalization():
    assert normalize("600519","stock")=="600519.SH"
    assert normalize("510300","etf")=="510300.SH"

def _market_frame():
    close=np.linspace(10,14,100)
    return pd.DataFrame({"date":pd.date_range("2024-01-01",periods=100),"open":close*.995,"high":close*1.02,"low":close*.98,"close":close,"volume":np.full(100,100000)})

def test_recommendation_has_required_four_outputs():
    rec=recommendation(_market_frame(),model_weight=.08)
    assert rec["qualified"]
    assert 0 < rec["entry_price_low"] <= rec["entry_price_high"]
    assert 0 <= rec["suggested_weight"] <= .08
    assert .03 <= rec["take_profit_pct"] <= .12
    assert rec["net_take_profit_pct"] >= rec["minimum_net_profit_pct"]
    assert rec["entry_price_high"] <= rec["last_price"]

def test_holding_advice_reconciles_cash_position_and_action():
    result=holding_advice(_market_frame(),100000,1000,10,model_weight=.08)
    assert np.isclose(result["market_value"],1000*result["last_price"])
    assert result["target_quantity"] % 100 == 0
    assert result["action"] in {"分批止盈","减仓","按批次买入","减仓至目标","持有观察"}
    assert result["action_reason"]
    assert result["estimated_sell_fees_at_take_profit"] > 0

def test_cost_model_applies_minimum_commission_and_sell_stamp_tax():
    buy = explicit_order_fees(1000, "buy", symbol="600519.SH")
    sell = explicit_order_fees(1000, "sell", symbol="600519.SH")
    etf_sell = explicit_order_fees(1000, "sell", symbol="510300.SH")
    assert buy["commission"] == 5.0
    assert sell["stamp_tax"] > 0
    assert etf_sell["stamp_tax"] == 0
    assert round_trip_friction_rate(10000, symbol="510300.SH", holding_days=20) > 0

def test_alternative_data_is_pit_and_reports_missing_optional_sources(tmp_path):
    assert score_text("增长 回购 bullish") > 0
    prices = _market_frame().rename(columns={"date": "unused"})
    context = {"assets": ["600519.SH"], "asset_ohlcv": {"600519.SH": prices}, "trading_days_dt": [pd.Timestamp("2026-08-04")], "config": {}}
    result = build_alternative_signals(context)
    assert result["alternative_data_evidence"]["future_records_excluded"]
    assert result["alternative_data_evidence"]["news"]["status"] == "MISSING_OPTIONAL_SOURCE"
    assert np.isfinite(result["alternative_signals"].loc[0, "alternative_signal"])

def test_local_llm_sentiment_is_bounded_and_optional():
    class FakeClient:
        def list_models(self): return ["local-test"]
        def generate(self, model, prompt):
            assert model == "local-test"
            return '{"results":[{"id":"news:0","score":0.8,"confidence":0.9}]}'
    frame = pd.DataFrame([{"published_at": pd.Timestamp("2026-08-01", tz="UTC"), "symbol": "600519.SH", "text": "增长", "sentiment": 1.0}])
    enhanced, evidence = enhance_sentiment_with_local_llm({"news": frame, "forum": frame.iloc[:0]}, {"local_llm_model": "local-test", "local_llm_max_records_total": 1}, client=FakeClient())
    assert evidence["status"] == "ANALYZED"
    assert enhanced["news"].loc[0, "effective_sentiment"] == 0.8

def test_local_llm_missing_model_falls_back_without_failure():
    class MissingClient:
        def list_models(self): return []
    frame = pd.DataFrame(columns=["published_at", "symbol", "text", "sentiment"])
    unchanged, evidence = enhance_sentiment_with_local_llm({"news": frame}, {"local_llm_model": "absent"}, client=MissingClient())
    assert evidence["status"] == "MODEL_NOT_FOUND" and evidence["fallback_used"]
    assert unchanged["news"].empty

def test_stage_report_is_bilingual_and_interpreted(tmp_path):
    reporter = StageReporter(tmp_path, "run", "abc123")
    reporter.start("Phase_3.step3_pit_setup")
    report = reporter.finish("Phase_3.step3_pit_setup", {"alternative_signals": pd.DataFrame({"x": [1]})}, True)
    text = report.read_text(encoding="utf-8")
    assert "结论解读 / Conclusion" in text
    assert "术语 / Glossary" in text
    assert "PIT features and alternative data" in text

def test_four_stage_chain_has_multiple_experts_and_normalized_dynamic_weights():
    frame = _market_frame(); snap = technical_snapshot(frame); perf = metrics(frame["close"])
    chain = four_stage_decision_chain(frame, snap, perf, 0.002, alternative_signal=0.4)
    for stage in ("selection", "entry", "holding", "take_profit"):
        assert len(STAGE_METHODS[stage]) >= 8
        assert np.isclose(sum(chain[stage]["method_weights"].values()), 1.0)
        assert -1 <= chain[stage]["nonlinear_score"] <= 1

def test_dominant_method_directly_changes_next_stage_prior():
    frame = _market_frame(); chain = four_stage_decision_chain(frame, technical_snapshot(frame), metrics(frame["close"]), 0.002)
    for current, following in (("selection", "entry"), ("entry", "holding"), ("holding", "take_profit")):
        dominant = chain[current]["dominant_method"]
        assert chain[following]["inherited_from"] == dominant
        compatible = TRANSITIONS.get(dominant, set()) & set(STAGE_METHODS[following])
        assert compatible
        assert all(chain[following]["transition_boost"][method] > 0 for method in compatible)

def test_distributed_plan_respects_hardware_memory_and_network_caps():
    profile = HardwareProfile(32, 16, 6.2, 32, 100, [], "test")
    plan = build_resource_plan(profile, {"distributed_cpu_worker_cap": 12, "distributed_io_worker_cap": 20}, {"market_https": False})
    assert plan["cpu_workers"] == 4
    assert plan["download_workers"] == 1
    assert plan["optimization_workers"] <= plan["io_workers"]
    merged = apply_resource_plan({"lgb_params": {"deterministic": True}}, {"resource_plan": plan})
    assert merged["lgb_params"]["num_threads"] == plan["model_threads"]

def test_user_worker_override_is_preserved_by_resource_plan():
    profile = HardwareProfile(8, 4, 16, 32, 100, [], "test")
    plan = build_resource_plan(profile, {"download_workers": 2, "data_load_workers": 3}, {"market_https": True})
    assert plan["download_workers"] == 2 and plan["data_load_workers"] == 3

def test_execute_report_integrates_phase_evidence_and_governance(tmp_path):
    reporter = StageReporter(tmp_path, "run", "abc123")
    reporter.start("Phase_1.step1_data_foundation"); reporter.finish("Phase_1.step1_data_foundation", {"assets": ["A"]}, True)
    context = {"run_metadata": {"git_hash": "abc123"}, "cio_decision": "HOLD_FOR_REVIEW", "audit_passed": False, "recon_passed": False, "final_nav": 99.0, "transaction_costs": {"total": 1.25}, "compute_audit": {"hardware": {"logical_cpu": 8}, "connectivity": {"market_https": True}, "resource_plan": {"cpu_workers": 4}}}
    report, data = generate_execute_report(reporter.root, context, ["Phase_1.step1_data_foundation"])
    rendered = report.read_text(encoding="utf-8")
    assert "执行报告 / Execute Report" in rendered and "HOLD_FOR_REVIEW" in rendered
    assert "Phase Evidence" in rendered and "Data SHA-256" in rendered
    assert json.loads(data.read_text(encoding="utf-8"))["passed_phases"] == 1

def test_candidate_report_is_readable_and_hashed(tmp_path):
    rec=recommendation(_market_frame(),model_weight=.08)
    frame=pd.DataFrame([{"symbol":"510300.SH","asset_type":"ETF",**rec}])
    md,csv=write_candidate_report(frame,tmp_path)
    assert md.exists() and csv.exists()
    content=md.read_text(encoding="utf-8")
    assert "510300.SH" in content and "CSV SHA-256" in content

def _backtest_frame(rows=620):
    rng=np.random.default_rng(42)
    close=100*np.exp(np.cumsum(rng.normal(.0003,.01,rows)))
    open_=close*np.exp(rng.normal(0,.002,rows))
    return pd.DataFrame({"date":pd.bdate_range("2020-01-01",periods=rows),"open":open_,"close":close})

def _small_grid():
    return {"fast_window":[5,10],"slow_window":[20,40],"vol_window":[10,20],"target_vol":[.10,.15]}

def test_walk_forward_proves_temporal_order_and_frozen_parameters():
    result=walk_forward_backtest(_backtest_frame(),_small_grid(),train_size=180,test_size=60,embargo=5)
    assert result["summary"]["lookahead_audit_passed"]
    assert (pd.to_datetime(result["returns"]["signal_time"]) < pd.to_datetime(result["returns"]["execution_time"])).all()
    assert result["audit"]["parameters_frozen"].all()
    assert (pd.to_datetime(result["folds"]["train_end"]) < pd.to_datetime(result["folds"]["test_start"])).all()
    assert np.allclose(result["returns"]["turnover"], result["returns"]["weight"].diff().abs().fillna(result["returns"]["weight"].abs()))
    assert "benchmark_annual_return" in result["summary"]
    assert 0 <= result["summary"]["positive_fold_ratio"] <= 1

def test_walk_forward_rejects_negative_costs():
    try:
        walk_forward_backtest(_backtest_frame(),_small_grid(),train_size=180,test_size=60,embargo=5,fee_rate=-0.001)
    except ValueError:
        pass
    else:
        raise AssertionError("negative trading costs must be rejected")

def test_adaptive_parameter_iteration_advances_only_on_new_data(tmp_path):
    from Main.adaptive_parameter_state import prepare_iteration, finalize_iteration
    base=_small_grid(); symbol="600519.SH"
    first=prepare_iteration(symbol,"2026-08-01",base,tmp_path)
    result=walk_forward_backtest(_backtest_frame(),base,train_size=180,test_size=60,embargo=5)
    evidence=finalize_iteration(symbol,first,result["folds"])
    assert evidence["generation"]==1 and evidence["advanced"]
    replay=prepare_iteration(symbol,"2026-08-01",base,tmp_path)
    assert replay["status"]=="REPLAY_NO_NEW_DATA" and not replay["advance"]
    assert finalize_iteration(symbol,replay,result["folds"])["generation"]==1
    advanced=prepare_iteration(symbol,"2026-08-02",base,tmp_path)
    assert advanced["status"]=="NEW_DATA_ITERATION" and advanced["generation"]==2
    assert all(len(values)<=5 for values in advanced["grid"].values())

def test_adaptive_parameter_state_rejects_tampering(tmp_path):
    from Main.adaptive_parameter_state import prepare_iteration, finalize_iteration
    base=_small_grid(); symbol="600519.SH"
    prepared=prepare_iteration(symbol,"2026-08-01",base,tmp_path)
    result=walk_forward_backtest(_backtest_frame(),base,train_size=180,test_size=60,embargo=5)
    finalize_iteration(symbol,prepared,result["folds"])
    path=tmp_path/"600519_SH.json"
    state=json.loads(path.read_text(encoding="utf-8")); state["generation"]=99
    path.write_text(json.dumps(state),encoding="utf-8")
    try:
        prepare_iteration(symbol,"2026-08-02",base,tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered adaptive state must be rejected")

def test_future_mutation_cannot_change_first_fold():
    original=_backtest_frame()
    first=walk_forward_backtest(original,_small_grid(),train_size=180,test_size=60,embargo=5)
    mutated=original.copy()
    mutated.loc[mutated.index>=300,["open","close"]]*=10
    second=walk_forward_backtest(mutated,_small_grid(),train_size=180,test_size=60,embargo=5)
    keys=["fast_window","slow_window","vol_window","target_vol"]
    assert first["folds"].iloc[0][keys].to_dict()==second["folds"].iloc[0][keys].to_dict()
    pd.testing.assert_series_equal(first["returns"].query("fold == 0")["strategy_return"],second["returns"].query("fold == 0")["strategy_return"])

def test_eleven_phase_dag_and_fingerprint_are_deterministic():
    audit=validate_orchestration(PHASE_MODULES,PHASE_DEPENDENCIES,PHASE_INPUT_SCHEMA,PHASE_OUTPUT_SCHEMA)
    assert audit["phase_count"]==11 and audit["last_phase"].startswith("Phase_11")
    first,_=build_run_fingerprint("abc",{"x":1},PHASE_MODULES)
    second,_=build_run_fingerprint("abc",{"x":1},PHASE_MODULES)
    changed,_=build_run_fingerprint("abc",{"x":2},PHASE_MODULES)
    assert first==second and first!=changed

def test_cli_phase_aliases_resolve_and_unknown_values_fail():
    assert resolve_phase_name("1") == PHASE_MODULES[0]
    assert resolve_phase_name("Phase_11") == PHASE_MODULES[-1]
    assert resolve_phase_name(PHASE_MODULES[4]) == PHASE_MODULES[4]
    try:
        resolve_phase_name("99")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown phase must not be accepted")

def test_offline_data_manager_does_not_register_network_sources(tmp_path):
    from Main.datasource_manager import FreeDataSourceManager
    manager = FreeDataSourceManager(cache_dir=tmp_path, offline_debug=True)
    assert manager._sources == []
    assert manager.fetch_stock_list()

def test_phase1_contract_rejects_empty_market_data():
    empty = {"assets": [], "adv_data": pd.DataFrame(), "theoretical_aum_limit": 0.0}
    assert not validate_phase_contract(PHASE_MODULES[0], empty, "output")

def test_load_asset_history_fetches_on_cache_miss(tmp_path):
    from Main.data_bus import PITDataBus
    frame = _market_frame()
    class Manager:
        cache_dir = tmp_path
        offline_debug = False
        def fetch_historical(self, symbol, start, end):
            return frame.copy()
    loaded = PITDataBus(Manager()).load_asset_history("600519.SH", "2024-01-01", "2024-12-31")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert len(loaded) == len(frame) and "amount" not in loaded.columns

def test_delisted_stock_interfaces_are_combined():
    from Phase_1.step1_2_returns import _get_delisted_a_stocks
    class Ak:
        def stock_info_sh_delist(self):
            return pd.DataFrame({"公司代码": ["600001"]})
        def stock_info_sz_delist(self):
            return pd.DataFrame({"证券代码": ["000003"]})
    manager = type("Manager", (), {"_ak": Ak()})()
    assert set(_get_delisted_a_stocks(manager)) == {"600001.SH", "000003.SZ"}

def test_point_in_time_atoms_respect_announcement_boundary(tmp_path):
    from Main.data_bus import PITDataBus
    class Manager:
        cache_dir = tmp_path
        offline_debug = True
        def fetch_historical(self, *args):
            return pd.DataFrame()
    manager = Manager()
    bus = PITDataBus(manager)
    bus.append_atom("600519.SH", "2024-01-03", 42.0, "factor", "2024-01-02")
    assert bus.query_by_pit("600519.SH", "2024-01-01", "factor") is None
    assert bus.query_by_pit("600519.SH", "2024-01-02", "factor") == 42.0

def test_cached_history_restores_derived_columns(tmp_path):
    from Main.data_bus import PITDataBus
    frame = _market_frame()
    frame.to_parquet(tmp_path / "600519.SH_history.parquet", index=False)
    manager = type("Manager", (), {"cache_dir": tmp_path, "offline_debug": True})()
    loaded = PITDataBus(manager).load_asset_history("600519.SH", "2024-01-01", "2024-12-31")
    assert {"log_return", "actual_log_return", "amount"}.issubset(loaded.columns)

def test_cache_rejects_fingerprint_mismatch(tmp_path,monkeypatch):
    import Main.context_io as cio
    monkeypatch.setattr(cio,"CACHE_ROOT",tmp_path)
    phase=PHASE_MODULES[0]
    cio.save_phase_result(phase,{"value":42},PHASE_MODULES,run_fingerprint="fingerprint-a")
    assert cio.load_phase_result(phase,PHASE_MODULES,"fingerprint-a")["value"]==42
    assert cio.load_phase_result(phase,PHASE_MODULES,"fingerprint-b") is None

def test_cache_rejects_artifact_tampering(tmp_path,monkeypatch):
    import Main.context_io as cio
    monkeypatch.setattr(cio,"CACHE_ROOT",tmp_path)
    phase=PHASE_MODULES[0]
    cio.save_phase_result(phase,{"value":42},PHASE_MODULES,run_fingerprint="fingerprint-a")
    artifact=tmp_path/"parquet"/"Phase_1"/"value.json"
    artifact.write_text('{"value": 999}',encoding="utf-8")
    assert cio.load_phase_result(phase,PHASE_MODULES,"fingerprint-a") is None

def test_parameter_proposals_are_allowlisted_bounded_and_never_applied():
    current={"max_single_stock_weight":.05,"sector_limit":.30}
    good=validate_parameter_proposal({"max_single_stock_weight":.055},current)
    assert good["valid"] and good["requires_human_approval"] and not good["applied"]
    bad=validate_parameter_proposal({"max_single_stock_weight":.10,"unknown":1},current)
    assert not bad["valid"] and bad["accepted"]=={}

def test_phase10_missing_evidence_holds_and_rejects_unsafe_proposal(tmp_path,monkeypatch):
    import Phase_10.step10_cio_reporting as phase10
    context={"run_metadata":{"timestamp":"test"},"config":{"max_single_stock_weight":.05},"parameter_proposal":{"max_single_stock_weight":.10},"_completed_phases":set()}
    result=phase10.execute(context)
    assert result["cio_decision"]=="HOLD_FOR_REVIEW"
    assert result["parameter_proposal_status"]=="REJECTED"

def test_phase10_reconciliation_failure_cannot_be_eligible():
    import Phase_10.step10_cio_reporting as phase10
    context={"run_metadata":{"timestamp":"test"},"audit_summary":{},"audit_passed":True,"final_nav":100.0,"reconciliation_mae":0.1,"recon_passed":False,"psi_consecutive_breaches":0,"_completed_phases":set()}
    assert phase10.execute(context)["cio_decision"]=="HOLD_FOR_REVIEW"

def test_phase11_downgrades_to_observation_when_cio_holds(tmp_path,monkeypatch):
    import Phase_11.step11_interactive_advisor as phase11
    monkeypatch.setattr(phase11,"build_pipeline_recommendations",lambda context: pd.DataFrame([{"symbol":"510300.SH"}]))
    monkeypatch.setattr(phase11,"write_candidate_report",lambda frame,path:(tmp_path/"x.md",tmp_path/"x.csv"))
    result=phase11.execute({"phase10_ready":True,"cio_decision":"HOLD_FOR_REVIEW","run_metadata":{"timestamp":"test"},"config":{"phase11_interactive":False}})
    assert result["phase11_observation_only"]
    assert not result["investment_candidates"]["action_allowed"].any()

def test_phase3_loader_marks_optional_pit_data_missing_instead_of_fabricating():
    from Phase_3.data_loader import _load_asset_data
    class Manager:
        def fetch_historical(self, *args): return _market_frame()
    loaded = _load_asset_data(Manager(), "600519.SH", "2024-01-01", "2024-12-31")
    optional = {"Free_Float_Cap", "Northbound_Flow", "Dragon_Tiger_Seats"}
    assert optional == set(loaded.attrs["missing_optional_pit_fields"])
    assert loaded[list(optional)].isna().all().all()

def test_crowding_cap_is_a_hard_optimizer_upper_bound():
    bounded = apply_allocation_cap(np.array([0.30, 0.08, -0.01]), {"enforce_crowded_allocation_cap": 0.10}, {})
    assert np.allclose(bounded, [0.10, 0.08, 0.0])

def test_dsr_fails_closed_without_num_trials_evidence():
    from Phase_8.dsr_audit import run_dsr_audit
    context = {"daily_nav": pd.Series(np.linspace(100, 120, 300)), "config": {"min_samples_for_dsr": 20}}
    run_dsr_audit(context)
    assert not context["dsr_pass"]
    assert context["dsr_evidence_status"] == "MISSING_NUM_TRIALS"

def test_reconciliation_failure_freezes_non_live_orders_without_claiming_liquidation():
    from Phase_9.shadow_reconciliation import enforce_reconciliation_gate
    result=enforce_reconciliation_gate({"recon_passed":False,"is_live":False})
    assert result["trading_halted"]
    assert result["kill_switch_report"]["status"]=="ORDER_GENERATION_FROZEN"
    assert not result["kill_switch_report"]["liquidation_attempted"]

def test_reconciliation_fails_closed_without_target_portfolio():
    from Phase_9.shadow_reconciliation import run_shadow_reconciliation
    result=run_shadow_reconciliation({"config":{}})
    assert not result["recon_passed"]
    assert result["reconciliation_evidence_status"]=="MISSING_TARGET_WEIGHTS"

def test_reconciliation_uses_latest_daily_target_weights():
    from Phase_9.shadow_reconciliation import run_shadow_reconciliation
    weights=pd.DataFrame([{"A":0.1},{"A":0.2}])
    result=run_shadow_reconciliation({"daily_weights":weights,"config":{"reconciliation_mae_ceiling":1.0}})
    assert result["target_weights"]=={"A":0.2}

def test_phase11_defense_in_depth_rejects_false_reconciliation(tmp_path,monkeypatch):
    import Phase_11.step11_interactive_advisor as phase11
    monkeypatch.setattr(phase11,"build_pipeline_recommendations",lambda context: pd.DataFrame([{"symbol":"510300.SH"}]))
    monkeypatch.setattr(phase11,"write_candidate_report",lambda frame,path:(tmp_path/"x.md",tmp_path/"x.csv"))
    result=phase11.execute({"phase10_ready":True,"cio_decision":"ELIGIBLE_FOR_PHASE_11","audit_passed":True,"recon_passed":False,"run_metadata":{"timestamp":"test"},"config":{"phase11_interactive":False}})
    assert result["phase11_observation_only"]
    assert not result["investment_candidates"]["action_allowed"].any()

def test_phase11_eligible_research_still_cannot_authorize_trades(tmp_path,monkeypatch):
    import Phase_11.step11_interactive_advisor as phase11
    monkeypatch.setattr(phase11,"build_pipeline_recommendations",lambda context: pd.DataFrame([{"symbol":"510300.SH"}]))
    monkeypatch.setattr(phase11,"write_candidate_report",lambda frame,path:(tmp_path/"x.md",tmp_path/"x.csv"))
    result=phase11.execute({"phase10_ready":True,"cio_decision":"ELIGIBLE_FOR_PHASE_11","audit_passed":True,"recon_passed":True,"run_metadata":{"timestamp":"test"},"config":{"phase11_interactive":False,"analysis_only":True}})
    assert not result["phase11_observation_only"]
    assert result["phase11_analysis_only"]
    assert result["investment_candidates"]["advisory_mode"].eq("ANALYSIS_ONLY").all()
    assert not result["investment_candidates"]["action_allowed"].any()

def test_production_preflight_reports_boundaries():
    from Main.production_readiness import run_preflight
    result=run_preflight(require_clean_git=False)
    assert result["ready"]
    assert result["checks"]["orchestration"]["evidence"]["phase_count"]==11
    assert any("does not prove" in boundary for boundary in result["boundaries"])

def test_bounded_universe_filters_screening_cache(tmp_path):
    import pytz
    from Phase_1.step1_1_screening import run_screening
    today=pd.Timestamp.now(tz="Asia/Shanghai").normalize()
    pd.DataFrame({"symbol":["600519.SH","000001.SZ"],"adv":[2e7,3e7],"cache_date":[today.date(),today.date()]}).to_parquet(tmp_path/"screening_results.parquet",index=False)
    class Bus:
        _tz=pytz.timezone("Asia/Shanghai")
        def get_universe(self): return ["600519.SH"]
    manager=type("Manager",(),{"cache_dir":tmp_path})()
    context={"trading_days_dt":[today],"config":{"bounded_universe":True}}
    run_screening(context,Bus(),manager)
    assert context["assets"]==["600519.SH"]
    assert set(context["adv_data"])=={"600519.SH"}
