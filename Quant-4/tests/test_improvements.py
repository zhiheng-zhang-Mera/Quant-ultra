import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT))
from Main.data_quality import validate_ohlcv
from Main.fast_math import log_returns
from Main.portfolio_analytics import holding_advice, metrics, nearest_psd, recommendation, risk_parity_weights
from analyze_cn_asset import normalize
from Main.investment_advisor import write_candidate_report
from Main.walk_forward_backtest import walk_forward_backtest
from Main.orchestration_guard import build_run_fingerprint, validate_orchestration
from Main.parameter_governance import validate_parameter_proposal
from Main.schema_contracts import PHASE_DEPENDENCIES, PHASE_INPUT_SCHEMA, PHASE_MODULES, PHASE_OUTPUT_SCHEMA

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
    assert 0 <= rec["suggested_weight"] <= .10
    assert .08 <= rec["take_profit_pct"] <= .30

def test_holding_advice_reconciles_cash_position_and_action():
    result=holding_advice(_market_frame(),100000,1000,10,model_weight=.08)
    assert np.isclose(result["market_value"],1000*result["last_price"])
    assert result["target_quantity"] % 100 == 0
    assert result["action"] in {"分批止盈","减仓","按批次买入","减仓至目标","持有观察"}
    assert result["action_reason"]

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

def test_cache_rejects_fingerprint_mismatch(tmp_path,monkeypatch):
    import Main.context_io as cio
    monkeypatch.setattr(cio,"CACHE_ROOT",tmp_path)
    phase=PHASE_MODULES[0]
    cio.save_phase_result(phase,{"value":42},PHASE_MODULES,run_fingerprint="fingerprint-a")
    assert cio.load_phase_result(phase,PHASE_MODULES,"fingerprint-a")["value"]==42
    assert cio.load_phase_result(phase,PHASE_MODULES,"fingerprint-b") is None

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

def test_phase11_downgrades_to_observation_when_cio_holds(tmp_path,monkeypatch):
    import Phase_11.step11_interactive_advisor as phase11
    monkeypatch.setattr(phase11,"build_pipeline_recommendations",lambda context: pd.DataFrame([{"symbol":"510300.SH"}]))
    monkeypatch.setattr(phase11,"write_candidate_report",lambda frame,path:(tmp_path/"x.md",tmp_path/"x.csv"))
    result=phase11.execute({"phase10_ready":True,"cio_decision":"HOLD_FOR_REVIEW","run_metadata":{"timestamp":"test"},"config":{"phase11_interactive":False}})
    assert result["phase11_observation_only"]
    assert not result["investment_candidates"]["action_allowed"].any()
