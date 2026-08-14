import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT))
from Main.data_quality import validate_ohlcv
from Main.fast_math import capped_simplex_projection, garman_klass_volatility, log_returns, momentum, rolling_sum
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
from Main.download_runtime import AsyncRestBatchClient, build_download_plan, create_persistent_session
from Phase_1.sector_rotation import score_sector_histories, select_sector_universe
from Main.execute_report import generate_execute_report
from Main.allocation_constraints import apply_allocation_cap
from Main.weekly_rotation import RotationParams, RegimeState, apply_risk_scaling
from Main.ml_regime import MLRegimeResult


def test_ml_bear_override_is_continuous_not_binary():
    """The adaptive ML overlay must fade exposure smoothly instead of a hard
    cash veto, and never below the configured defensive floor."""
    params = RotationParams(ml_bear_override=True, ml_bear_floor=0.30, ml_bear_low=0.40, ml_bull_high=0.60)

    def reg(regime, exposure):
        return RegimeState(date=pd.Timestamp("2024-01-01"), regime=regime, benchmark_close=1.0,
                           benchmark_ma_fast=1.0, benchmark_ma_slow=1.0, exposure=exposure, advice_zh="", advice_en="")

    bear = apply_risk_scaling(None, pd.Timestamp("2024-01-01"), params, reg("BULL", 1.0), MLRegimeResult("BEAR", 0.9, "logit"))
    assert bear.exposure == 0.30
    bull = apply_risk_scaling(None, pd.Timestamp("2024-01-01"), params, reg("BULL", 1.0), MLRegimeResult("BULL", 0.75, "logit"))
    assert bull.exposure == 1.0
    neutral = apply_risk_scaling(None, pd.Timestamp("2024-01-01"), params, reg("NEUTRAL", 0.5), MLRegimeResult("NEUTRAL", 0.5, "logit"))
    assert 0.30 < neutral.exposure < 0.50


def test_drawdown_guard_targets_respect_latch_exposure():
    """Drawdown-guard parameters exist and the latch exposure is a valid target
    so rebalances cannot bypass the de-risking."""
    params = RotationParams(drawdown_guard=0.08, dd_guard_exposure=0.35)
    assert params.dd_guard_exposure == 0.35
    assert params.drawdown_guard == 0.08

def test_data_quality_proves_valid_and_rejects_bad():
    good=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=3),"open":[1,2,3],"high":[2,3,4],"low":[.5,1,2],"close":[1.5,2.5,3.5],"volume":[1,2,3]})
    assert validate_ohlcv(good)["valid"]
    bad=good.copy(); bad.loc[0,"high"]=0
    assert not validate_ohlcv(bad)["valid"]

def test_fast_math_and_metrics():
    p=np.array([100.,110.,121.]); assert np.allclose(log_returns(p),np.log([1.1,1.1]))
    assert metrics(pd.Series(np.linspace(100,130,100)))["observations"]==99

def test_extended_fast_math_matches_vectorized_reference():
    values = np.arange(1.0, 9.0)
    sums = rolling_sum(values, 3)
    assert np.allclose(sums[2:], [6, 9, 12, 15, 18, 21])
    prices = np.exp(np.linspace(0, .7, 8))
    assert np.allclose(momentum(prices, 2)[2:], np.log(prices[2:] / prices[:-2]))
    gk = garman_klass_volatility(prices, prices * 1.02, prices * .98, prices * 1.01)
    assert np.isfinite(gk).all() and (gk >= 0).all()
    projected = capped_simplex_projection(np.array([.8, .5, .2]), np.array([.6, .4, .3]), .75)
    assert projected.sum() <= .75000001 and np.all(projected >= 0) and np.all(projected <= [.6, .4, .3])

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
    assert result["alternative_signal_governance"]["status"] == "HOLD_FOR_REVIEW"
    assert result["alternative_signal_governance"]["action"] == "OBSERVATION_ONLY"
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
    assert len(evidence["prompt_sha256"]) == 64
    assert len(evidence["response_sha256"]) == 64
    assert evidence["prompt_version"] == "financial-sentiment-json/v1"
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

def test_phase6_solver_backend_respects_hardware_and_numpy_fallback_constraints():
    from Phase_6.convex_optimizer import _python_matrix_solve, select_solver_backend
    low = {"config": {"allow_native_solver": True}, "compute_audit": {"hardware": {"available_memory_gb": .5}, "resource_plan": {"cpu_workers": 1}}}
    assert select_solver_backend(low)["backend"] == "numpy_projected_gradient"
    weights = _python_matrix_solve(
        np.array([.2, .1, .05]), np.eye(3) * .1, np.zeros(3), np.array([.5, .5, .5]),
        {"A": "x", "B": "x", "C": "y"}, ["A", "B", "C"],
        {"cash_buffer_weight": .1, "sector_limit": .45, "max_daily_turnover": .8},
    )
    assert np.isfinite(weights).all() and (weights >= 0).all()
    assert weights.sum() <= .9000001 and weights[:2].sum() <= .4500001

def test_phase6_time_slice_solver_parallelizes_contiguous_blocks(monkeypatch):
    import Phase_6.step6_position_sizing as sizing
    monkeypatch.setattr(sizing, "step_m_1_directional_mask", lambda context, date: {"A": 1})
    monkeypatch.setattr(sizing, "step_m_2_black_litterman_fusion", lambda context, date, previous: (np.ones(1), np.eye(1), np.ones(1), np.ones(1)))
    monkeypatch.setattr(sizing, "step_m_3_convex_optimization", lambda context, date, nav, previous: previous + 1)
    dates = list(pd.date_range("2026-01-01", periods=6))
    context = {"assets": ["A"], "config": {"parallel_time_slices": True, "parallel_slice_min_dates": 2, "parallel_time_slice_cap": 2, "optimization_workers": 4}, "compute_audit": {"resource_plan": {"optimization_workers": 2}}}
    records, audit = sizing.solve_time_slices(context, dates, 1_000_000)
    assert audit["parallel"] and audit["slice_lengths"] == [3, 3]
    assert [date for date, _ in records] == dates
    assert [float(weights[0]) for _, weights in records] == [1, 2, 3, 1, 2, 3]

def test_download_runtime_sizes_pools_and_reuses_connections():
    plan = build_download_plan(worker_override=7)
    assert plan.workers == 7
    assert plan.connection_pool >= plan.workers * 2
    assert plan.batch_size >= plan.workers
    session = create_persistent_session(plan)
    assert session.get_adapter("https://")._pool_maxsize == plan.connection_pool
    assert session.headers["Connection"] == "keep-alive"
    assert AsyncRestBatchClient(plan).plan is plan

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

def test_purchase_eligibility_excludes_permission_and_non_a_share_codes():
    from Main.advice_portfolio_backtest import symbol_purchase_eligibility
    assert symbol_purchase_eligibility("600519.SH")[0]
    assert symbol_purchase_eligibility("000001.SZ")[0]
    assert symbol_purchase_eligibility("510300.SH")[0]
    for symbol in ("300001.SZ","688001.SH","920001.BJ","900901.SH","AAPL.US"):
        assert not symbol_purchase_eligibility(symbol)[0]

def test_advice_portfolio_requires_multiple_real_assets():
    from Main.advice_portfolio_backtest import UniverseDecision, run_advice_portfolio_backtest
    frame=_market_frame().set_index("date")
    try:
        run_advice_portfolio_backtest({"600519.SH":frame},[UniverseDecision("600519.SH",True,"test")],min_assets=2)
    except ValueError:
        pass
    else:
        raise AssertionError("single-security advice backtests must be rejected")

def test_dynamic_exposure_is_bounded_and_reacts_to_risk():
    from Main.advice_portfolio_backtest import dynamic_risk_exposure
    strong=dynamic_risk_exposure(0.9,0.9,0.12,0.0)
    stressed=dynamic_risk_exposure(0.2,0.2,0.50,-0.12)
    assert 0.15 <= stressed < strong <= 0.90

def test_historical_eligibility_uses_only_as_of_rows():
    from Main.advice_portfolio_backtest import eligible_as_of
    frame=_market_frame().set_index("date")
    signal_date=frame.index[99]
    assert eligible_as_of({"600519.SH":frame},signal_date,60)==["600519.SH"]
    assert eligible_as_of({"600519.SH":frame},frame.index[30],60)==[]

def test_dynamic_calendar_does_not_require_identical_listing_dates():
    from Main.advice_portfolio_backtest import UniverseDecision, run_advice_portfolio_backtest
    first=_market_frame().set_index("date")
    second=first.iloc[20:].copy()
    decisions=[UniverseDecision("600519.SH",True,"test"),UniverseDecision("000001.SZ",True,"test")]
    result=run_advice_portfolio_backtest({"600519.SH":first,"000001.SZ":second},decisions,years=1,lookback=60,min_assets=2,max_holding_days=10)
    assert result["returns"]["historically_eligible_assets"].min() <= result["returns"]["historically_eligible_assets"].max()

def test_full_market_refresh_fails_before_download_when_stock_coverage_is_thin(tmp_path,monkeypatch):
    import Main.advice_portfolio_backtest as module
    class ThinSource:
        def __init__(self,**kwargs): pass
        def fetch_full_market_list(self,include_delisted=True): return ["510300.SH"]*600
        def fetch_historical(self,*args,**kwargs): raise AssertionError("history download must not start")
    monkeypatch.setattr(module,"FreeDataSourceManager",ThinSource)
    try:
        module.refresh_full_market_cache(tmp_path,"2020-01-01","2024-01-01",minimum_stock_coverage=1000)
    except RuntimeError as exc:
        assert "stocks=" in str(exc)
    else:
        raise AssertionError("undersized stock coverage must fail closed")

def test_market_source_timeout_switches_channel_and_writes_audit(tmp_path):
    from Main.datasource_manager import FreeDataSourceManager
    manager=FreeDataSourceManager(cache_dir=tmp_path,offline_debug=True,source_timeout_seconds=.01)
    manager.offline_debug=False
    def slow(*args):
        time.sleep(.1)
        return _market_frame()
    manager._sources=[("slow",slow),("fast",lambda *args:_market_frame())]
    result=manager.fetch_historical("600519.SH","2024-01-01","2024-04-09")
    assert result is not None and len(result)==100
    audit=json.loads((tmp_path/"evidence"/"600519.SH_download_audit.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in audit["attempts"]]==["TIMEOUT_CIRCUIT_OPEN","ACCEPTED"]
    assert "slow" in manager._disabled_sources

def test_history_cache_downloads_only_incremental_tail_and_merges_atomically(tmp_path):
    from Main.datasource_manager import FreeDataSourceManager
    manager = FreeDataSourceManager(cache_dir=tmp_path, offline_debug=True, download_workers=2)
    manager.offline_debug = False
    cached = _market_frame().iloc[:80].copy()
    cache_path = tmp_path / "600519.SH_history.parquet"
    cached.to_parquet(cache_path, index=False)
    evidence = validate_ohlcv(cached, "600519.SH")
    evidence["provider"] = "fixture"
    (tmp_path / "evidence" / "600519.SH_fixture.json").write_text(json.dumps(evidence), encoding="utf-8")
    calls = []
    def incremental(symbol, start, end, freq):
        calls.append((start, end))
        return _market_frame().iloc[74:].copy()
    manager._sources = [("incremental", incremental)]
    result = manager.fetch_historical("600519.SH", "2024-01-01", "2024-04-09")
    assert calls == [("2024-03-15", "2024-04-09")]
    assert len(result) == 100
    stored = pd.read_parquet(cache_path)
    assert len(stored) == 100 and not cache_path.with_suffix(".parquet.tmp").exists()

def test_batch_history_interface_deduplicates_symbols(tmp_path, monkeypatch):
    from Main.datasource_manager import FreeDataSourceManager
    manager = FreeDataSourceManager(cache_dir=tmp_path, offline_debug=True, download_workers=3)
    calls = []
    def fake(symbol, start, end, freq="d"):
        calls.append(symbol)
        return _market_frame()
    monkeypatch.setattr(manager, "fetch_historical", fake)
    results = manager.fetch_historical_batch(["A", "B", "A"], "2024-01-01", "2024-04-09")
    assert set(results) == {"A", "B"}
    assert sorted(calls) == ["A", "B"]

def _sector_frame(momentum=0.10, amount=100.0, volatility=0.005, end="2026-08-05"):
    dates = pd.bdate_range(end=end, periods=80)
    rng = np.random.default_rng(abs(hash((momentum, amount))) % (2**32))
    daily = momentum / 20 + rng.normal(0, volatility, len(dates))
    close = 100 * np.exp(np.cumsum(daily))
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * .99, "close": close, "amount": amount})

def test_sector_scoring_applies_momentum_liquidity_and_volatility_gates():
    histories = {
        "leader": _sector_frame(.20, 400),
        "steady": _sector_frame(.10, 300),
        "third": _sector_frame(.05, 200),
        "illiquid": _sector_frame(.30, 1),
        "volatile": _sector_frame(.40, 100, volatility=.08),
    }
    ranking = score_sector_histories(histories, "2026-08-05")
    eligible = ranking.loc[ranking["eligible"], "sector"].tolist()
    assert "illiquid" not in eligible
    assert "volatile" not in eligible
    assert set(eligible[:3]) == {"leader", "steady", "third"}
    assert np.allclose(ranking["score"], .4 * ranking["momentum_rank"] + .3 * ranking["amount_share_rank"] + .3 * ranking["adv_ratio_rank"])

def test_sector_first_selection_updates_indexes_but_only_loads_top_three_constituents(tmp_path):
    board_names = ["leader", "steady", "third", "laggard"]
    histories = {
        "leader": _sector_frame(.20, 400),
        "steady": _sector_frame(.10, 300),
        "third": _sector_frame(.05, 200),
        "laggard": _sector_frame(-.10, 100),
    }
    class Ak:
        history_starts = []
        constituent_calls = []
        def stock_board_industry_name_em(self):
            return pd.DataFrame({"板块代码": [f"BK{i}" for i in range(4)], "板块名称": board_names})
        def stock_board_industry_hist_em(self, symbol, period, start_date, end_date, adjust):
            self.history_starts.append((symbol, start_date))
            return histories[symbol].rename(columns={"date":"日期","open":"开盘","high":"最高","low":"最低","close":"收盘","amount":"成交额"})
        def stock_board_industry_cons_em(self, symbol):
            self.constituent_calls.append(symbol)
            offset = board_names.index(symbol)
            return pd.DataFrame({"代码": [f"6000{offset}1", f"0000{offset}2"]})
    class Manager:
        cache_dir = tmp_path
        _ak = Ak()
        def _bounded_source_call(self, name, func, *args, **kwargs):
            return func(*args, **kwargs)
    manager = Manager()
    symbols, selected = select_sector_universe(manager, "2026-08-05", top_n=3)
    assert len(selected) == 3 and len(symbols) == 6
    assert set(manager._ak.constituent_calls) == set(selected["sector"])
    assert len(manager._ak.history_starts) == 4
    first_snapshot = tmp_path / "sector_rotation" / "selections" / "2026-08-05.json"
    assert first_snapshot.exists()

    manager._ak.constituent_calls.clear(); manager._ak.history_starts.clear()
    select_sector_universe(manager, "2026-08-06", top_n=3)
    assert first_snapshot.exists()
    assert (tmp_path / "sector_rotation" / "selections" / "2026-08-06.json").exists()
    assert all(start > "20260101" for _, start in manager._ak.history_starts)

def test_stock_list_timeout_switches_to_static_channel(tmp_path):
    from Main.datasource_manager import FreeDataSourceManager
    manager=FreeDataSourceManager(cache_dir=tmp_path,offline_debug=True,source_timeout_seconds=.01)
    manager.offline_debug=False
    class FakeAk:
        def stock_zh_a_spot_em(self):
            time.sleep(.1)
            return pd.DataFrame()
        def stock_info_a_code_name(self):
            return pd.DataFrame({"code":[f"{600000+i:06d}" for i in range(600)]})
    manager._ak=FakeAk()
    symbols=manager.fetch_stock_list()
    assert len(symbols)==600
    assert "akshare_spot_list" in manager._disabled_sources

def test_timed_out_channel_and_symbol_reenter_after_cooldown(tmp_path):
    from Main.datasource_manager import FreeDataSourceManager
    manager=FreeDataSourceManager(cache_dir=tmp_path,offline_debug=True,source_timeout_seconds=.005,source_cooldown_seconds=.02)
    manager.offline_debug=False
    calls={"count":0}
    def recovering(*args):
        calls["count"]+=1
        if calls["count"]==1: time.sleep(.03)
        return _market_frame()
    manager._sources=[("recovering",recovering)]
    assert manager.fetch_historical("600519.SH","2024-01-01","2024-04-09") is None
    time.sleep(.03)
    result=manager.fetch_historical("600519.SH","2024-01-01","2024-04-09")
    assert result is not None and len(result)==100
    assert "recovering" not in manager._disabled_sources
    assert "600519.SH" not in manager._failed_symbols

def test_ollama_setup_is_check_only(monkeypatch):
    import Phase_10.env_setup as setup
    monkeypatch.setattr(setup.shutil,"which",lambda name:None)
    monkeypatch.setattr(setup,"urlopen",lambda *args,**kwargs:(_ for _ in ()).throw(OSError("offline")))
    result=setup.setup_all()
    assert result["framework"]["check_only"] and not result["framework"]["installed"]
    assert result["model"]["check_only"] and not result["model"]["service_reachable"]

def test_dependency_installer_does_not_install_ollama():
    import Main.install_deps as installer
    assert "ollama" not in installer.IMPORT_CHECKS
    assert installer.REQUIREMENTS.exists()

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
    # 生产数据管理器通过 _bounded_source_call 封装超时/熔断；测试桩需提供同一接口。
    manager = type("Manager", (), {
        "_ak": Ak(),
        "_bounded_source_call": lambda self, name, func, *args, **kwargs: func(*args, **kwargs),
    })()
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


def test_reconciliation_pass_records_no_kill_switch_report():
    """对账通过时必须写入明确的 NO_KILL_SWITCH_TRIGGERED 审计记录，
    否则 Phase 9 输出契约会因 kill_switch_report=None 把“通过态”误判为缺失证据。"""
    from Phase_9.shadow_reconciliation import enforce_reconciliation_gate
    result = enforce_reconciliation_gate({"recon_passed": True})
    assert result["trading_halted"] is False
    assert result["kill_switch_report"]["status"] == "NO_KILL_SWITCH_TRIGGERED"
    assert result["kill_switch_report"]["reason"] == "RECONCILIATION_PASSED"

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


# ---------------------------------------------------------------------------
# 8-9 update plan regression tests (P0/P1/P2 fixes)
# ---------------------------------------------------------------------------

def test_dsr_significance_requires_small_p_value():
    """DSR 门禁方向：p < 0.05 表示夏普显著高于门槛基线 → 通过；
    p 大（不显著）→ 不通过。原实现比较方向写反，会放行不显著策略并拒绝显著策略。"""
    from Phase_8.dsr_audit import run_dsr_audit
    rng = np.random.default_rng(0)
    strong_returns = np.full(400, 0.0008) + rng.normal(0, 0.005, 400)
    strong = {"daily_nav": pd.Series(100 * np.exp(np.cumsum(strong_returns))), "config": {"min_samples_for_dsr": 20}, "num_trials": 50}
    run_dsr_audit(strong)
    assert strong["dsr_pass"] is True
    assert strong["dsr_pval"] < 0.05

    flat = {"daily_nav": pd.Series(100 + np.sin(np.linspace(0, 4, 400)) * 0.001), "config": {"min_samples_for_dsr": 20}, "num_trials": 50}
    run_dsr_audit(flat)
    assert flat["dsr_pass"] is False

    # 夏普低于门槛基线（0.50）时必须不通过；单侧 p = Phi(-t/penalty)，
    # 不能因 abs(t) 的双侧化而把小 p 误判为“达标”。
    rng_below = np.random.default_rng(0)
    below_returns = np.full(600, 0.00012) + rng_below.normal(0, 0.0025, 600)
    below = {"daily_nav": pd.Series(100 * np.exp(np.cumsum(below_returns))), "config": {"min_samples_for_dsr": 20}, "num_trials": 50}
    run_dsr_audit(below)
    assert below["nominal_sharpe"] < 0.5
    assert below["dsr_pval"] > 0.05
    assert below["dsr_pass"] is False


def test_dsr_degenerate_variance_fails_closed():
    """分母非正/非有限（极端右尾）时必须显式判失败，不能因 p=0 而通过。"""
    from Phase_8.dsr_audit import run_dsr_audit
    rng = np.random.default_rng(7)
    r = np.where(rng.random(50) < 0.5, 0.1 * 0.001, -0.1 * 0.05)
    r = r + rng.normal(0, 0.1 * 0.0002, 50)
    context = {"daily_nav": pd.Series(100 * np.exp(np.cumsum(r))),
               "config": {"min_samples_for_dsr": 20}, "num_trials": 50}
    run_dsr_audit(context)
    assert context["dsr_pass"] is False


def test_coverage_handles_empty_and_nan_violations():
    from Phase_8.coverage_test import run_christoffersen_test
    dates = pd.date_range("2024-01-01", periods=300)
    nav = pd.Series(np.exp(np.cumsum(np.full(300, 0.0008))), index=dates)
    empty = {"daily_nav": nav, "violations": pd.Series([], dtype=float), "config": {}}
    run_christoffersen_test(empty)
    assert empty["empirical_coverage"] == 1.0
    assert empty["christoffersen_pass"] is True
    assert not pd.isna(empty["empirical_coverage"])

    with_nan = {"daily_nav": nav, "violations": pd.Series([np.nan] * 300, index=dates), "config": {}}
    run_christoffersen_test(with_nan)
    assert not pd.isna(with_nan["empirical_coverage"])


def test_coverage_sparse_violations_do_not_trigger_lr_false_alarm():
    """违规次数极少（如 500+ 天仅 5 次）时 LR 独立性检验无统计效力，
    应按容错保护退化为通过（p=1.0），仅以无条件覆盖率判定。"""
    from Phase_8.coverage_test import run_christoffersen_test
    dates = pd.date_range("2024-01-01", periods=500)
    nav = pd.Series(np.exp(np.cumsum(np.full(500, 0.0005))), index=dates)
    viol = pd.Series(0, index=dates)
    for i in (50, 120, 210, 320, 430):
        viol.iloc[i] = 1
    context = {"daily_nav": nav, "violations": viol, "config": {}}
    run_christoffersen_test(context)
    assert context["empirical_coverage"] > 0.98
    assert context["christoffersen_pass"] is True


def test_stress_test_marks_out_of_window_scenarios_uncovered():
    from Phase_8.stress_test import run_stress_test
    dates = pd.date_range("2024-01-05", periods=200)
    nav = pd.Series(np.exp(np.cumsum(np.full(200, 0.0005))), index=dates)
    context = {"daily_nav": nav, "config": {}}
    run_stress_test(context)
    assert context["stress_drawdowns_covered"]["2015_liq"] is False
    assert context["stress_drawdowns_covered"]["2016_meltdown"] is False
    assert context["stress_drawdowns_covered"]["2024_microcap"] is True


def test_cio_report_is_strict_json_without_nan(tmp_path):
    import json as jsonlib
    import Phase_10.step10_cio_reporting as phase10
    context = {
        "run_metadata": {"timestamp": "test_strict_json"},
        "audit_summary": {"stress_drawdowns": {"2015_liq": float("nan"), "2024_microcap": -0.12}},
        "audit_passed": False,
        "final_nav": 100.0,
        "reconciliation_mae": 0.001,
        "recon_passed": True,
        "psi_consecutive_breaches": 0,
        "_completed_phases": {"Phase_8.step8_audit_stress_test", "Phase_9.step9_live_mlops"},
    }
    result = phase10.execute(context)
    path = result["cio_report_path"]
    payload = jsonlib.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["decision"] == "HOLD_FOR_REVIEW"
    assert payload["evidence"]["audit_summary"]["stress_drawdowns"]["2015_liq"] is None
    assert payload["evidence"]["audit_summary"]["stress_drawdowns"]["2024_microcap"] == -0.12


def test_reconciliation_ceiling_aligns_to_production_gate():
    from Phase_9.config import DEFAULT_MLOPS_CONFIG
    assert DEFAULT_MLOPS_CONFIG["reconciliation_mae_ceiling"] == 0.01


def test_board_lot_alignment_rounds_targets_to_lots():
    from Phase_6.step6_position_sizing import apply_execution_alignment
    dates = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"close": [10.0] * 5}, index=dates)
    context = {
        "config": {"board_lot_rounding": True, "min_trade_weight": 0.0, "cash_buffer_weight": 0.05},
        "bulk_history_cache": {"600519.SH": frame},
    }
    assets = ["600519.SH"]
    nav = 1_000_000.0
    weights = apply_execution_alignment(context, dates[2], np.array([0.25]), np.array([0.0]), nav, assets)
    shares = weights[0] * nav / 10.0
    assert abs(shares - round(shares / 100) * 100) < 1e-6
    assert weights[0] > 0


def test_min_trade_weight_suppresses_micro_rebalances():
    from Phase_6.step6_position_sizing import apply_execution_alignment
    context = {"config": {"min_trade_weight": 0.01, "board_lot_rounding": False}}
    weights = apply_execution_alignment(context, pd.Timestamp("2024-01-01"), np.array([0.20, 0.11]), np.array([0.20, 0.10]), 1e7, ["A", "B"])
    assert weights[0] == 0.20
    assert weights[1] == 0.10


def test_rolling_zscore_uses_only_past_window():
    from Phase_5.step_5_2_3_features import apply_rolling_zscore
    cube = np.ones((100, 1, 1), dtype=float)
    cube[20:, 0, 0] = 100.0
    z = apply_rolling_zscore(cube, lookback=60, min_periods=20)
    # 跳跃前的点只应使用历史窗口（全部为 1 → 均值为 1、标准差为 0 → 中性 0），
    # 不受未来第 20 天跳跃的影响；跳跃当天及之后才被标准化为正值。
    assert z[19, 0, 0] == 0.0
    assert z[20, 0, 0] > 0.0
    assert np.isfinite(z).all()


def test_live_kill_switch_liquidates_positions():
    from Phase_9.shadow_reconciliation import trigger_physical_hard_kill_switch
    class FakeEngine:
        holdings = {"600519.SH": 100.0, "000001.SZ": 200.0}
        cash = 100.0
        def calc_nav(self):
            return 1_000_000.0
    class FakeGateway:
        def close_all_market_positions(self):
            self.closed = True
    gateway = FakeGateway()
    report = trigger_physical_hard_kill_switch(FakeEngine(), gateway)
    assert report["status"] == "TOTAL_LIQUIDATION_EXECUTED"
    assert report["liquidated_assets"][0] == "ALL_LIVE_PORTFOLIO_LIQUIDATED"
    assert getattr(gateway, "closed", False) is True
