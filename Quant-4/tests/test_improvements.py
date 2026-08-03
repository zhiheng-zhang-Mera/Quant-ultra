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
