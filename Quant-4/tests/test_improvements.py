import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT))
from Main.data_quality import validate_ohlcv
from Main.fast_math import log_returns
from Main.portfolio_analytics import metrics, nearest_psd, risk_parity_weights
from analyze_cn_asset import normalize

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
