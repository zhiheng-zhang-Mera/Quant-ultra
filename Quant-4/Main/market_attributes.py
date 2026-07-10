import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger("MarketAttributes")

def get_free_float_market_cap(bus_obj, asset: str, date: datetime) -> float:
    if asset.endswith(".US"):
        return 500000.0
    d_str = date.strftime('%Y-%m-%d')
    cache_key = f"{asset}_{d_str}"
    if cache_key in bus_obj._mcap_cache:
        return bus_obj._mcap_cache[cache_key]
    s_key = f"series_{asset}"
    if s_key not in bus_obj._mcap_cache:
        try:
            code = f"sh.{asset.split('.')[0]}" if asset.endswith('.SH') else f"sz.{asset.split('.')[0]}"
            rs = bus_obj.manager._bs.query_history_k_data_plus(code=code, fields="date,free_float,close", start_date="2010-01-01", end_date="2026-12-31", adjustflag="2")
            d = []
            while rs.next():
                d.append(rs.get_row_data())
            if d:
                df = pd.DataFrame(d, columns=["date", "free_float", "close"])
                df["mcap"] = (pd.to_numeric(df["free_float"]) * pd.to_numeric(df["close"])) / 1e4
                bus_obj._mcap_cache[s_key] = dict(zip(df["date"], df["mcap"]))
            else:
                bus_obj._mcap_cache[s_key] = {}
        except:
            bus_obj._mcap_cache[s_key] = {}
    m_dict = bus_obj._mcap_cache[s_key]
    if d_str in m_dict:
        val = m_dict[d_str]
        bus_obj._mcap_cache[cache_key] = val
        logger.debug("[OP] Query Extracted Map Database | [SOURCE] Fast Preloaded Memory Dict Storage | [RESULT] Cap size: %s | [SIGNIFICANCE] Resolves inner optimization bottleneck instantaneously", val)
        logger.debug("[操作] 查询解算映射数据库 | [来源] 高速预载内存字典空间 | [结果] 流通市值: %s | [意义] 瞬间消除内层优化器的横截面检索性能瓶颈")
        return val
    if m_dict:
        sd = sorted(m_dict.keys())
        idx = pd.Index(sd).searchsorted(d_str, side='right') - 1
        if idx >= 0:
            val = m_dict[sd[idx]]
            bus_obj._mcap_cache[cache_key] = val
            return val
    return bus_obj._handle_failure("get_free_float_market_cap", asset, Exception("Mcap missing"), 0.0)

def get_sector(bus_obj, asset: str) -> str:
    if asset in bus_obj._sector_cache:
        return bus_obj._sector_cache[asset]
    if asset.endswith(".US"):
        return "科技与成长"
    try:
        if not hasattr(bus_obj, "_global_sw_df") or bus_obj._global_sw_df is None:
            bus_obj._global_sw_df = bus_obj.manager._ak.stock_industry_sw()
        pure = asset.split('.')[0]
        col = "股票代码" if "股票代码" in bus_obj._global_sw_df.columns else "代码"
        row = bus_obj._global_sw_df[bus_obj._global_sw_df[col] == pure]
        if not row.empty:
            for c in ['申万行业', '行业', '申万一级行业']:
                if c in row.columns:
                    res = row.iloc[0][c]
                    bus_obj._sector_cache[asset] = res
                    return res
        raise Exception("Sector unknown")
    except Exception as e:
        return bus_obj._handle_failure("get_sector", asset, e, "未知")

def is_marginable(bus_obj, asset: str) -> bool:
    if asset.endswith(".US"):
        return True
    if asset in bus_obj._margin_cache:
        return bus_obj._margin_cache[asset]
    try:
        if not hasattr(bus_obj, "_global_margin_set") or bus_obj._global_margin_set is None:
            sse = set(bus_obj.manager._ak.stock_margin_sse(start_date="", end_date="")['证券代码'])
            szse = set(bus_obj.manager._ak.stock_margin_sz(start_date="", end_date="")['证券代码'])
            bus_obj._global_margin_set = sse | szse
        res = asset.split('.')[0] in bus_obj._global_margin_set
        bus_obj._margin_cache[asset] = res
        return res
    except Exception as e:
        return bus_obj._handle_failure("is_marginable", asset, e, False)

def compute_market_risk_aversion(bus_obj, end_date: str, window_years=5) -> float:
    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        df = bus_obj.manager.fetch_historical(bus_obj.get_benchmark_code(), (end - timedelta(days=window_years*365)).strftime("%Y-%m-%d"), end_date)
        df.set_index("date", inplace=True)
        rets = np.log(df["close"] / df["close"].shift(1)).resample('W').last()
        lambda_mkt = (rets.mean() * 52 - 0.025) / ((rets.std() * np.sqrt(52)) ** 2)
        logger.info("[OP] Resample Portfolio Macro Premium | [SOURCE] Benchmark Vector Master Array | [RESULT] Risk aversion metric: %s | [SIGNIFICANCE] Sets scaling parameters for Black-Litterman matrix prior equilibrium", lambda_mkt)
        logger.info("[操作] 重采样组合宏观溢价 | [来源] 基准向量大表序列 | [结果] 风险厌恶指标系数: %s | [意义] 设定Black-Litterman矩阵先验均衡的缩放基准")
        return lambda_mkt
    except Exception as e:
        return bus_obj._handle_failure("compute_market_risk_aversion", "market", e, 0.02)