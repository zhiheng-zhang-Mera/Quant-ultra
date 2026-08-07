import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

logger = logging.getLogger("MarketAttributes")

# [调整3] 实现数据与计算分离：强制将外部数据读取本地化，绝不在逻辑流中发起 HTTP
LOCAL_CACHE_DIR = Path(__file__).parent.parent / "Data_Cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
US_STATIC_CACHE_FILE = LOCAL_CACHE_DIR / "us_static_meta.json"
ASHARE_SW_CACHE_FILE = LOCAL_CACHE_DIR / "ashare_sw_meta.parquet"
ASHARE_MARGIN_CACHE_FILE = LOCAL_CACHE_DIR / "ashare_margin_meta.parquet"

def _load_us_static_cache():
    if US_STATIC_CACHE_FILE.exists():
        with open(US_STATIC_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_us_static_cache(data):
    with open(US_STATIC_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_free_float_market_cap(bus_obj, asset: str, date: datetime) -> float:
    d_str = date.strftime('%Y-%m-%d')
    cache_key = f"{asset}_{d_str}"
    if cache_key in bus_obj._mcap_cache:
        return bus_obj._mcap_cache[cache_key]
        
    s_key = f"series_{asset}"
    
    # [调整2] 清除美股Mock：转为计算真实市值
    if asset.endswith(".US"):
        if s_key not in bus_obj._mcap_cache:
            try:
                # 尝试从 DataBus 缓存池或预拉取的 parquet 读取
                c_path = LOCAL_CACHE_DIR / f"us_{asset}_history.parquet"
                if c_path.exists():
                    df = pd.read_parquet(c_path)
                    # 粗略市值预估 = volume * close 或取自独立的 Fundamental 库
                    # 生产环境应当有专门的财务数据接口推送 real free_float
                    df["mcap"] = df["close"] * df["volume"] / 1e4 
                    bus_obj._mcap_cache[s_key] = dict(zip(df["date"].dt.strftime('%Y-%m-%d'), df["mcap"]))
                else:
                    bus_obj._mcap_cache[s_key] = {}
            except Exception as e:
                logger.warning(f"Failed to load US Mcap for {asset}: {e}")
                bus_obj._mcap_cache[s_key] = {}
    else:
        # A股原有逻辑
        if s_key not in bus_obj._mcap_cache:
            try:
                code = f"sh.{asset.split('.')[0]}" if asset.endswith('.SH') else f"sz.{asset.split('.')[0]}"
                cache_path = LOCAL_CACHE_DIR / f"mcap_{asset}.parquet"
                mcap_series = {}
                if cache_path.exists():
                    try:
                        df_disk = pd.read_parquet(cache_path)
                        mcap_series = dict(zip(df_disk["date"].astype(str), df_disk["mcap"]))
                    except Exception:
                        mcap_series = {}
                if not mcap_series:
                    def _query_mcap():
                        with bus_obj.manager._bs_lock:
                            if not bus_obj.manager._bs_logged:
                                bus_obj.manager._bs.login()
                                bus_obj.manager._bs_logged = True
                            rs = bus_obj.manager._bs.query_history_k_data_plus(code=code, fields="date,free_float,close", start_date="2010-01-01", end_date="2030-12-31", adjustflag="2")
                            out = []
                            while rs.next():
                                out.append(rs.get_row_data())
                            return out
                    raw = bus_obj.manager._bounded_source_call(f"baostock_mcap:{asset}", _query_mcap)
                    if raw:
                        frame = pd.DataFrame(raw, columns=["date", "free_float", "close"])
                        frame["mcap"] = (pd.to_numeric(frame["free_float"]) * pd.to_numeric(frame["close"])) / 1e4
                        frame[["date", "mcap"]].to_parquet(cache_path, index=False)
                        mcap_series = dict(zip(frame["date"].astype(str), frame["mcap"]))
                bus_obj._mcap_cache[s_key] = mcap_series
            except:
                bus_obj._mcap_cache[s_key] = {}

    m_dict = bus_obj._mcap_cache[s_key]
    if d_str in m_dict:
        val = m_dict[d_str]
        bus_obj._mcap_cache[cache_key] = val
        return val
    if m_dict:
        sd = sorted(m_dict.keys())
        idx = pd.Index(sd).searchsorted(d_str, side='right') - 1
        if idx >= 0:
            val = m_dict[sd[idx]]
            bus_obj._mcap_cache[cache_key] = val
            return val
            
    # [调整2] 移除写死的 500000.0, 记录真实缺失
    return bus_obj._handle_failure("get_free_float_market_cap", asset, Exception("Mcap missing in TS"), 0.0)

def get_sector(bus_obj, asset: str) -> str:
    if asset in bus_obj._sector_cache:
        return bus_obj._sector_cache[asset]
        
    # [调整2 & 3] 移除美股一律 "科技与成长" 的 Mock，使用本地 JSON 缓存
    if asset.endswith(".US"):
        us_meta = _load_us_static_cache()
        if asset in us_meta and "sector" in us_meta[asset]:
            bus_obj._sector_cache[asset] = us_meta[asset]["sector"]
            return us_meta[asset]["sector"]
        else:
            # Fallback 写入未知，避免破坏矩阵计算
            logger.warning(f"Missing sector metadata for US asset {asset} in local cache. Required ETL update.")
            bus_obj._sector_cache[asset] = "US_Unknown"
            return "US_Unknown"

    # [调整3] 杜绝内层循环动态爬网，改从本地构建好的缓存 Parquet 加载 A股行业
    try:
        if not hasattr(bus_obj, "_global_sw_df") or bus_obj._global_sw_df is None:
            if ASHARE_SW_CACHE_FILE.exists():
                bus_obj._global_sw_df = pd.read_parquet(ASHARE_SW_CACHE_FILE)
            else:
                logger.warning("ETL Sw-Sector Cache missing, triggering ONE-TIME fallback fetch. (NOT recommended in live mode)")
                bus_obj._global_sw_df = bus_obj.manager._bounded_source_call("akshare_sw_sector", bus_obj.manager._ak.stock_industry_sw)
                bus_obj._global_sw_df.to_parquet(ASHARE_SW_CACHE_FILE)

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
    if asset in bus_obj._margin_cache:
        return bus_obj._margin_cache[asset]
        
    if asset.endswith(".US"):
        # 绝大多数成熟美股标的都支持 Margin，但仍应从基础静态库读取
        us_meta = _load_us_static_cache()
        res = us_meta.get(asset, {}).get("marginable", True)
        bus_obj._margin_cache[asset] = res
        return res
        
    # [调整3] 同理，杜绝两融数据的内层爬网
    try:
        if not hasattr(bus_obj, "_global_margin_set") or bus_obj._global_margin_set is None:
            if ASHARE_MARGIN_CACHE_FILE.exists():
                df_margin = pd.read_parquet(ASHARE_MARGIN_CACHE_FILE)
                bus_obj._global_margin_set = set(df_margin['证券代码'])
            else:
                logger.warning("ETL Margin Cache missing, ONE-TIME fallback fetch.")
                sse = set(bus_obj.manager._bounded_source_call("akshare_margin_sse", bus_obj.manager._ak.stock_margin_sse, start_date="", end_date="")['证券代码'])
                szse = set(bus_obj.manager._bounded_source_call("akshare_margin_sz", bus_obj.manager._ak.stock_margin_sz, start_date="", end_date="")['证券代码'])
                bus_obj._global_margin_set = sse | szse
                pd.DataFrame({"证券代码": list(bus_obj._global_margin_set)}).to_parquet(ASHARE_MARGIN_CACHE_FILE)

        res = asset.split('.')[0] in bus_obj._global_margin_set
        bus_obj._margin_cache[asset] = res
        return res
    except Exception as e:
        return bus_obj._handle_failure("is_marginable", asset, e, False)

def compute_market_risk_aversion(bus_obj, end_date: str, window_years=5) -> float:
    # Market risk aversion is a slow-moving, market-wide parameter; compute at
    # most once per calendar month to avoid repeated network work per day.
    cache_key = f"ra_{end_date[:7]}_{window_years}"
    if cache_key in bus_obj._risk_aversion_cache:
        return bus_obj._risk_aversion_cache[cache_key]
    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        df = bus_obj.manager.fetch_historical(bus_obj.get_benchmark_code(), (end - timedelta(days=window_years*365)).strftime("%Y-%m-%d"), end_date)
        df.set_index("date", inplace=True)
        rets = np.log(df["close"] / df["close"].shift(1)).resample('W').last()
        lambda_mkt = (rets.mean() * 52 - 0.025) / ((rets.std() * np.sqrt(52)) ** 2)
        if not np.isfinite(lambda_mkt):
            raise ValueError("non-finite market risk aversion")
        bus_obj._risk_aversion_cache[cache_key] = float(lambda_mkt)
        return float(lambda_mkt)
    except Exception as e:
        value = bus_obj._handle_failure("compute_market_risk_aversion", "market", e, 0.02)
        bus_obj._risk_aversion_cache[cache_key] = value
        return value
