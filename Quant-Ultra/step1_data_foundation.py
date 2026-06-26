"""
Phase 1: Asset Screening and Basic Data Cleaning (Data Foundation)
Now uses the unified PITDataBus from main.py, no internal redefinition.
"""
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import akshare as ak
import baostock as bs
import pandas_market_calendars as mcal

logger = logging.getLogger("DataFoundation")

# 全局配置（与 main.py 保持一致）
CONFIG = {
    "ADV_WINDOW": 20,
    "MIN_ADV_THRESHOLD": 1e7,
    "IPO_SAFETY_DAYS": 20,
    "MAX_PARTICIPATION_RATE": 0.05,
    "EXPECTED_TURNOVER": 0.05,
    "MAX_SINGLE_STOCK_WEIGHT": 0.05,
    "DEFAULT_RESIDUAL_RATE": 0.0,
    "TIMEZONE": "Asia/Shanghai",
    "MS_PRECISION": True,
}

# 交易日历服务（独立，可保留）
class TradingCalendar:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start = start_date
        self.end = end_date
        self._tz = pytz.timezone(CONFIG["TIMEZONE"])
        self._calendar = self._generate_calendar()

    def _generate_calendar(self) -> List[datetime]:
        try:
            sse = mcal.get_calendar('SSE')
            schedule = sse.schedule(start_date=self.start.strftime('%Y-%m-%d'),
                                    end_date=self.end.strftime('%Y-%m-%d'))
            dates = [dt.to_pydatetime().replace(tzinfo=self._tz) for dt in schedule.index]
            if dates:
                return dates
        except Exception:
            pass
        try:
            trade_df = ak.tool_trade_date_hist_sina()
            trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            dates = [dt.to_pydatetime().replace(tzinfo=self._tz) for dt in trade_df['trade_date'].values
                     if self.start <= dt.to_pydatetime() <= self.end]
            if dates:
                return dates
        except Exception:
            pass
        # 兜底
        dates = []
        current = self.start
        while current <= self.end:
            if current.weekday() < 5:
                dates.append(current.replace(tzinfo=self._tz))
            current += timedelta(days=1)
        return dates

    def get_trading_days(self) -> List[datetime]:
        return self._calendar

# 数据获取辅助（可保留原有 DataFetcher）
class DataFetcher:
    @staticmethod
    def get_all_stocks() -> pd.DataFrame:
        try:
            df = ak.stock_info_a_code_name()
            df.rename(columns={'code': 'symbol', 'name': 'short_name'}, inplace=True)
            return df
        except Exception:
            try:
                bs.login()
                rs = bs.query_all_stock()
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                bs.logout()
                df = pd.DataFrame(data, columns=rs.fields)
                df.rename(columns={'code': 'symbol'}, inplace=True)
                return df
            except Exception:
                return pd.DataFrame()

    @staticmethod
    def get_stock_basic_info(symbol: str) -> Dict:
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            info = df.set_index('item')['value'].to_dict()
            return {
                'symbol': symbol,
                'short_name': info.get('股票简称', ''),
                'list_date': info.get('上市时间', None),
                'board': DataFetcher._infer_board(symbol),
            }
        except Exception:
            return {'symbol': symbol, 'list_date': None, 'board': '未知'}

    @staticmethod
    def _infer_board(symbol: str) -> str:
        if symbol.startswith('688'):
            return '科创板'
        elif symbol.startswith('300') or symbol.startswith('301'):
            return '创业板'
        elif symbol.startswith('600') or symbol.startswith('601') or symbol.startswith('603'):
            return '主板'
        elif symbol.startswith('000') or symbol.startswith('001') or symbol.startswith('002') or symbol.startswith('003'):
            return '主板'
        else:
            return '其他'

    @staticmethod
    def get_historical_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = symbol.replace('.SH', '').replace('.SZ', '')
        try:
            df = ak.stock_zh_a_hist(symbol=code, adjust='qfq',
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''))
            if df.empty:
                raise Exception("empty")
            df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                               '最高': 'high', '最低': 'low', '成交量': 'volume',
                               '成交额': 'amount'}, inplace=True, errors='ignore')
            df['symbol'] = symbol
            return df
        except Exception:
            try:
                bs.login()
                bs_code = f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
                rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume,amount",
                                                  start_date=start_date, end_date=end_date,
                                                  frequency="d", adjustflag="2")
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                bs.logout()
                if data:
                    df = pd.DataFrame(data, columns=["date","open","high","low","close","volume","amount"])
                    df['symbol'] = symbol
                    return df
                else:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()

    @staticmethod
    def get_st_status(symbol: str) -> bool:
        try:
            bs.login()
            bs_code = f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            rs = bs.query_history_k_data_plus(bs_code, "date,isST", start_date=start, end_date=end, frequency="d")
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            bs.logout()
            if data:
                return data[-1][1] == '1'
            return False
        except Exception:
            return False

# 步骤实现（使用传入的 bus）
def step_1_1_screening(context: dict, bus, calendar: TradingCalendar):
    print("[Step 1.1] Asset screening and capacity estimation")
    fetcher = DataFetcher()
    now = datetime.now(bus._tz)
    all_stocks_df = fetcher.get_all_stocks()
    if all_stocks_df.empty:
        context['assets'] = []
        return
    raw_assets = []
    for _, row in all_stocks_df.iterrows():
        symbol = row['symbol']
        if not symbol[0].isdigit():
            continue
        basic_info = fetcher.get_stock_basic_info(symbol)
        start_date = (now - timedelta(days=CONFIG["ADV_WINDOW"] * 3)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        hist_df = fetcher.get_historical_daily(symbol, start_date, end_date)
        if hist_df.empty:
            continue
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.sort_values('date')
        adv_series = hist_df['amount'].tail(CONFIG["ADV_WINDOW"])
        adv = adv_series.mean() if len(adv_series) >= CONFIG["ADV_WINDOW"] * 0.5 else 0
        bus.append_atom(symbol, now, adv, "adv", now - timedelta(days=1))
        list_date_str = basic_info.get('list_date')
        if list_date_str and isinstance(list_date_str, str):
            try:
                list_date = datetime.strptime(list_date_str, '%Y-%m-%d').replace(tzinfo=bus._tz)
            except:
                list_date = datetime(2000, 1, 1, tzinfo=bus._tz)
        else:
            list_date = datetime(2000, 1, 1, tzinfo=bus._tz)
        bus.append_atom(symbol, now, list_date, "listing_date", now)
        bus.append_atom(symbol, now, basic_info.get('board', '未知'), "board", now)
        raw_assets.append({'symbol': symbol, 'list_date': list_date, 'adv': adv})
    filtered = []
    for info in raw_assets:
        sym = info['symbol']
        adv = info['adv']
        if adv < CONFIG["MIN_ADV_THRESHOLD"]:
            continue
        days_since_list = (now - info['list_date']).days if info['list_date'] else 999
        if days_since_list < CONFIG["IPO_SAFETY_DAYS"]:
            continue
        filtered.append(sym)
    context['assets'] = filtered
    # 容量估算
    aum_capacity = float('inf')
    for sym in filtered:
        adv = bus.query_by_pit(sym, now, "adv")
        if adv and adv > 0:
            single_capacity = (adv * CONFIG["MAX_PARTICIPATION_RATE"]) / \
                              (CONFIG["MAX_SINGLE_STOCK_WEIGHT"] * CONFIG["EXPECTED_TURNOVER"])
            aum_capacity = min(aum_capacity, single_capacity)
    if aum_capacity == float('inf'):
        aum_capacity = 0
    context['baseline_aum_limit'] = aum_capacity
    context['adv_data'] = {sym: bus.query_by_pit(sym, now, "adv") for sym in filtered}
    print(f"[筛选] 通过初筛的资产: {len(filtered)} 只，AUM上限: {aum_capacity:,.0f}")

def step_1_2_survivorship_and_returns(context: dict, bus, audit):
    print("[Step 1.2] Total return computation")
    assets = context.get('assets', [])
    now = datetime.now(bus._tz)
    fetcher = DataFetcher()
    for sym in assets:
        try:
            start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            hist_df = fetcher.get_historical_daily(sym, start_date, end_date)
            if hist_df.empty:
                continue
            for _, row in hist_df.iterrows():
                dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
                price = float(row['close'])
                bus.append_atom(sym, dt, price, "total_return_price", dt)
        except Exception as e:
            logger.warning(f"处理 {sym} 全收益率失败: {e}")
    print("[完成] 全收益率价格序列已生成")

def step_1_3_trading_status_mapping(context: dict, bus):
    print("[Step 1.3] Trading status mapping")
    assets = context.get('assets', [])
    now = datetime.now(bus._tz)
    fetcher = DataFetcher()
    for sym in assets:
        try:
            board = bus.query_by_pit(sym, now, "board")
            if board is None:
                board = DataFetcher._infer_board(sym)
            is_st = fetcher.get_st_status(sym)
            list_date = bus.query_by_pit(sym, now, "listing_date")
            days_listed = (now - list_date).days if list_date else 999
            prev_date = now - timedelta(days=1)
            prev_price = bus.query_by_pit(sym, prev_date, "total_return_price")
            if prev_price is None:
                prev_price = 100.0
            if is_st:
                limit_ratio = 0.05
            elif board in ["科创板", "创业板"]:
                limit_ratio = 0.20
            else:
                limit_ratio = 0.10
            mapping = {
                "board": board, "is_st": is_st, "days_listed": days_listed,
                "limit_up": prev_price * (1 + limit_ratio),
                "limit_down": prev_price * (1 - limit_ratio),
                "prev_close": prev_price
            }
            bus.append_atom(sym, now, mapping, "trading_status_mapping", now)
        except Exception as e:
            logger.warning(f"处理 {sym} 交易状态映射失败: {e}")

def step_1_4_calendar_and_timezone(context: dict, bus):
    print("[Step 1.4] Trading calendar validation")
    # 已经由 main 预加载，此处仅校验
    if 'trading_days_dt' not in context or not context['trading_days_dt']:
        raise RuntimeError("交易日历未加载")
    print(f"[完成] 交易日历共 {len(context['trading_days_dt'])} 个交易日")

def execute(pipeline_context: dict):
    print("\n[Phase 1] 开始数据基础层构建...")
    bus = pipeline_context['data_bus']          # 使用主调度器传入的统一 bus
    audit = pipeline_context['audit_logger']
    # 执行子步骤
    step_1_4_calendar_and_timezone(pipeline_context, bus)
    # 创建日历用于筛选（可以复用 bus 中的日历，但这里用 TradingCalendar 独立）
    now = datetime.now(bus._tz)
    cal = TradingCalendar(now - timedelta(days=400), now)
    step_1_1_screening(pipeline_context, bus, cal)
    step_1_2_survivorship_and_returns(pipeline_context, bus, audit)
    step_1_3_trading_status_mapping(pipeline_context, bus)
    pipeline_context['data_foundation_ready'] = True
    print(f"[Phase 1] 完成，资产池 {len(pipeline_context.get('assets', []))} 只")
    return pipeline_context