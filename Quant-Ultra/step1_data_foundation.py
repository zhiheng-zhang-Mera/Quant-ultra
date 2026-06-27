"""
Phase 1: Asset Screening and Basic Data Cleaning (Data Foundation)
Optimized with local caching, single BaoStock login, merged data loading, and progress display.
Fixed: Added None checks for BaoStock responses to prevent AttributeError.
Optimized: Multi-threaded concurrent downloading for faster screening.
Optimized: Batch fetching of ST status to eliminate thousands of network calls in step_1_3.
Enhanced: Full caching of screening results, total return prices, and trading status mappings,
          with automatic cache invalidation based on the latest effective trading day (not the run date).
          This handles weekends and holidays correctly: if run on Saturday, the latest effective day is Friday.
"""
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import logging
import akshare as ak
import baostock as bs
import pandas_market_calendars as mcal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

logger = logging.getLogger("DataFoundation")

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_CACHE = PROJECT_ROOT / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)

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
    "CACHE_EXPIRE_DAYS": 7,              # 仅用于历史K线缓存（本地历史数据增量更新）
    "DOWNLOAD_WORKERS": 8,
}


def print_progress(current, total, symbol=None, extra=""):
    """保留原有简单进度打印（用于非并发场景）"""
    if symbol:
        msg = f"处理股票: {symbol} ({current}/{total}) {extra}"
    else:
        msg = f"进度: {current}/{total} {extra}"
    print(f"\r{msg:<80}", end='', flush=True)
    if current == total:
        print()


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
        dates = []
        current = self.start
        while current <= self.end:
            if current.weekday() < 5:
                dates.append(current.replace(tzinfo=self._tz))
            current += timedelta(days=1)
        return dates

    def get_trading_days(self) -> List[datetime]:
        return self._calendar


class DataFetcher:
    _bs_logged = False   # 类变量，确保只登录一次
    _bs_lock = threading.Lock()   # 登录锁

    # ST 状态缓存（批量获取后缓存 1 小时，避免重复请求）
    _st_cache: Optional[Set[str]] = None
    _st_cache_time: Optional[datetime] = None

    @staticmethod
    def _ensure_bs_login():
        if not DataFetcher._bs_logged:
            with DataFetcher._bs_lock:
                if not DataFetcher._bs_logged:
                    bs.login()
                    DataFetcher._bs_logged = True

    @staticmethod
    def _get_cache_path(symbol: str, start_date: str, end_date: str) -> Path:
        return DATA_CACHE / f"{symbol}_{start_date}_{end_date}.parquet"

    @staticmethod
    def _is_cache_valid(cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).days > CONFIG.get("CACHE_EXPIRE_DAYS", 7):
            return False
        return True

    @staticmethod
    def get_historical_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        增量获取历史日线数据（优先缓存，只下载缺失部分）
        start_date 和 end_date 为字符串格式 'YYYY-MM-DD'，用于限定查询范围，
        但缓存会保存全量历史，每次仅追加新数据。
        """
        cache_path = DATA_CACHE / f"{symbol}_history.parquet"
        code = symbol.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')

        # 尝试加载本地缓存
        existing_df = None
        if cache_path.exists():
            try:
                existing_df = pd.read_parquet(cache_path)
                if not existing_df.empty:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    existing_df.sort_values('date', inplace=True)
            except Exception:
                existing_df = None

        # 确定需要下载的日期范围
        if existing_df is not None and not existing_df.empty:
            last_date = existing_df['date'].max()
            # 如果缓存已覆盖到 end_date，则直接返回所需区间
            if last_date >= pd.to_datetime(end_date):
                # 直接从缓存中截取所需区间
                mask = (existing_df['date'] >= pd.to_datetime(start_date)) & (existing_df['date'] <= pd.to_datetime(end_date))
                return existing_df.loc[mask].copy()
            # 否则从 last_date + 1 天开始下载
            fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            fetch_start = start_date  # 无缓存，全量下载

        # 如果 fetch_start <= end_date，则进行网络请求
        if pd.to_datetime(fetch_start) <= pd.to_datetime(end_date):
            # 尝试 AkShare
            try:
                df_new = ak.stock_zh_a_hist(symbol=code, adjust='qfq',
                                            start_date=fetch_start.replace('-', ''),
                                            end_date=end_date.replace('-', ''))
                if not df_new.empty:
                    df_new.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                        '最高': 'high', '最低': 'low', '成交量': 'volume',
                                        '成交额': 'amount'}, inplace=True, errors='ignore')
                    df_new['symbol'] = symbol
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    df_new['date'] = pd.to_datetime(df_new['date'])
                    # 合并
                    if existing_df is not None:
                        combined = pd.concat([existing_df, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
                    else:
                        combined = df_new
                    combined.to_parquet(cache_path, index=False)
                    # 返回所需区间
                    mask = (combined['date'] >= pd.to_datetime(start_date)) & (combined['date'] <= pd.to_datetime(end_date))
                    return combined.loc[mask].copy()
            except Exception as e:
                logger.debug(f"AkShare 增量获取 {symbol} 失败，尝试 BaoStock: {e}")

            # 降级 BaoStock
            try:
                DataFetcher._ensure_bs_login()
                bs_code = f"sh.{code}" if symbol.endswith('.SH') or symbol.startswith('6') else f"sz.{code}"
                rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume,amount",
                                                start_date=fetch_start, end_date=end_date,
                                                frequency="d", adjustflag="2")
                if rs is None:
                    logger.warning(f"BaoStock 返回 None，跳过 {symbol}")
                    return existing_df if existing_df is not None else pd.DataFrame()
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                if data:
                    df_new = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
                    df_new['symbol'] = symbol
                    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    df_new['date'] = pd.to_datetime(df_new['date'])
                    if existing_df is not None:
                        combined = pd.concat([existing_df, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
                    else:
                        combined = df_new
                    combined.to_parquet(cache_path, index=False)
                    mask = (combined['date'] >= pd.to_datetime(start_date)) & (combined['date'] <= pd.to_datetime(end_date))
                    return combined.loc[mask].copy()
                else:
                    # 无新数据，返回已有缓存区间
                    if existing_df is not None:
                        mask = (existing_df['date'] >= pd.to_datetime(start_date)) & (existing_df['date'] <= pd.to_datetime(end_date))
                        return existing_df.loc[mask].copy()
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"BaoStock 增量获取 {symbol} 失败: {e}")
                if existing_df is not None:
                    mask = (existing_df['date'] >= pd.to_datetime(start_date)) & (existing_df['date'] <= pd.to_datetime(end_date))
                    return existing_df.loc[mask].copy()
                return pd.DataFrame()
        else:
            # 缓存已覆盖全部日期，直接返回所需区间
            if existing_df is not None:
                mask = (existing_df['date'] >= pd.to_datetime(start_date)) & (existing_df['date'] <= pd.to_datetime(end_date))
                return existing_df.loc[mask].copy()
            return pd.DataFrame()

    @staticmethod
    def get_dividend_info(symbol: str) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def get_all_stocks() -> pd.DataFrame:
        try:
            df = ak.stock_info_a_code_name()
            df.rename(columns={'code': 'symbol', 'name': 'short_name'}, inplace=True)
            return df
        except Exception:
            try:
                DataFetcher._ensure_bs_login()
                rs = bs.query_all_stock()
                if rs is None:
                    return pd.DataFrame()
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
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
    def get_st_status_batch() -> Set[str]:
        """
        批量获取当前全部 ST / *ST 股票代码（纯数字，如 '000001'）。
        优先 AkShare（东方财富），失败后降级 BaoStock（query_all_stock）。
        结果缓存 1 小时，避免重复请求。
        """
        # 检查缓存（1小时有效）
        if DataFetcher._st_cache is not None and DataFetcher._st_cache_time is not None:
            if (datetime.now() - DataFetcher._st_cache_time).seconds < 3600:
                return DataFetcher._st_cache

        st_codes = set()

        # 1) 优先 AkShare 东方财富 ST 列表
        try:
            df = ak.stock_zh_a_st_em()
            if df is not None and not df.empty:
                # 代码列为 "000001" 格式
                st_codes = set(df["代码"].astype(str).str.strip().tolist())
                DataFetcher._st_cache = st_codes
                DataFetcher._st_cache_time = datetime.now()
                logger.info(f"AkShare 批量获取 ST/*ST 股票成功，共 {len(st_codes)} 只")
                return st_codes
        except Exception as e:
            logger.debug(f"AkShare 获取 ST 列表失败，降级 BaoStock: {e}")

        # 2) 降级 BaoStock query_all_stock
        try:
            DataFetcher._ensure_bs_login()
            rs = bs.query_all_stock()
            if rs is None:
                logger.warning("BaoStock query_all_stock 返回 None")
                return st_codes
            if rs.error_code != "0":
                logger.warning(f"BaoStock query_all_stock 错误: {rs.error_code}")
                return st_codes
            while rs.next():
                row = rs.get_row_data()
                # row 结构: [code, code_name, IPOdate, status]
                # status: 0=正常, 1=ST, 2=*ST, 3=退市整理期, 4=暂停上市
                if row[3] in ["1", "2", "3", "4"]:
                    code = row[0].split(".")[1]  # 提取纯数字
                    st_codes.add(code)
            DataFetcher._st_cache = st_codes
            DataFetcher._st_cache_time = datetime.now()
            logger.info(f"BaoStock 批量获取 ST/*ST 股票成功，共 {len(st_codes)} 只")
            return st_codes
        except Exception as e:
            logger.error(f"BaoStock 获取 ST 列表失败: {e}")
            return st_codes

    @staticmethod
    def get_st_status(symbol: str) -> bool:
        """
        单只股票 ST 状态查询（保留用于兼容，但 step_1_3 已改用批量接口，此方法不再被高频调用）
        """
        try:
            DataFetcher._ensure_bs_login()
            bs_code = f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            rs = bs.query_history_k_data_plus(bs_code, "date,isST", start_date=start, end_date=end, frequency="d")
            if rs is None:
                return False
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if data:
                return data[-1][1] == '1'
            return False
        except Exception:
            return False


# ==================== 缓存加载辅助函数 ====================

def _load_price_cache(bus, cache_path: Path, latest_trading_day: datetime) -> bool:
    """若缓存有效则加载价格数据到 bus，返回是否加载成功"""
    if not cache_path.exists():
        logger.debug(f"价格缓存文件不存在: {cache_path}")
        return False
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            logger.debug("价格缓存文件为空")
            return False
        # 确保日期列为 datetime
        df['date'] = pd.to_datetime(df['date'])
        max_date = df['date'].max()
        # 放宽条件：缓存最大日期 >= 最新交易日即视为有效（避免时区/精度问题）
        if max_date.date() < latest_trading_day.date():
            logger.info(f"价格缓存过期: 缓存最大日期 {max_date.date()} < 最新交易日 {latest_trading_day.date()}")
            return False
        # 写入 bus
        for _, row in df.iterrows():
            sym = row['symbol']
            dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
            price = float(row['price'])
            bus.append_atom(sym, dt, price, "total_return_price", dt)
        logger.info(f"价格缓存加载成功，共 {len(df)} 条记录，最大日期 {max_date.date()}")
        return True
    except Exception as e:
        logger.warning(f"加载价格缓存失败: {e}")
        return False


def _load_status_cache(bus, cache_path: Path, latest_trading_day: datetime) -> bool:
    """若缓存有效则加载交易状态映射到 bus，返回是否加载成功"""
    if not cache_path.exists():
        return False
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return False
        cache_date = pd.to_datetime(df['date'].iloc[0])
        if cache_date.date() < latest_trading_day.date():
            logger.info(f"状态缓存过期: 缓存日期 {cache_date.date()} < 最新交易日 {latest_trading_day.date()}")
            return False
        # 写入 bus
        for _, row in df.iterrows():
            sym = row['symbol']
            mapping = {
                'board': row['board'],
                'is_st': row['is_st'],
                'days_listed': row['days_listed'],
                'limit_up': row['limit_up'],
                'limit_down': row['limit_down'],
                'prev_close': row['prev_close']
            }
            dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
            bus.append_atom(sym, dt, mapping, "trading_status_mapping", dt)
        logger.info(f"状态映射缓存加载成功，共 {len(df)} 只股票")
        return True
    except Exception as e:
        logger.warning(f"加载状态映射缓存失败: {e}")
        return False


# ==================== Phase 1 步骤函数 ====================

def _process_single_stock(symbol: str, start_date: str, end_date: str, bus, now):
    """
    处理单只股票：下载历史、计算adv、获取基本信息。
    返回 (symbol, hist_df, adv, list_date, board) 或 None（如果失败）
    """
    try:
        hist_df = DataFetcher.get_historical_daily(symbol, start_date, end_date)
        if hist_df.empty:
            return None

        adv_series = hist_df['amount'].tail(CONFIG["ADV_WINDOW"])
        adv = adv_series.mean() if len(adv_series) >= CONFIG["ADV_WINDOW"] * 0.5 else 0
        if adv == 0:
            return None

        basic_info = DataFetcher.get_stock_basic_info(symbol)
        list_date_str = basic_info.get('list_date')
        if list_date_str and isinstance(list_date_str, str):
            try:
                list_date = datetime.strptime(list_date_str, '%Y-%m-%d').replace(tzinfo=bus._tz)
            except:
                list_date = datetime(2000, 1, 1, tzinfo=bus._tz)
        else:
            list_date = datetime(2000, 1, 1, tzinfo=bus._tz)

        board = basic_info.get('board', DataFetcher._infer_board(symbol))

        # 写入原子存储（线程安全）
        bus.append_atom(symbol, now, adv, "adv", now - timedelta(days=1))
        bus.append_atom(symbol, now, list_date, "listing_date", now)
        bus.append_atom(symbol, now, board, "board", now)

        return {
            'symbol': symbol,
            'hist_df': hist_df,
            'adv': adv,
            'list_date': list_date,
            'board': board
        }
    except Exception as e:
        logger.debug(f"处理 {symbol} 失败: {e}")
        return None


def step_1_1_screening(context: dict, bus):
    print("[Step 1.1] Asset screening and capacity estimation (with caching & concurrency)")
    fetcher = DataFetcher()
    now = datetime.now(bus._tz)

    # ---- 关键修复：计算最新有效交易日（不包含未来日期） ----
    trading_days = context.get('trading_days_dt', [])
    past_trading_days = [d for d in trading_days if d <= now]
    if not past_trading_days:
        raise RuntimeError("没有可用的历史交易日，请检查日历数据")
    effective_latest_trading_day = past_trading_days[-1]
    # 存入 context，供后续步骤复用
    context['effective_latest_trading_day'] = effective_latest_trading_day
    latest_trading_day = effective_latest_trading_day
    # --------------------------------------------------------

    # ---------- 1. 尝试加载筛选缓存 ----------
    screening_cache = DATA_CACHE / "screening_results.parquet"
    cache_valid = False
    if screening_cache.exists() and latest_trading_day is not None:
        try:
            df_cache = pd.read_parquet(screening_cache)
            if 'cache_date' in df_cache.columns:
                cache_build = pd.to_datetime(df_cache['cache_date'].iloc[0])
                # 比较日期部分（忽略时区）
                if cache_build.date() == latest_trading_day.date():
                    cache_valid = True
                else:
                    logger.info(f"筛选缓存日期不匹配: 缓存 {cache_build.date()} vs 最新 {latest_trading_day.date()}")
        except Exception as e:
            logger.warning(f"读取筛选缓存失败: {e}")

    if cache_valid:
        try:
            df_cache = pd.read_parquet(screening_cache).drop(columns=['cache_date'], errors='ignore')
            assets = df_cache['symbol'].tolist()
            context['assets'] = assets
            context['adv_data'] = {row['symbol']: row['adv'] for _, row in df_cache.iterrows()}

            # 重建 AUM
            aum_capacity = float('inf')
            for _, row in df_cache.iterrows():
                adv = row['adv']
                if adv > 0:
                    single_capacity = (adv * CONFIG["MAX_PARTICIPATION_RATE"]) / \
                                      (CONFIG["MAX_SINGLE_STOCK_WEIGHT"] * CONFIG["EXPECTED_TURNOVER"])
                    aum_capacity = min(aum_capacity, single_capacity)
            context['baseline_aum_limit'] = aum_capacity if aum_capacity != float('inf') else 0

            # 写入 bus（筛选相关）
            for _, row in df_cache.iterrows():
                sym = row['symbol']
                bus.append_atom(sym, now, row['adv'], "adv", now - timedelta(days=1))
                list_date = pd.to_datetime(row['list_date']).to_pydatetime().replace(tzinfo=bus._tz)
                bus.append_atom(sym, now, list_date, "listing_date", now)
                bus.append_atom(sym, now, row['board'], "board", now)

            print(f"[缓存] 加载筛选结果：{len(assets)} 只股票，AUM上限: {context['baseline_aum_limit']:,.0f}")

            # 尝试加载价格和状态缓存
            price_ok = _load_price_cache(bus, DATA_CACHE / "total_return_prices.parquet", latest_trading_day)
            status_ok = _load_status_cache(bus, DATA_CACHE / "trading_status.parquet", latest_trading_day)

            if price_ok and status_ok:
                context['phase1_fully_cached'] = True
                print("[缓存] Phase 1 全部缓存有效，将跳过 Step 1.2 和 1.3")
            return  # 筛选缓存加载完成，结束
        except Exception as e:
            logger.warning(f"加载筛选缓存数据失败，将重新构建: {e}")

    # ---------- 2. 正常流程：并发下载/处理 ----------
    all_stocks_df = fetcher.get_all_stocks()
    if all_stocks_df.empty:
        context['assets'] = []
        return

    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')

    symbols = [row['symbol'] for _, row in all_stocks_df.iterrows() if row['symbol'][0].isdigit()]
    total = len(symbols)
    logger.info(f"开始并发处理 {total} 只股票（线程数: {CONFIG['DOWNLOAD_WORKERS']}）")

    raw_results = []
    with ThreadPoolExecutor(max_workers=CONFIG['DOWNLOAD_WORKERS']) as executor:
        future_to_symbol = {
            executor.submit(_process_single_stock, sym, start_date, end_date, bus, now): sym
            for sym in symbols
        }
        for future in tqdm(as_completed(future_to_symbol), total=total, desc="下载/处理股票", unit="只"):
            sym = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    raw_results.append(result)
                    if 'asset_histories' not in context:
                        context['asset_histories'] = {}
                    context['asset_histories'][result['symbol']] = result['hist_df']
            except Exception as e:
                logger.warning(f"股票 {sym} 处理异常: {e}")

    logger.info(f"成功处理 {len(raw_results)} 只股票（共 {total} 只）")

    # 筛选
    filtered_results = []
    for item in raw_results:
        sym = item['symbol']
        adv = item['adv']
        list_date = item['list_date']
        board = item['board']
        if adv < CONFIG["MIN_ADV_THRESHOLD"]:
            continue
        days_since_list = (now - list_date).days if list_date else 999
        if days_since_list < CONFIG["IPO_SAFETY_DAYS"]:
            continue
        filtered_results.append({
            'symbol': sym,
            'adv': adv,
            'list_date': list_date.strftime('%Y-%m-%d %H:%M:%S'),
            'board': board
        })

    assets = [item['symbol'] for item in filtered_results]
    context['assets'] = assets

    # 计算 AUM 容量
    aum_capacity = float('inf')
    adv_data = {}
    for item in filtered_results:
        sym = item['symbol']
        adv = item['adv']
        adv_data[sym] = adv
        if adv > 0:
            single_capacity = (adv * CONFIG["MAX_PARTICIPATION_RATE"]) / \
                              (CONFIG["MAX_SINGLE_STOCK_WEIGHT"] * CONFIG["EXPECTED_TURNOVER"])
            aum_capacity = min(aum_capacity, single_capacity)
    if aum_capacity == float('inf'):
        aum_capacity = 0
    context['baseline_aum_limit'] = aum_capacity
    context['adv_data'] = adv_data

    print(f"\n[筛选] 通过初筛的资产: {len(assets)} 只，AUM上限: {aum_capacity:,.0f}")

    # ---------- 3. 保存筛选结果到持久化缓存 ----------
    if filtered_results:
        try:
            df_cache = pd.DataFrame(filtered_results)
            # 使用 effective_latest_trading_day 作为缓存日期，而非运行时间
            df_cache['cache_date'] = latest_trading_day.strftime('%Y-%m-%d')
            df_cache.to_parquet(screening_cache, index=False)
            logger.info(f"筛选结果已永久保存至 {screening_cache}")
        except Exception as e:
            logger.warning(f"保存筛选缓存失败: {e}")


def step_1_2_survivorship_and_returns(context: dict, bus, audit):
    print("[Step 1.2] Total return computation with dividends (reusing cached data)")

    # 如果已经完整缓存，直接跳过
    if context.get('phase1_fully_cached', False):
        print("[跳过] 价格缓存已加载")
        return

    assets = context.get('assets', [])
    now = datetime.now(bus._tz)
    # 使用 step_1_1 中计算的最新有效交易日
    latest_trading_day = context.get('effective_latest_trading_day')
    if latest_trading_day is None:
        # 若未传递，则重新计算
        trading_days = context.get('trading_days_dt', [])
        past = [d for d in trading_days if d <= now]
        latest_trading_day = past[-1] if past else None
        context['effective_latest_trading_day'] = latest_trading_day

    cache_path = DATA_CACHE / "total_return_prices.parquet"

    # ---------- 尝试加载价格缓存 ----------
    if cache_path.exists() and latest_trading_day is not None:
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                max_date = df['date'].max()
                if max_date.date() >= latest_trading_day.date():  # 放宽条件
                    # 加载到 bus
                    for _, row in df.iterrows():
                        sym = row['symbol']
                        dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
                        price = float(row['price'])
                        bus.append_atom(sym, dt, price, "total_return_price", dt)
                    print(f"[缓存] 全收益价格加载成功，共 {len(df)} 条记录，最大日期 {max_date.date()}")
                    return
                else:
                    print(f"[缓存] 价格缓存过期: 最大日期 {max_date.date()} < 最新交易日 {latest_trading_day.date()}")
        except Exception as e:
            logger.warning(f"读取价格缓存失败: {e}")

    # ---------- 正常计算 ----------
    asset_histories = context.get('asset_histories', {})
    all_price_data = []  # 用于保存缓存

    for sym in tqdm(assets, desc="写入全收益价格", unit="只"):
        hist_df = asset_histories.get(sym)
        if hist_df is None or hist_df.empty:
            # 尝试重新下载
            try:
                start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
                end_date = now.strftime('%Y-%m-%d')
                hist_df = DataFetcher.get_historical_daily(sym, start_date, end_date)
            except Exception as e:
                logger.warning(f"重新下载 {sym} 失败: {e}")
                continue
        for _, row in hist_df.iterrows():
            dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
            price = float(row['close'])
            bus.append_atom(sym, dt, price, "total_return_price", dt)
            all_price_data.append({'symbol': sym, 'date': dt, 'price': price})
        audit.log_event("TOTAL_RETURN_WARNING", {"symbol": sym, "msg": "Using adjusted close, no dividend reinvestment"})

    # 保存缓存
    if all_price_data:
        try:
            df_cache = pd.DataFrame(all_price_data)
            df_cache.to_parquet(cache_path, index=False)
            logger.info(f"全收益价格已缓存至 {cache_path}")
        except Exception as e:
            logger.warning(f"保存价格缓存失败: {e}")
    print()


def step_1_3_trading_status_mapping(context: dict, bus):
    print("[Step 1.3] Trading status mapping (optimized with batch ST fetching)")

    if context.get('phase1_fully_cached', False):
        print("[跳过] 交易状态映射缓存已加载")
        return

    assets = context.get('assets', [])
    now = datetime.now(bus._tz)
    # 获取有效最新交易日
    latest_trading_day = context.get('effective_latest_trading_day')
    if latest_trading_day is None:
        trading_days = context.get('trading_days_dt', [])
        past = [d for d in trading_days if d <= now]
        latest_trading_day = past[-1] if past else None
        context['effective_latest_trading_day'] = latest_trading_day

    cache_path = DATA_CACHE / "trading_status.parquet"

    # ---------- 尝试加载状态缓存 ----------
    if cache_path.exists() and latest_trading_day is not None:
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                cache_date = pd.to_datetime(df['date'].iloc[0])
                if cache_date.date() >= latest_trading_day.date():  # 放宽条件
                    for _, row in df.iterrows():
                        sym = row['symbol']
                        mapping = {
                            'board': row['board'],
                            'is_st': row['is_st'],
                            'days_listed': row['days_listed'],
                            'limit_up': row['limit_up'],
                            'limit_down': row['limit_down'],
                            'prev_close': row['prev_close']
                        }
                        dt = pd.to_datetime(row['date']).to_pydatetime().replace(tzinfo=bus._tz)
                        bus.append_atom(sym, dt, mapping, "trading_status_mapping", dt)
                    print(f"[缓存] 交易状态映射加载成功，共 {len(df)} 只股票")
                    return
                else:
                    print(f"[缓存] 状态缓存过期: 缓存日期 {cache_date.date()} < 最新交易日 {latest_trading_day.date()}")
        except Exception as e:
            logger.warning(f"读取状态缓存失败: {e}")

    # ---------- 正常计算 ----------
    st_set = DataFetcher.get_st_status_batch()
    logger.info(f"批量获取 ST/*ST 股票数量: {len(st_set)}")

    all_status = []
    for sym in tqdm(assets, desc="映射涨跌停", unit="只"):
        try:
            code = sym.split('.')[0]
            is_st = code in st_set
            board = bus.query_by_pit(sym, now, "board") or DataFetcher._infer_board(sym)
            list_date = bus.query_by_pit(sym, now, "listing_date")
            days_listed = (now - list_date).days if list_date else 999
            prev_date = now - timedelta(days=1)
            prev_price = bus.query_by_pit(sym, prev_date, "total_return_price") or 100.0

            if is_st:
                limit_ratio = 0.05
            elif board in ["科创板", "创业板"]:
                limit_ratio = 0.20
            else:
                limit_ratio = 0.10

            mapping = {
                "board": board,
                "is_st": is_st,
                "days_listed": days_listed,
                "limit_up": prev_price * (1 + limit_ratio),
                "limit_down": prev_price * (1 - limit_ratio),
                "prev_close": prev_price
            }
            bus.append_atom(sym, now, mapping, "trading_status_mapping", now)
            # 用于缓存：日期使用 latest_trading_day 而不是 now
            record = {
                'symbol': sym,
                'board': board,
                'is_st': is_st,
                'days_listed': days_listed,
                'limit_up': mapping['limit_up'],
                'limit_down': mapping['limit_down'],
                'prev_close': prev_price,
                'date': latest_trading_day  # 关键修改：使用有效交易日
            }
            all_status.append(record)
        except Exception as e:
            logger.warning(f"处理 {sym} 交易状态映射失败: {e}")

    # 保存缓存
    if all_status:
        try:
            df_cache = pd.DataFrame(all_status)
            df_cache.to_parquet(cache_path, index=False)
            logger.info(f"交易状态映射已缓存至 {cache_path}")
        except Exception as e:
            logger.warning(f"保存状态缓存失败: {e}")
    print()


def step_1_4_calendar_and_timezone(context: dict, bus):
    print("[Step 1.4] Trading calendar validation")
    if 'trading_days_dt' not in context or not context['trading_days_dt']:
        raise RuntimeError("交易日历未加载")
    print(f"[完成] 交易日历共 {len(context['trading_days_dt'])} 个交易日")


def execute(pipeline_context: dict):
    print("\n[Phase 1] 开始数据基础层构建...")
    bus = pipeline_context['data_bus']
    audit = pipeline_context['audit_logger']

    # 第一步：日历（必须执行）
    step_1_4_calendar_and_timezone(pipeline_context, bus)

    # 第二步：筛选（内部会尝试加载全部缓存）
    step_1_1_screening(pipeline_context, bus)

    # 如果全部缓存已加载，则跳过后续计算
    if not pipeline_context.get('phase1_fully_cached', False):
        step_1_2_survivorship_and_returns(pipeline_context, bus, audit)
        step_1_3_trading_status_mapping(pipeline_context, bus)
    else:
        print("[Phase 1] 所有缓存有效，跳过 Step 1.2 和 1.3")

    pipeline_context['data_foundation_ready'] = True
    print(f"[Phase 1] 完成，资产池 {len(pipeline_context.get('assets', []))} 只")
    return pipeline_context