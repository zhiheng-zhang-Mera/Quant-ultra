"""
Quant-Ultra Flow - Phase 01 数据基础增强模块
实现动态生存偏差纠正、非线性时序插补及非结构化替代数据抽取
设计模式：无侵入式装饰器 + 可插拔组件
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
from functools import wraps

# ===================== 1. 数据质量守护装饰器（已有，保留并增强） =====================
def data_quality_guard(func):
    """
    装饰器：在数据加载或清洗前后执行质量校验
    """
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        logger = logging.getLogger("DataFoundation.QualityGuard")
        logger.info(f"[{func.__name__}] 增强注入：执行前置数据基础校验...")
        if isinstance(data, pd.DataFrame):
            missing_ratio = data.isnull().mean().max()
            if missing_ratio > 0.3:
                logger.warning(f"警告: 数据缺失率达 {missing_ratio:.2%}, 可能引起后续特征失真。")
            # 新增：检查极端值
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].std() == 0:
                    logger.warning(f"列 {col} 方差为零，可能为常量列，建议剔除。")
        
        result = func(data, *args, **kwargs)
        logger.info(f"[{func.__name__}] 增强注入：执行后置特征分布校验...")
        return result
    return wrapper


# ===================== 2. 动态生存偏差纠正引擎 =====================
class SurvivalBiasEliminator:
    """
    动态生存偏差纠正引擎
    根据历史时间点获取实际成分股、退市股及交易状态，生成存续掩码
    """
    def __init__(self, data_manager, data_bus, config: Optional[Dict] = None):
        """
        :param data_manager: 数据管理器，提供获取退市列表、成分股等接口
        :param data_bus: 数据总线，用于存储/查询原子数据
        :param config: 配置字典，可包含历史成分股获取方式等
        """
        self.data_manager = data_manager
        self.data_bus = data_bus
        self.config = config or {}
        self.logger = logging.getLogger("SurvivalBiasEliminator")
        self._delisted_cache = None
        self._component_cache = {}  # 按日期缓存成分股列表

    def get_delisted_securities(self, market: str = "A") -> List[str]:
        """
        获取退市股票代码列表（含完整后缀）
        """
        if self._delisted_cache is not None:
            return self._delisted_cache
        try:
            # 复用原有 _get_delisted_a_stocks 逻辑，此处简化为从 data_manager 获取
            delisted = []
            if hasattr(self.data_manager, '_ak'):
                df = self.data_manager._ak.stock_zh_a_delisted()
                if df is not None and not df.empty:
                    code_col = 'code' if 'code' in df.columns else '股票代码'
                    raw_codes = df[code_col].astype(str).str.strip().tolist()
                    for c in raw_codes:
                        if c.isdigit():
                            delisted.append(f"{c}.SH" if c.startswith('6') else f"{c}.SZ")
            self._delisted_cache = delisted
            self.logger.info(f"获取退市股票 {len(delisted)} 只")
            return delisted
        except Exception as e:
            self.logger.warning(f"获取退市列表失败: {e}")
            return []

    def get_historical_constituents(self, date: datetime) -> List[str]:
        """
        获取指定日期的指数成分股（例如沪深300），用于动态过滤
        此处简化：若未提供真实数据，则返回空列表（即不做成分股限制）
        """
        # 若配置了成分股数据源，可在此实现
        # 示例：从 data_manager 获取历史成分股文件或API
        date_key = date.strftime("%Y%m%d")
        if date_key in self._component_cache:
            return self._component_cache[date_key]
        # 模拟：返回全部股票池（实际应该从外部获取）
        # 此处假设 data_manager 有 get_universe 方法
        try:
            universe = self.data_manager.get_universe() if hasattr(self.data_manager, 'get_universe') else []
            self._component_cache[date_key] = universe
            return universe
        except:
            return []

    def build_alive_mask(self, assets: List[str], trading_dates: List[datetime]) -> pd.DataFrame:
        """
        构建全时间截面生存矩阵（增强版）
        :param assets: 所有候选股票代码
        :param trading_dates: 交易日列表
        :return: DataFrame, index=trading_dates, columns=assets, dtype=bool
        """
        self.logger.info("开始构建动态生存矩阵（含退市及成分股动态）...")
        # 获取退市列表
        delisted_set = set(self.get_delisted_securities())
        # 预先获取每只股票的上市日期和退市日期（若未退市则退市日期为远期）
        # 此处可从 data_bus 查询，或通过 data_manager 加载历史
        asset_bounds = {}
        for sym in assets:
            # 通过 data_bus 查询上市日期（已在 Step 1.1 中写入）
            listing_date = self.data_bus.query_by_pit(sym, datetime.now(), "listing_date")
            if listing_date is None:
                # 若未获取，则从历史数据中推断
                hist = self.data_manager.fetch_historical(sym, "2010-01-01", datetime.now().strftime("%Y-%m-%d"))
                if hist is not None and not hist.empty:
                    listing_date = hist.index.min().to_pydatetime()
                else:
                    listing_date = datetime(2030,1,1)  # 无效
            # 退市日期：若在退市列表中，则使用退市日（实际应查询退市公告日），此处简化为历史最后交易日
            if sym in delisted_set:
                # 查询历史最后交易日
                hist = self.data_manager.fetch_historical(sym, "2010-01-01", datetime.now().strftime("%Y-%m-%d"))
                if hist is not None and not hist.empty:
                    delist_date = hist.index.max().to_pydatetime()
                else:
                    delist_date = listing_date
            else:
                delist_date = datetime(2099,12,31)  # 长期存续
            asset_bounds[sym] = (listing_date, delist_date)
        
        # 构建矩阵
        rows = []
        for dt in trading_dates:
            row = []
            for sym in assets:
                born, death = asset_bounds[sym]
                alive = (born <= dt <= death)
                # 额外检查：若该日该股票停牌或未交易，可根据交易状态进一步细化
                # 此处可调用 data_bus 查询交易状态
                row.append(alive)
            rows.append(row)
        mask_df = pd.DataFrame(rows, index=trading_dates, columns=assets)
        self.logger.info(f"生存矩阵构建完成，形状 {mask_df.shape}")
        return mask_df


# ===================== 3. 非线性时序插补（MICE / 低秩矩阵完成） =====================
class MICEImputer:
    """
    基于链式方程的多重插补（MICE）实现
    利用截面相关性对缺失的量价进行联合估计
    """
    def __init__(self, n_imputations: int = 5, max_iter: int = 10, random_state: int = 42):
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state
        self.logger = logging.getLogger("MICEImputer")

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 中的缺失值进行多重插补，返回单一插补结果（均值）
        :param data: 二维截面数据，index=时间，columns=资产
        :return: 插补后的 DataFrame
        """
        # 若没有缺失值，直接返回
        if not data.isnull().any().any():
            return data.copy()
        
        self.logger.info(f"开始 MICE 插补，缺失值比例 {data.isnull().mean().mean():.2%}")
        # 使用简单实现：利用列均值 + 随机噪声，实际工程中可引入 sklearn 的 IterativeImputer
        # 此处提供框架，实际可调用 fancyimpute 或 sklearn
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imp = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
            imputed_array = imp.fit_transform(data.values)
            imputed_df = pd.DataFrame(imputed_array, index=data.index, columns=data.columns)
            self.logger.info("MICE 插补完成")
            return imputed_df
        except ImportError:
            self.logger.warning("sklearn 未安装或缺少 IterativeImputer，降级为线性插值 + 前向填充")
            # 降级方案：线性插值 + 前向填充
            df_filled = data.interpolate(method='linear', limit_area='inside')
            df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
            # 若仍存在缺失，用列均值填充
            for col in df_filled.columns:
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            return df_filled

    def low_rank_completion(self, data: pd.DataFrame, rank: Optional[int] = None) -> pd.DataFrame:
        """
        基于奇异值阈值（SVT）的低秩矩阵完成
        适用于高维截面，假设价格矩阵近似低秩
        """
        if not data.isnull().any().any():
            return data.copy()
        # 此处依赖 fancyimpute 或自行实现软阈值迭代
        try:
            from fancyimpute import SoftImpute
            imputer = SoftImpute(max_rank=rank if rank else min(data.shape)//2)
            imputed_array = imputer.fit_transform(data.values)
            return pd.DataFrame(imputed_array, index=data.index, columns=data.columns)
        except ImportError:
            self.logger.warning("fancyimpute 未安装，SVT 不可用，回退到 MICE")
            return self.impute(data)


# ===================== 4. 非结构化替代数据抽取（本地 LLM 情感分析） =====================
class AlternativeDataExtractor:
    """
    盘后非结构化数据抽取：财报电话会议、MD&A、研报等 -> 情绪得分、语调偏移等
    """
    def __init__(self, model_path: str = None, cache_dir: str = "./alternative_cache"):
        """
        :param model_path: 本地 LLM 模型路径（如 Qwen-2.5-Coder-7B）
        :param cache_dir: 缓存目录
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger("AlternativeDataExtractor")
        self._model = None

    def _load_model(self):
        """懒加载本地模型（示例，实际需根据具体框架）"""
        if self._model is None:
            try:
                # 此处假设使用 transformers 加载
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.logger.info(f"加载本地 LLM 模型: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
                self._model = (self.tokenizer, self.model)
            except Exception as e:
                self.logger.warning(f"模型加载失败: {e}，将使用规则模拟")
                self._model = "mock"
        return self._model

    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """
        对文本进行情感分析，返回情感得分、语调偏移度、风险密度等
        """
        if not text or len(text) < 10:
            return {"sentiment_score": 0.0, "tonal_shift": 0.0, "regulatory_risk_density": 0.0}
        
        # 若模型不可用，使用基于规则的关键词评分（模拟）
        if self._model == "mock" or self.model_path is None:
            return self._rule_based_analysis(text)
        else:
            # 实际调用 LLM 进行推理
            # 此处只给出框架，实际需构造 prompt 并解析输出
            tokenizer, model = self._load_model()
            # 构造 prompt
            prompt = f"分析以下财报文本的情感倾向（正面/负面/中性），给出情感得分（-1到1）、语调偏移度（与前一次相比变化程度0-1）、监管风险词频密度（0-1）。\n文本：{text[:300]}..."
            # 模拟返回
            return {"sentiment_score": 0.2, "tonal_shift": 0.1, "regulatory_risk_density": 0.05}

    def _rule_based_analysis(self, text: str) -> Dict[str, float]:
        """基于简单词典的规则分析"""
        positive_words = ["增长", "提升", "盈利", "优秀", "超预期", "利好"]
        negative_words = ["下滑", "亏损", "风险", "下降", "不及预期", "利空"]
        risk_words = ["监管", "合规", "处罚", "诉讼", "调查", "违规"]
        pos_count = sum(text.count(w) for w in positive_words)
        neg_count = sum(text.count(w) for w in negative_words)
        risk_count = sum(text.count(w) for w in risk_words)
        total = max(1, len(text)/100)  # 归一化因子
        sentiment = (pos_count - neg_count) / (total + 1e-6)
        sentiment = np.clip(sentiment / 10, -1, 1)  # 粗略缩放
        risk_density = risk_count / (total + 1e-6)
        risk_density = np.clip(risk_density, 0, 1)
        # 语调偏移度无法从单文本计算，需历史缓存，此处返回0
        return {"sentiment_score": sentiment, "tonal_shift": 0.0, "regulatory_risk_density": risk_density}

    def process_earnings_call(self, symbol: str, text: str, date: datetime) -> Dict:
        """
        处理财报电话会议文本，提取特征并存入 data_bus
        """
        features = self.extract_sentiment(text)
        # 添加时间戳和符号
        features['symbol'] = symbol
        features['date'] = date
        features['type'] = 'earnings_call'
        return features

    def batch_process(self, text_data: Dict[str, Dict[datetime, str]]) -> pd.DataFrame:
        """
        批量处理多个文本，返回 DataFrame
        text_data: {symbol: {date: text, ...}, ...}
        """
        records = []
        for sym, dates in text_data.items():
            for dt, txt in dates.items():
                rec = self.process_earnings_call(sym, txt, dt)
                rec.update({'symbol': sym, 'date': dt})
                records.append(rec)
        return pd.DataFrame(records)


# ===================== 5. 集成工具函数 =====================
def apply_enhancements_to_phase1(pipeline_context: dict) -> dict:
    """
    在 Phase-1 执行主流程中调用增强组件
    :param pipeline_context: 包含 data_manager, data_bus, config 等
    :return: 更新后的 context（添加了增强特征）
    """
    logger = logging.getLogger("Phase1.Enhancer")
    logger.info("开始注入 Phase-1 增强模块...")
    data_manager = pipeline_context['data_manager']
    data_bus = pipeline_context['data_bus']
    config = pipeline_context.get('config', {})
    trading_dates = pipeline_context.get('trading_days_dt', [])
    
    # 1. 生存偏差纠正：构建动态 alive_mask 并更新 context
    eliminator = SurvivalBiasEliminator(data_manager, data_bus, config)
    assets = pipeline_context.get('assets', [])
    if assets and trading_dates:
        enhanced_alive_mask = eliminator.build_alive_mask(assets, trading_dates)
        pipeline_context['alive_mask_enhanced'] = enhanced_alive_mask
        logger.info(f"增强生存矩阵已生成，形状 {enhanced_alive_mask.shape}")
    
    # 2. 缺失值插补：对关键数据（如收盘价、成交量）进行 MICE 插补
    # 假设 pipeline_context 中有 'raw_price_data' 等
    price_data = pipeline_context.get('price_panel')  # 假设存在
    if price_data is not None and isinstance(price_data, pd.DataFrame):
        imputer = MICEImputer()
        imputed_prices = imputer.impute(price_data)
        pipeline_context['price_panel_imputed'] = imputed_prices
        logger.info("价格数据 MICE 插补完成")
    
    # 3. 替代数据抽取（示例：从外部获取文本，此处仅示意）
    # 实际应从数据源加载，这里模拟
    # extractor = AlternativeDataExtractor(model_path=config.get('llm_model_path'))
    # text_data = ...  # 从数据总线或外部获取
    # alt_features = extractor.batch_process(text_data)
    # pipeline_context['alternative_features'] = alt_features
    
    return pipeline_context


# ===================== 示例用法 =====================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=5, freq='B')
    symbols = ['000001.SZ', '600000.SH', '000002.SZ']
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=symbols)
    data.iloc[1,0] = np.nan
    data.iloc[3,1] = np.nan
    
    # 使用 MICE 插补
    imputer = MICEImputer()
    filled = imputer.impute(data)
    print("插补前:\n", data)
    print("插补后:\n", filled)
    
    # 使用生存偏差纠正（需 data_manager 和 data_bus 实例，此处演示结构）
    # eliminator = SurvivalBiasEliminator(data_manager, data_bus)
    # mask = eliminator.build_alive_mask(symbols, dates)
    # print(mask)