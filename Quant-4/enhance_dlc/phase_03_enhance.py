# -*- coding: utf-8 -*-
"""
附加增强模块：PIT (Point-in-Time) 市场状态平滑器 & HMM机制分类器 & PIT对齐引擎 (Phase 03)
设计模式：包装器/组合模式
使用方法：实例化以包装原始 Regime Switching 模型，或直接调用引擎类获取PIT数据和HMM状态。
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Tuple

# 尝试导入 hmmlearn，若未安装则抛出友好提示
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    logging.warning("hmmlearn not installed. HMMRegimeClassifier will be unavailable. "
                    "Install via: pip install hmmlearn")

logger = logging.getLogger("Phase03.Enhance")


# ============================================================================
# 原有平滑器（保留并增强）
# ============================================================================
class RegimeSmootherExt:
    """
    马尔可夫状态平滑器：剔除瞬时跳变的伪状态，使状态序列更连续。
    现可兼容HMM预测结果（整数标签）。
    """
    def __init__(self, original_model, lookback=3):
        """
        :param original_model: 任意具有 predict(X) 方法的模型对象
        :param lookback: 平滑窗口（目前仅使用前后各1帧，保留参数供扩展）
        """
        self.original_model = original_model
        self.lookback = lookback

    def predict(self, X):
        """
        获取原始预测并进行平滑。
        :param X: 输入特征（格式需与原始模型一致）
        :return: 平滑后的状态标签
        """
        logger.info("增强注入：应用马尔可夫状态平滑，剔除瞬时跳变微伪状态...")
        raw_pred = self.original_model.predict(X)
        return self._apply_smoothing(raw_pred)

    def _apply_smoothing(self, preds: np.ndarray) -> np.ndarray:
        """简单的三态平滑：若前后相同而中间不同，则将中间修正为前后值"""
        smoothed = np.copy(preds)
        for i in range(1, len(preds) - 1):
            if preds[i - 1] == preds[i + 1] and preds[i] != preds[i - 1]:
                smoothed[i] = preds[i - 1]
        return smoothed


# ============================================================================
# 新增：PIT 时点硬对齐引擎
# ============================================================================
class PITALignmentEngine:
    """
    PIT时点硬对齐引擎：管理双时间轴（财报所属期 vs 披露期），确保在任意回测时间T，
    只能使用T之前已公开披露的财报数据。
    """
    def __init__(self, fundamental_db: pd.DataFrame):
        """
        :param fundamental_db: DataFrame，必须包含列：
            - 'asset'：资产代码
            - 'fiscal_period_end'：财报所属期末日期 (datetime)
            - 'publication_date'：财报公开披露日期 (datetime)
            - 其他字段如 eps, pe, revenue, net_profit 等（可由用户定义）
        """
        self.db = fundamental_db.copy()
        # 确保日期类型
        for col in ['fiscal_period_end', 'publication_date']:
            if col in self.db.columns:
                self.db[col] = pd.to_datetime(self.db[col])
        # 按资产和披露日期排序，便于快速查找
        self.db.sort_values(['asset', 'publication_date'], inplace=True)
        logger.info("PITALignmentEngine 初始化完成，共加载 %d 条财报记录", len(self.db))

    def get_fundamental(self, asset: str, as_of_date: Union[str, datetime, pd.Timestamp],
                        fields: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取在 as_of_date 之前已公开的最新财报数据（按披露日期）。
        :param asset: 资产代码
        :param as_of_date: 查询时点
        :param fields: 需要返回的字段列表，若为None则返回全部字段（除 asset, fiscal_period_end, publication_date）
        :return: 字典 {field: value}，若未找到则返回None
        """
        as_of = pd.to_datetime(as_of_date)
        # 筛选该资产在 as_of 之前已披露的记录
        mask = (self.db['asset'] == asset) & (self.db['publication_date'] <= as_of)
        candidate = self.db.loc[mask]
        if candidate.empty:
            logger.debug("未找到资产 %s 在 %s 之前的财报记录", asset, as_of)
            return None
        # 取最新披露的那一条（publication_date最大）
        latest = candidate.iloc[candidate['publication_date'].argmax()]
        # 构造返回字典
        result = {}
        if fields is None:
            # 返回所有非时间轴字段（排除 asset, fiscal_period_end, publication_date）
            exclude = {'asset', 'fiscal_period_end', 'publication_date'}
            for col in self.db.columns:
                if col not in exclude:
                    result[col] = latest[col]
        else:
            for f in fields:
                if f in latest:
                    result[f] = latest[f]
        return result if result else None

    def get_fundamental_by_fiscal_period(self, asset: str, fiscal_end: Union[str, datetime, pd.Timestamp],
                                         as_of_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
                                         fields: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取特定财报所属期（fiscal_period_end）的数据，且可选要求披露日期早于 as_of_date。
        :param asset: 资产代码
        :param fiscal_end: 财报所属期末日期
        :param as_of_date: 可选，若指定则要求 publication_date <= as_of_date
        :param fields: 返回字段列表
        :return: 字典，若不存在或未披露则返回None
        """
        fe = pd.to_datetime(fiscal_end)
        mask = (self.db['asset'] == asset) & (self.db['fiscal_period_end'] == fe)
        if as_of_date is not None:
            as_of = pd.to_datetime(as_of_date)
            mask &= (self.db['publication_date'] <= as_of)
        candidate = self.db.loc[mask]
        if candidate.empty:
            return None
        # 若有多个（理论上不应重复），取披露日期最新的
        row = candidate.iloc[candidate['publication_date'].argmax()]
        result = {}
        if fields is None:
            exclude = {'asset', 'fiscal_period_end', 'publication_date'}
            for col in self.db.columns:
                if col not in exclude:
                    result[col] = row[col]
        else:
            for f in fields:
                if f in row:
                    result[f] = row[f]
        return result if result else None


# ============================================================================
# 新增：HMM 机制划分器
# ============================================================================
class HMMRegimeClassifier:
    """
    使用隐马尔可夫模型（HMM）对市场环境进行无监督分类。
    选取高频波动率、长短周期均线利差、市场交易量变化率作为状态特征，
    将市场划分为：0-低波牛市, 1-震荡盘整, 2-高波熊市（顺序可自定义）。
    """
    def __init__(self, n_states: int = 3, covariance_type: str = 'full', random_state: int = 42):
        """
        :param n_states: 状态数，默认为3（对应三种市场机制）
        :param covariance_type: HMM协方差类型，可选 'spherical', 'diag', 'full', 'tied'
        :param random_state: 随机种子
        """
        if hmm is None:
            raise ImportError("hmmlearn is required for HMMRegimeClassifier. Please install it.")
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.feature_columns = ['volatility', 'spread_ma', 'volume_change']  # 使用的特征
        self._trained = False

    def _prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        从市场指数日线数据计算特征矩阵。
        :param market_data: DataFrame，必须包含 'close', 'volume'，至少需 'high','low' 以计算波动率。
                            索引为日期。
        :return: (T, n_features) numpy数组
        """
        df = market_data.copy()
        # 保证索引为日期
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 1. 波动率：使用已实现波动率（每日对数收益率平方根），或者可以用Parkinson/GK
        # 简化：使用日收益率标准差（滚动20日）
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # 年化

        # 2. 长短周期均线利差：10日与60日移动平均线之差 / 价格
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['spread_ma'] = (df['ma10'] - df['ma60']) / df['close']

        # 3. 成交量变化率：当日成交量 / 20日均量 - 1
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_change'] = (df['volume'] / df['volume_ma20']) - 1

        # 丢弃NaN
        features = df[self.feature_columns].dropna()
        return features.values

    def fit(self, market_data: pd.DataFrame):
        """
        训练HMM模型。
        :param market_data: 包含市场指数日线数据的DataFrame（必须有 'close', 'high', 'low', 'volume'）
        """
        X = self._prepare_features(market_data)
        if len(X) < self.n_states * 5:
            logger.warning("训练数据过少，可能无法有效估计HMM。样本数=%d", len(X))
        self.model = hmm.GaussianHMM(n_components=self.n_states,
                                     covariance_type=self.covariance_type,
                                     random_state=self.random_state,
                                     n_iter=100)
        self.model.fit(X)
        self._trained = True
        logger.info("HMM训练完成，状态数=%d，样本数=%d", self.n_states, len(X))
        # 存储训练数据以便后续预测时作为参考（非必需）
        self._train_features = X
        return self

    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        对给定市场数据预测状态序列。
        :param market_data: 与训练数据格式相同
        :return: 状态标签数组
        """
        if not self._trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = self._prepare_features(market_data)
        if len(X) == 0:
            raise ValueError("输入数据无法生成有效特征")
        states = self.model.predict(X)
        return states

    def predict_proba(self, market_data: pd.DataFrame) -> np.ndarray:
        """返回状态概率矩阵 (T, n_states)"""
        if not self._trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = self._prepare_features(market_data)
        return self.model.predict_proba(X)

    def get_latest_state(self, market_data: pd.DataFrame) -> int:
        """
        返回最新一天的状态。
        :param market_data: 包含最新日期的数据
        :return: 最新状态标签
        """
        states = self.predict(market_data)
        return states[-1]

    def decode(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用维特比算法解码最可能的状态序列及对数似然。
        :return: (states, logprob)
        """
        if not self._trained:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = self._prepare_features(market_data)
        return self.model.decode(X)


# ============================================================================
# 可选：便捷集成函数，用于将HMM状态注入特征面板
# ============================================================================
def inject_regime_to_features(context: dict, hmm_classifier: HMMRegimeClassifier,
                              market_index_data: pd.DataFrame) -> dict:
    """
    将HMM预测的当前市场状态作为全局因子注入到 context 中。
    :param context: pipeline上下文（需包含 'trading_days_dt_cn' 或当前日期信息）
    :param hmm_classifier: 已训练的 HMMRegimeClassifier 实例
    :param market_index_data: 用于预测的市场指数日线数据（需包含最新日期）
    :return: 更新后的 context，增加 'regime_state_global' 字段
    """
    # 获取最新状态
    latest_state = hmm_classifier.get_latest_state(market_index_data)
    # 也可以获取整个序列，但这里只取最新
    context['regime_state_global'] = latest_state
    context['regime_state_proba'] = hmm_classifier.predict_proba(market_index_data)[-1].tolist()
    logger.info("全局市场机制状态注入成功：状态=%d", latest_state)
    return context


# ============================================================================
# 如果作为独立模块运行，可进行简单测试
# ============================================================================
if __name__ == "__main__":
    # 示例：模拟财报数据库
    logging.basicConfig(level=logging.INFO)
    print("=== 测试 PITALignmentEngine ===")
    fund_data = pd.DataFrame({
        'asset': ['000001.SH', '000001.SH', '000001.SH', 'AAPL'],
        'fiscal_period_end': ['2021-12-31', '2022-12-31', '2023-12-31', '2023-09-30'],
        'publication_date': ['2022-04-15', '2023-04-20', '2024-04-18', '2023-11-02'],
        'eps': [1.2, 1.5, 1.8, 6.1],
        'pe': [10, 12, 9, 28]
    })
    engine = PITALignmentEngine(fund_data)
    result = engine.get_fundamental('000001.SH', '2023-06-01', fields=['eps', 'pe'])
    print("查询结果:", result)  # 应返回2022年年报（披露于2023-04-20）

    print("\n=== 测试 HMMRegimeClassifier ===")
    # 模拟市场数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    close = np.cumsum(np.random.randn(500) * 0.01) + 100
    high = close * (1 + np.abs(np.random.randn(500) * 0.005))
    low = close * (1 - np.abs(np.random.randn(500) * 0.005))
    volume = np.random.randint(1e6, 1e8, size=500)
    market_df = pd.DataFrame({'close': close, 'high': high, 'low': low, 'volume': volume}, index=dates)

    classifier = HMMRegimeClassifier(n_states=3)
    classifier.fit(market_df)
    states = classifier.predict(market_df)
    print("状态序列:", states[-10:])
    proba = classifier.predict_proba(market_df)
    print("最新状态概率:", proba[-1])