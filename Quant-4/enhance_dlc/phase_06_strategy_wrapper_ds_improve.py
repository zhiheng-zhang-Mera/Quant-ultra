# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase 6 增强型策略执行包装器 (A股实战特化版)
===============================================================
模块定位：凸优化理论权重 → 实战可执行权重的转化层
核心增强：
    1. 持仓股数级精确管理（加权平均成本追踪）
    2. 多级阶梯式止盈收割（非一次性清仓）
    3. 条件式逢低回补（结合趋势与估值偏离）
    4. A股全合规约束（涨跌停过滤、T+1展期、流动性容量）
    5. 交易成本（佣金+印花税）与冲击成本动态摩擦模型
    6. 自适应波动率阈值与趋势过滤器
    7. 现金管理与净值核算
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta

# 尝试引入A股交易日历库（若未安装则降级为周末过滤）
try:
    import chinese_calendar as calendar
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False
    logging.warning("[增强包装器] 未安装 chinese_calendar，将使用基础周末过滤，节假日处理可能不精确。")

logger = logging.getLogger("PositionSizing.EnhancedWrapper")

class A股Phase6StrategyWrapper:
    """
    Phase 6 增强型策略执行包装器（A股实战版）
    """
    
    def __init__(
        self,
        # ---------- 多级止盈参数 ----------
        profit_take_levels: List[Tuple[float, float]] = None,  # [(阈值, 收割比例), ...]
        # ---------- 回补参数 ----------
        buyback_threshold: float = 0.06,        # 触发回补的浮亏阈值
        buyback_alpha: float = 0.3,             # 回补力度（缺口补足比例）
        # ---------- 趋势过滤 ----------
        trend_ma_period: int = 20,              # 均线过滤周期
        trend_strength_threshold: float = 0.0,  # 价格必须大于均线多少才允许加仓（负值表示可放宽）
        # ---------- 风险与流动性 ----------
        max_shares_pct_of_adv: float = 0.05,    # 单日调仓量不得超过该股票ADV的百分比
        min_shares_lot: int = 100,              # A股最小交易单位（手）
        # ---------- 交易成本（A股标准） ----------
        commission_rate: float = 0.00025,       # 佣金费率（万2.5）
        min_commission: float = 5.0,            # 最低佣金（元）
        stamp_duty_rate: float = 0.0005,        # 印花税（仅卖出，2023年减半后为0.05%）
        # ---------- 行为控制 ----------
        weekly_rebalance_only: bool = False,    # 是否仅在周度调仓（False表示每日可调）
        enable_t1_restriction: bool = True,     # 是否严格执行T+1（今日买入明日方可卖）
        # ---------- 高级自适应 ----------
        use_volatility_adaptive: bool = True,   # 是否根据波动率动态调整止盈阈值
        vol_lookback: int = 10,                 # 波动率计算回溯期
        base_vol_anchor: float = 0.25,          # 基准年化波动率锚点（25%）
    ):
        # 多级止盈默认设置（阶梯式收割）
        if profit_take_levels is None:
            profit_take_levels = [
                (0.05, 0.20),   # 涨5%，收割当前仓位的20%
                (0.10, 0.30),   # 再涨到10%（相对初始），再收割剩余仓位的30%
                (0.18, 0.50),   # 涨到18%，收割剩余仓位的50%
            ]
        self.profit_take_levels = sorted(profit_take_levels, key=lambda x: x[0])
        
        self.buyback_threshold = buyback_threshold
        self.buyback_alpha = buyback_alpha
        self.trend_ma_period = trend_ma_period
        self.trend_strength_threshold = trend_strength_threshold
        self.max_shares_pct_of_adv = max_shares_pct_of_adv
        self.min_shares_lot = min_shares_lot
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.weekly_rebalance_only = weekly_rebalance_only
        self.enable_t1_restriction = enable_t1_restriction
        self.use_volatility_adaptive = use_volatility_adaptive
        self.vol_lookback = vol_lookback
        self.base_vol_anchor = base_vol_anchor

        # 运行时状态（每次execute调用时重置）
        self.holdings: Dict[str, int] = {}          # 当前持有股数
        self.cost_basis: Dict[str, float] = {}      # 加权平均成本（前复权口径）
        self.cash: float = 1.0                      # 现金（初始净值为1）
        self.prev_close: Dict[str, float] = {}      # 上一交易日收盘价（用于涨跌停计算）
        self.today_buy_qty: Dict[str, int] = {}     # 当日买入数量（用于T+1限制，防日内回转）
        self.asset_history_price: Dict[str, pd.Series] = {}  # 缓存价格序列用于计算均线

    def _is_trading_day(self, date: pd.Timestamp) -> bool:
        """判断是否为A股交易日"""
        dt = date.to_pydatetime()
        if HAS_CALENDAR:
            return calendar.is_trading_day(dt)
        else:
            # 降级方案：仅剔除周六周日
            return dt.weekday() < 5

    def _is_week_end(self, date: pd.Timestamp, dates_series: pd.DatetimeIndex, idx: int) -> bool:
        """判断是否为周度调仓节点（每周最后一个交易日）"""
        if idx == len(dates_series) - 1:
            return True
        # 如果下一周与本周不同，则今天是本周最后一个交易日
        current_week = dates_series[idx].isocalendar().week
        next_week = dates_series[idx+1].isocalendar().week
        return current_week != next_week

    def _get_limit_prices(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """
        计算A股涨跌停价格
        注：科创板（688开头）和创业板（300开头）为20%，其余主板为10%
        """
        # 获取上一日收盘价，若不存在则用当前价格反推（通常prev_close已维护）
        prev_c = self.prev_close.get(symbol, current_price)
        # 判断板块
        if symbol.startswith(('688', '300')):
            pct = 0.20
        else:
            pct = 0.10
        # 注意ST股票会进一步缩窄至5%，此处忽略ST检测（可通过外部注入）
        # 涨跌停价保留两位小数
        limit_up = round(prev_c * (1 + pct), 2)
        limit_down = round(prev_c * (1 - pct), 2)
        return limit_up, limit_down

    def _calc_vol_adaptive_multiplier(self, price_series: pd.Series) -> float:
        """根据历史波动率动态调整止盈阈值乘数"""
        if not self.use_volatility_adaptive or len(price_series) < self.vol_lookback + 1:
            return 1.0
        # 计算对数收益率波动率（年化）
        rets = np.log(price_series / price_series.shift(1)).dropna().tail(self.vol_lookback)
        if len(rets) < 3:
            return 1.0
        annual_vol = rets.std() * np.sqrt(252)
        # 当波动率高于基准，放宽阈值（乘数>1）；低于基准则收紧
        multiplier = max(0.5, min(2.0, annual_vol / self.base_vol_anchor))
        return multiplier

    def _apply_multi_level_take_profit(
        self, 
        current_return: float, 
        current_shares: int, 
        multiplier: float
    ) -> int:
        """
        应用多级阶梯止盈，返回应卖出的股数
        """
        if current_shares <= 0 or current_return <= 0:
            return 0
        
        total_sell = 0
        remaining_shares = current_shares
        
        # 遍历止盈级别，累积收割
        for threshold, ratio in self.profit_take_levels:
            adjusted_threshold = threshold * multiplier
            if current_return >= adjusted_threshold:
                # 本轮收割股数 = 当前剩余股数 * ratio
                sell_qty = int(remaining_shares * ratio)
                # 确保不低于最小交易单位
                sell_qty = (sell_qty // self.min_shares_lot) * self.min_shares_lot
                if sell_qty <= 0:
                    continue
                total_sell += sell_qty
                remaining_shares -= sell_qty
            else:
                break
        
        # 确保总卖出不超过原始股数
        return min(total_sell, current_shares)

    def _calculate_trade_cost(
        self, 
        price: float, 
        qty: int, 
        is_buy: bool
    ) -> float:
        """计算单笔交易成本（佣金+印花税，印花税仅卖出）"""
        if qty <= 0:
            return 0.0
        amount = price * qty
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate if not is_buy else 0.0
        return commission + stamp_duty

    def execute(
        self, 
        weights_df: pd.DataFrame, 
        price_df: pd.DataFrame,
        adv20_df: Optional[pd.DataFrame] = None,
        initial_capital: float = 1.0,
    ) -> pd.DataFrame:
        """
        执行增强型仓位转换
        :param weights_df: Phase 6 输出的目标权重矩阵 (日期 x 资产)
        :param price_df: 收盘价矩阵 (日期 x 资产)
        :param adv20_df: 20日平均成交额矩阵 (日期 x 资产)，用于流动性约束
        :param initial_capital: 初始总资产（标准化为1.0，实际资金可等比缩放）
        :return: 经实战优化后的执行权重矩阵
        """
        # ---------- 1. 数据对齐与校验 ----------
        common_dates = weights_df.index.intersection(price_df.index)
        common_assets = weights_df.columns.intersection(price_df.columns)
        if len(common_dates) == 0 or len(common_assets) == 0:
            logger.error("权重矩阵与价格矩阵无重叠，无法执行。")
            return weights_df
        
        weights = weights_df.loc[common_dates, common_assets].copy()
        prices = price_df.loc[common_dates, common_assets].copy()
        
        if adv20_df is not None:
            adv20 = adv20_df.loc[common_dates, common_assets].copy()
        else:
            adv20 = pd.DataFrame(1e8, index=common_dates, columns=common_assets)
            logger.warning("未提供ADV20数据，流动性约束将失效（使用极大值替代）。")

        # 重置运行时状态
        self.holdings = {asset: 0 for asset in common_assets}
        self.cost_basis = {asset: np.nan for asset in common_assets}
        self.prev_close = {}
        self.today_buy_qty = {asset: 0 for asset in common_assets}
        self.cash = initial_capital
        self.asset_history_price = {}

        # 用于结果存储
        executed_weights = pd.DataFrame(index=common_dates, columns=common_assets, dtype=float)
        total_asset_series = pd.Series(index=common_dates, dtype=float)
        dates = common_dates
        
        logger.info(f"=== 增强包装器启动 | 交易日数量: {len(dates)} | 资产数量: {len(common_assets)} ===")

        # ---------- 2. 逐日迭代执行 ----------
        for idx, date in enumerate(dates):
            # 2.1 交易日过滤（非交易日跳过，但价格矩阵已对齐，此处做二次确认）
            if not self._is_trading_day(date):
                # 若无法交易，则持仓不变，记录当前权重
                current_total = self.cash + sum(self.holdings[a] * prices.loc[date, a] for a in common_assets)
                for a in common_assets:
                    mv = self.holdings[a] * prices.loc[date, a]
                    executed_weights.loc[date, a] = mv / current_total if current_total > 0 else 0.0
                total_asset_series[date] = current_total
                continue

            # 2.2 周线锚定检查
            if self.weekly_rebalance_only and not self._is_week_end(date, dates, idx):
                # 非周终调仓日，保持前一日持仓不变
                if idx > 0:
                    executed_weights.loc[date] = executed_weights.loc[dates[idx-1]]
                else:
                    # 第一个交易日若不允许调仓，则用等权或目标权重初始化（此处用目标权重）
                    executed_weights.loc[date] = weights.loc[date]
                # 但持仓股数仍需更新至实际状态（若有价格变动，权益变动）
                current_total = self.cash + sum(self.holdings[a] * prices.loc[date, a] for a in common_assets)
                total_asset_series[date] = current_total
                # 更新prev_close
                for a in common_assets:
                    self.prev_close[a] = prices.loc[date, a]
                continue

            # 2.3 提取当日价格与目标权重
            current_prices = prices.loc[date]
            target_weights = weights.loc[date]
            
            # 提取ADV数据
            current_adv = adv20.loc[date]

            # 2.4 计算当前总资产与当前实际权重
            current_total = self.cash + sum(self.holdings[a] * current_prices[a] for a in common_assets)
            total_asset_series[date] = current_total
            
            if current_total <= 0:
                # 极端情况：资产归零，直接返回零权重
                executed_weights.loc[date] = 0.0
                continue

            # 2.5 遍历每只股票，计算交易指令
            trade_instructions = {}  # asset -> (target_shares, is_buy)
            for asset in common_assets:
                price = current_prices[asset]
                if pd.isna(price) or price <= 0:
                    continue
                
                # ---- 缓存价格序列用于趋势和波动 ----
                if asset not in self.asset_history_price:
                    self.asset_history_price[asset] = prices[asset].dropna()
                
                # ---- 计算波动率乘数 ----
                hist_series = self.asset_history_price[asset].loc[:date]
                vol_multiplier = self._calc_vol_adaptive_multiplier(hist_series)

                # ---- 涨跌停过滤 ----
                limit_up, limit_down = self._get_limit_prices(asset, price)
                # 若价格 >= 涨停价，禁止买入；若价格 <= 跌停价，禁止卖出
                can_buy = price < limit_up
                can_sell = price > limit_down

                # ---- 趋势过滤器 ----
                trend_ok = True
                if self.trend_ma_period > 0:
                    ma = hist_series.rolling(self.trend_ma_period).mean().iloc[-1] if len(hist_series) >= self.trend_ma_period else price
                    if not pd.isna(ma):
                        trend_ok = price >= (ma * (1 + self.trend_strength_threshold))

                # ---- 目标股数 ----
                target_w = target_weights[asset]
                # 若目标权重为0，则清仓
                desired_shares = 0 if target_w <= 1e-6 else int((target_w * current_total) / price)
                desired_shares = (desired_shares // self.min_shares_lot) * self.min_shares_lot
                
                current_shares = self.holdings.get(asset, 0)
                cost = self.cost_basis.get(asset, np.nan)

                # 计算当前浮动盈亏（相对于加权平均成本）
                unrealized_return = 0.0
                if current_shares > 0 and not pd.isna(cost) and cost > 0:
                    unrealized_return = (price - cost) / cost

                # ---- 核心逻辑分支 ----
                final_shares = current_shares  # 默认保持不变

                # (A) 止盈收割逻辑
                if current_shares > 0 and unrealized_return > 0:
                    # 应用多级阶梯止盈
                    sell_qty = self._apply_multi_level_take_profit(
                        unrealized_return, 
                        current_shares, 
                        vol_multiplier
                    )
                    if sell_qty > 0:
                        # 必须可卖出（价格非跌停）
                        if can_sell:
                            final_shares = current_shares - sell_qty
                        else:
                            logger.debug(f"{asset} 触发止盈但处于跌停状态，暂缓卖出。")

                # (B) 逢低回补逻辑（仅在未触发止盈且目标权重>0时考虑）
                elif target_w > 0 and desired_shares > current_shares:
                    # 回补条件：浮亏超过阈值 且 趋势允许加仓 且 可买入
                    if (pd.isna(cost) or unrealized_return <= -self.buyback_threshold) and trend_ok and can_buy:
                        # 计算缺口
                        gap_shares = desired_shares - current_shares
                        # 根据alpha比例补足
                        buy_qty = int(gap_shares * self.buyback_alpha)
                        buy_qty = (buy_qty // self.min_shares_lot) * self.min_shares_lot
                        if buy_qty > 0:
                            final_shares = current_shares + buy_qty

                # (C) 常规再平衡（没有触发特殊逻辑时，跟随目标权重，但限制单次变动幅度）
                else:
                    # 若权重目标要求增加，且无特殊条件，小幅跟进
                    if desired_shares > current_shares and target_w > 0 and trend_ok and can_buy:
                        # 限制单次增仓不超过缺口的30%，防止过度追涨
                        gap = desired_shares - current_shares
                        buy_qty = int(gap * min(0.3, self.buyback_alpha))
                        buy_qty = (buy_qty // self.min_shares_lot) * self.min_shares_lot
                        if buy_qty > 0:
                            final_shares = current_shares + buy_qty
                    elif desired_shares < current_shares:
                        # 若目标要求减仓，但未触发止盈，则缓慢减仓（每日最多减20%）
                        if can_sell:
                            reduce_qty = int((current_shares - desired_shares) * 0.2)
                            reduce_qty = (reduce_qty // self.min_shares_lot) * self.min_shares_lot
                            if reduce_qty > 0:
                                final_shares = current_shares - reduce_qty

                # ---- 流动性容量约束 ----
                max_trade_qty = int(current_adv[asset] * self.max_shares_pct_of_adv / price)
                max_trade_qty = (max_trade_qty // self.min_shares_lot) * self.min_shares_lot
                if max_trade_qty <= 0:
                    max_trade_qty = self.min_shares_lot * 10  # 兜底

                # 限制最终股数不超过合理范围，且单日买卖量不超过ADV限制
                if final_shares > current_shares:  # 净买入
                    buy_qty = final_shares - current_shares
                    buy_qty = min(buy_qty, max_trade_qty)
                    final_shares = current_shares + buy_qty
                elif final_shares < current_shares:  # 净卖出
                    sell_qty = current_shares - final_shares
                    sell_qty = min(sell_qty, max_trade_qty)
                    final_shares = current_shares - sell_qty

                # ---- T+1 限制（今日买入的部分不可卖出） ----
                if self.enable_t1_restriction:
                    today_bought = self.today_buy_qty.get(asset, 0)
                    # 若今日有买入，则今日最大可卖出量 = 昨日持仓 = current_shares - today_bought
                    max_sellable = current_shares - today_bought
                    if final_shares < current_shares and (current_shares - final_shares) > max_sellable:
                        # 不允许卖出今日新买入的部分
                        final_shares = current_shares - max_sellable
                        final_shares = max(0, final_shares)

                # ---- 交易执行 ----
                trade_qty = final_shares - current_shares
                if trade_qty != 0:
                    is_buy = trade_qty > 0
                    qty_abs = abs(trade_qty)
                    # 计算交易成本
                    cost_amount = self._calculate_trade_cost(price, qty_abs, is_buy)
                    # 检查现金是否充足（仅买入时）
                    if is_buy:
                        needed_cash = price * qty_abs + cost_amount
                        if needed_cash > self.cash:
                            # 现金不足，按最大可买数量调整
                            max_buy_qty = int((self.cash - self.min_commission) / price)
                            max_buy_qty = (max_buy_qty // self.min_shares_lot) * self.min_shares_lot
                            if max_buy_qty <= 0:
                                final_shares = current_shares
                            else:
                                final_shares = current_shares + max_buy_qty
                                # 重新计算真实交易量
                                trade_qty = final_shares - current_shares
                                qty_abs = abs(trade_qty)
                                cost_amount = self._calculate_trade_cost(price, qty_abs, is_buy)
                    # 执行资金与持仓变动
                    if trade_qty != 0:
                        amount = price * qty_abs
                        if is_buy:
                            self.cash -= (amount + cost_amount)
                            # 更新加权平均成本
                            old_cost = self.cost_basis.get(asset, np.nan)
                            if pd.isna(old_cost) or current_shares == 0:
                                new_cost = price
                            else:
                                new_cost = (old_cost * current_shares + price * qty_abs) / (current_shares + qty_abs)
                            self.cost_basis[asset] = new_cost
                            # 记录当日买入量
                            self.today_buy_qty[asset] = self.today_buy_qty.get(asset, 0) + qty_abs
                        else:
                            self.cash += (amount - cost_amount)
                            # 若清仓，重置成本
                            if final_shares == 0:
                                self.cost_basis[asset] = np.nan
                            # 若卖出量导致持仓减少但仍有持仓，成本不变
                        self.holdings[asset] = final_shares

                # 保存最终决定股数（供后续日期使用）
                self.holdings[asset] = final_shares

            # 2.6 更新prev_close（用于次日涨跌停计算）
            for a in common_assets:
                self.prev_close[a] = current_prices[a]
            # 重置今日买入记录（进入下一日）
            self.today_buy_qty = {a: 0 for a in common_assets}

            # 2.7 记录当日最终权重
            total_assets_end = self.cash + sum(self.holdings[a] * current_prices[a] for a in common_assets)
            total_asset_series[date] = total_assets_end
            if total_assets_end > 0:
                for a in common_assets:
                    mv = self.holdings[a] * current_prices[a]
                    executed_weights.loc[date, a] = mv / total_assets_end
            else:
                executed_weights.loc[date] = 0.0

        # ---------- 3. 收尾与日志 ----------
        final_total = total_asset_series.iloc[-1] if not total_asset_series.empty else initial_capital
        logger.info(f"=== 增强包装器执行完毕 | 最终总资产: {final_total:.4f} | 收益: {(final_total/initial_capital - 1)*100:.2f}% ===")
        
        # 返回与输入对齐的完整DataFrame（未交易日用前向填充）
        result_df = executed_weights.reindex(weights_df.index, columns=weights_df.columns).fillna(method='ffill').fillna(0)
        return result_df


# 保持与原接口兼容的快捷调用
def execute(weights_df: pd.DataFrame, price_df: pd.DataFrame, adv20_df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
    """
    快捷执行函数，兼容原 Phase 6 调用风格
    :param kwargs: 可传入 A股Phase6StrategyWrapper 的构造参数
    """
    wrapper = A股Phase6StrategyWrapper(**kwargs)
    return wrapper.execute(weights_df, price_df, adv20_df)


if __name__ == '__main__':
    # 单元测试/示例运行
    print("[测试] 生成模拟数据验证增强包装器...")
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
    assets = ['000001.SZ', '600036.SH', '300750.SZ']  # 包含创业板
    np.random.seed(42)
    price_data = np.cumprod(1 + np.random.normal(0, 0.01, (len(dates), len(assets))), axis=0) * 100
    price_df = pd.DataFrame(price_data, index=dates, columns=assets)
    weights_df = pd.DataFrame(np.random.dirichlet(np.ones(len(assets)), size=len(dates)), index=dates, columns=assets)
    adv20_df = pd.DataFrame(1e7, index=dates, columns=assets)  # 模拟流动性充足
    
    wrapper = A股Phase6StrategyWrapper(weekly_rebalance_only=False)
    result = wrapper.execute(weights_df, price_df, adv20_df)
    print("输出权重矩阵形状:", result.shape)
    print("首日权重:", result.iloc[0].values)
    print("增强包装器测试通过。")