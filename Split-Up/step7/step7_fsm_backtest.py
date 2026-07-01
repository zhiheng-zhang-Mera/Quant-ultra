# -*- coding: utf-8 -*-
"""
step7/step7_fsm_backtest.py
按日穿透年化对齐有限状态机交易内核、多空组合区间破位监控回测中枢
"""
import logging
import numpy as np
import pandas as pd

from .config import *
from .market_utils import get_prices_for_date, estimate_adaptive_kappa
from .risk_guard import compute_dynamic_upper_bound
from .execution_fsm import (
    process_state_4_execution,
    process_state_5_equity,
    process_state_6_credit,
    process_state_7_reconciliation
)

logger = logging.getLogger("FSMBacktest.Engine")

class FSMEngine:
    def __init__(self, context):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        
        # 参数深度合并，外部未宣告时完美继承config配置箱内的默认因子
        self.config = context.get('config', {})
        self._load_config_layer()
        
        self.cash = float(self.config.get('initial_cash', 1_000_000.0))
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.nav_series = []
        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}
        
        # 数据向下完全兼容，从全局管道上下文中读取预计算好的持仓序列矩阵
        self.daily_weights = context.get('daily_weights')
        intervals = context.get('daily_intervals')
        
        if self.daily_weights is None or intervals is None:
            raise ValueError("❌ 关键前置特征缺失：缺少 daily_weights 或 daily_intervals。请确保阶段 6 凸优化分配已成功执行。")
            
        self.q_low_df = intervals['q_low']
        self.q_high_df = intervals['q_high']
        
        # 强类型日期轴转换索引清洗防线
        self.daily_weights.index = pd.DatetimeIndex(self.daily_weights.index)
        self.q_low_df.index = pd.DatetimeIndex(self.q_low_df.index)
        self.q_high_df.index = pd.DatetimeIndex(self.q_high_df.index)
        
        # 有限状态机底层运行期特殊状态簿
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}
        self.impairment_factor = {sym: 1.0 for sym in self.assets}
        self.impairment_applied = {sym: False for sym in self.assets}
        self.price_cache = {}
        
        # 历史运行审计归档簿
        self.daily_returns_list = []
        self.violations_list = []

    def _load_config_layer(self):
        """将分散的全局常数无缝融入局部交易引擎配置字典"""
        self.config.setdefault('handling_fee', DEFAULT_HANDLING_FEE)
        self.config.setdefault('management_fee', DEFAULT_MANAGEMENT_FEE)
        self.config.setdefault('stamp_tax', DEFAULT_STAMP_TAX)
        self.config.setdefault('auction_slippage_bps', DEFAULT_SLIPPAGE_BPS)
        self.config.setdefault('halt_days', DEFAULT_HALT_DAYS_LIMIT)
        self.config.setdefault('impairment_rate', DEFAULT_IMPAIRMENT_RATE)
        self.config.setdefault('default_residual_rate', DEFAULT_RESIDUAL_RATE)
        self.config.setdefault('margin_interest', DEFAULT_MARGIN_INTEREST)
        self.config.setdefault('short_interest', DEFAULT_SHORT_INTEREST)
        self.config.setdefault('maintenance_ratio', MAINTENANCE_RATIO_LIMIT)
        
        self.impact_kappa_base = self.config.get('impact_kappa_base', 0.05)

    def calc_nav(self):
        """
        全天候资产负债清算计算器：
        $NAV = Cash + \sum (Shares \times Price \times ImpairmentFactor)$
        """
        total_mv = 0.0
        prices = get_prices_for_date(self.assets, self.current_date, self.bus, self.price_cache)
        for sym in self.assets:
            price = prices.get(sym)
            if price is None or np.isnan(price):
                continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            total_mv += self.holdings[sym] * adj_price
        return float(self.cash + total_mv)

    def _sell_asset_action(self, sym, shares, price):
        """物理卖出订单落盘指令执行"""
        kappa = estimate_adaptive_kappa(sym, self.current_date, self.bus, self.config, self.impact_kappa_base)
        adv = self.bus.query_by_pit(sym, self.current_date, "adv")
        adv = adv if (adv is not None and adv > 0) else 1e7
        
        impact_factor = kappa * ((shares * price / adv) ** self.config.get('impact_alpha', 0.5))
        exec_price = price * (1.0 - impact_factor) - self.config.get('auction_slippage_bps', 0.0002) * price
        
        stamp = self.config.get('stamp_tax', 0.0005) * shares * exec_price
        fees = (self.config.get('handling_fee', 0.0000487) + self.config.get('management_fee', 0.00002)) * shares * exec_price
        
        self.cash += (shares * exec_price - stamp - fees)
        self.holdings[sym] -= shares

    def _check_is_delisted(self, sym):
        """判断该资产生命周期在当前历史截面中是否触发退市剔除"""
        list_date = self.bus.query_by_pit(sym, self.current_date, "listing_date")
        if list_date is None:
            return False
        years = (self.current_date - list_date).days / 365.25
        return years > 20  # 模拟边界案例

    def run_engine_pipeline(self):
        """时间步进器大循环，驱动交易引擎"""
        test_dates = self.daily_weights.index
        if len(test_dates) == 0:
            raise ValueError("❌ 无法拉起有限状态机回测流：凸优化后的 Test 测试日期序列为空!")
            
        logger.info(f"⏳ 交易状态机引擎已就位。开始前向穿透回测，时间轴跨度: {test_dates[0].strftime('%Y-%m-%d')} -> {test_dates[-1].strftime('%Y-%m-%d')} 共 {len(test_dates)} 个交易日。")
        
        for t, date in enumerate(test_dates):
            self.current_date = date
            
            # 1. 拦截目标权重并提取当前截面原始配属
            target_w_series = self.daily_weights.loc[date]
            raw_target_weights = {sym: target_w_series[sym] if sym in target_w_series else 0.0 for sym in self.assets}
            
            # 2. 动态风控风控层：在下单前针对最新NAV核算持仓比率，对超标资产实行刚性降维截断
            current_nav = self.calc_nav()
            adjusted_target_weights = {}
            for sym in self.assets:
                w = raw_target_weights.get(sym, 0.0)
                if w != 0.0:
                    upper_limit = compute_dynamic_upper_bound(sym, current_nav, date, self.bus)
                    if w > upper_limit:
                        logger.info(f"[{date.strftime('%Y-%m-%d')}] 🛡️ 触发防投毒举牌风控! 截断多头持仓偏置 {sym}: {w:.4f} -> {upper_limit:.4f} (当前资产包NAV={current_nav:.0f})")
                        w = upper_limit
                    elif w < -upper_limit:
                        w = -upper_limit
                adjusted_target_weights[sym] = w
                
            # 3. 提取当天的预测边界数据
            q_low_series = self.q_low_df.loc[date]
            q_high_series = self.q_high_df.loc[date]
            
            # 4. 获取当天最新的价格映射快照
            prices_snapshot = get_prices_for_date(self.assets, date, self.bus, self.price_cache)
            
            # 🚀 有限状态机顺序执行流推进
            process_state_4_execution(self, adjusted_target_weights, prices_snapshot)
            process_state_5_equity(self)
            process_state_6_credit(self, adjusted_target_weights, prices_snapshot)
            process_state_7_reconciliation(self, prices_snapshot)
            
            # 5. 核算T日终盘收盘后的清算净值
            nav_after_close = self.calc_nav()
            self.nav_series.append(nav_after_close)
            
            daily_ret = 0.0 if t == 0 else (nav_after_close - self.nav_series[-2]) / self.nav_series[-2]
            self.daily_returns_list.append((date, daily_ret))
            
            # 6. 统计学破位违规情况核算 (Conformal Interval Verification)
            w_vec = np.array([adjusted_target_weights.get(sym, 0.0) for sym in self.assets])
            q_low_vec = np.array([q_low_series.get(sym, 0.0) for sym in self.assets])
            q_high_vec = np.array([q_high_series.get(sym, 0.0) for sym in self.assets])
            
            active_mask = np.abs(w_vec) > 1e-6
            if active_mask.sum() == 0:
                violation = False
            else:
                # 组合级联合预测目标边界推演
                portfolio_low = np.sum(w_vec[active_mask] * q_low_vec[active_mask]) / np.sum(np.abs(w_vec[active_mask]))
                portfolio_high = np.sum(w_vec[active_mask] * q_high_vec[active_mask]) / np.sum(np.abs(w_vec[active_mask]))
                violation = not (portfolio_low <= daily_ret <= portfolio_high)
                
            self.violations_list.append((date, violation))
            self.prev_weights = adjusted_target_weights.copy()
            
        # 大循环结束，全量资产包数据格式回倒灌，向下游报告
        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates)
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.calc_nav()
        self.context['daily_returns'] = pd.Series(dict(self.daily_returns_list))
        self.context['violations'] = pd.Series(dict(self.violations_list))