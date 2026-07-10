# -*- coding: utf-8 -*-
"""
step7/step7_fsm_backtest.py
有限状态机交易回测中枢 [2026 生产级规范化净化版]
修复标准 execute 接口命名错位，彻底移除自建本地 I/O
"""
import logging
import numpy as np
import pandas as pd

from .config import *
from .market_utils import get_prices_for_date
from .risk_guard import compute_individual_position_limit
from .execution_fsm import (
    process_state_4_execution,
    process_state_5_equity,
    process_state_6_reconciliation
)

logger = logging.getLogger("FSMBacktest.Engine")

class FSMEngine:
    # ... [保持 FSMEngine 类内部数理逻辑及有限状态机核心不产生任何变动，此处略去以防混淆] ...
    def __init__(self, context):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        self.config = context.get('config', {}).copy()
        self._load_config_layer()
        self.cash = float(self.config.get('initial_cash', 1_000_000.0))
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.nav_series = []
        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}
        self.daily_weights = context.get('daily_weights')
        intervals = context.get('daily_intervals')
        
        if self.daily_weights is None or intervals is None:
            raise ValueError("关键前置特征缺失：缺少 daily_weights 或 daily_intervals。")
            
        self.q_low_df = intervals['q_low']
        self.q_high_df = intervals['q_high']
        self.daily_weights.index = pd.DatetimeIndex(self.daily_weights.index)
        self.q_low_df.index = pd.DatetimeIndex(self.q_low_df.index)
        self.q_high_df.index = pd.DatetimeIndex(self.q_high_df.index)
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}
        self.impairment_factor = {sym: 1.0 for sym in self.assets}
        self.impairment_applied = {sym: False for sym in self.assets}
        self.price_cache = {}
        self.daily_returns_list = []
        self.violations_list = []

    def _load_config_layer(self):
        self.config.setdefault('handling_fee', DEFAULT_HANDLING_FEE)
        self.config.setdefault('management_fee', DEFAULT_MANAGEMENT_FEE)
        self.config.setdefault('stamp_tax', DEFAULT_STAMP_TAX)
        self.config.setdefault('auction_slippage_bps', DEFAULT_SLIPPAGE_BPS)
        self.config.setdefault('halt_days', DEFAULT_HALT_DAYS_LIMIT)
        self.config.setdefault('impairment_rate', DEFAULT_IMPAIRMENT_RATE)
        self.config.setdefault('default_residual_rate', DEFAULT_RESIDUAL_RATE)
        self.config.setdefault('static_kappa_impact', STATIC_KAPPA_IMPACT)
        self.config.setdefault('static_alpha_impact', STATIC_ALPHA_IMPACT)
        self.config.setdefault('gap_up_threshold', GAP_UP_THRESHOLD)
        self.config.setdefault('max_single_ticket_prop', MAX_SINGLE_TICKET_PROP)
        self.config.setdefault('star_market_lot', STAR_MARKET_LOT)
        self.config.setdefault('main_board_lot', MAIN_BOARD_LOT)
        self.config.setdefault('clearing_threshold', 0.001)
        self.impact_kappa_base = STATIC_KAPPA_IMPACT

    def calc_nav(self):
        total_mv = 0.0
        prices = get_prices_for_date(self.assets, self.current_date, self.bus, self.price_cache)
        for sym in self.assets:
            price = prices.get(sym)
            if price is None or np.isnan(price): continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            total_mv += self.holdings[sym] * adj_price
        return float(self.cash + total_mv)

    def _sell_asset_action(self, sym, shares, price):
        adv = self.bus.query_by_pit(sym, self.current_date, "adv")
        adv = adv if (adv is not None and adv > 0) else 1e7
        kappa = self.config.get('static_kappa_impact', 0.001)
        alpha = self.config.get('static_alpha_impact', 0.5)
        impact_factor = kappa * ((shares * price / adv) ** alpha)
        exec_price = price * (1.0 - impact_factor) - self.config.get('auction_slippage_bps', 0.0002) * price
        stamp = self.config.get('stamp_tax', 0.0005) * shares * exec_price
        fees = (self.config.get('handling_fee', 0.0000487) + self.config.get('management_fee', 0.00002)) * shares * exec_price
        self.cash += (shares * exec_price - stamp - fees)
        self.holdings[sym] -= shares

    def _check_is_delisted(self, sym):
        delisted_date = self.bus.query_by_pit(sym, self.current_date, "delisting_date")
        if delisted_date is not None and self.current_date >= pd.Timestamp(delisted_date): return True
        return False

    def run_engine_pipeline(self):
        test_dates = self.daily_weights.index
        if len(test_dates) == 0: raise ValueError("❌ 凸优化后的 Test 测试日期序列为空!")
        logger.info(f"⏳ 现货FSM交易引擎启动。当前本金规模: {self.cash:.0f} 元。")
        prev_nav = None
        for t, date in enumerate(test_dates):
            self.current_date = date
            target_w_series = self.daily_weights.loc[date]
            raw_target_weights = {sym: target_w_series[sym] if sym in target_w_series else 0.0 for sym in self.assets}
            current_nav = self.calc_nav() if t > 0 else self.cash
            adjusted_target_weights = {}
            for sym in self.assets:
                w = max(0.0, raw_target_weights.get(sym, 0.0))
                if w > 0.0:
                    upper_limit = compute_individual_position_limit(sym, current_nav, date, self.bus, self.config)
                    if w > upper_limit: w = upper_limit
                adjusted_target_weights[sym] = w
            q_low_series = self.q_low_df.loc[date]
            q_high_series = self.q_high_df.loc[date]
            prices_snapshot = get_prices_for_date(self.assets, date, self.bus, self.price_cache)
            process_state_4_execution(self, adjusted_target_weights, prices_snapshot)
            process_state_5_equity(self)
            process_state_6_reconciliation(self, prices_snapshot, prev_nav)
            nav_after_close = self.calc_nav()
            self.nav_series.append(nav_after_close)
            daily_ret = 0.0 if t == 0 else (nav_after_close - self.nav_series[-2]) / self.nav_series[-2]
            self.daily_returns_list.append((date, daily_ret))
            w_vec = np.array([adjusted_target_weights.get(sym, 0.0) for sym in self.assets])
            q_low_vec = np.array([q_low_series.get(sym, 0.0) for sym in self.assets])
            q_high_vec = np.array([q_high_series.get(sym, 0.0) for sym in self.assets])
            active_mask = w_vec > 1e-6
            if active_mask.sum() == 0: violation = False
            else:
                portfolio_low = np.sum(w_vec[active_mask] * q_low_vec[active_mask]) / np.sum(w_vec[active_mask])
                portfolio_high = np.sum(w_vec[active_mask] * q_high_vec[active_mask]) / np.sum(w_vec[active_mask])
                violation = not (portfolio_low <= daily_ret <= portfolio_high)
            self.violations_list.append((date, violation))
            self.prev_weights = adjusted_target_weights.copy()
            prev_nav = nav_after_close
        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates)
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.cash
        self.context['daily_returns'] = pd.Series(dict(self.daily_returns_list))
        self.context['violations'] = pd.Series(dict(self.violations_list))

def execute(pipeline_context: dict) -> dict:
    """
    💡 完美的标准管道入口定义。
    抛弃手写 execute_step7_with_cache 与本地 Parquet 校验。
    """
    logger.info("=" * 60)
    logger.info("推进核心阶段 7: 有限状态机交易回测中枢 [纯净化解耦版]")
    logger.info("=" * 60)
    
    engine = FSMEngine(pipeline_context)
    engine.run_engine_pipeline()
    
    # 构建要交回给总线并存储的纯脏数据资产
    result_update = {
        'daily_nav': pipeline_context['daily_nav'],
        'daily_returns': pipeline_context['daily_returns'],
        'violations': pipeline_context['violations'].astype(int),
        'final_nav': float(pipeline_context['final_nav']),
        'fsm_backtest_ready': True
    }
    
    logger.info("✅ Step 7 状态机前向时序模拟全量收敛，回传指标由主控一键落盘。")
    return result_update