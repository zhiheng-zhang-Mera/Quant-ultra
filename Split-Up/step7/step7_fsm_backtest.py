# -*- coding: utf-8 -*-
"""
step7/step7_fsm_backtest.py
有限状态机交易回测中枢（全套落实静态容量终审与会话双格式持久化生命周期管理）
"""
import os
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
    def __init__(self, context):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        
        # 参数继承配置箱默认因子
        self.config = context.get('config', {})
        self._load_config_layer()
        
        self.cash = float(self.config.get('initial_cash', 1_000_000.0))
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.nav_series = []
        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}
        
        # 数据向下兼容，从全局管道上下文中读取预计算好的持仓序列矩阵
        self.daily_weights = context.get('daily_weights')
        intervals = context.get('daily_intervals')
        
        if self.daily_weights is None or intervals is None:
            raise ValueError("❌ 关键前置特征缺失：缺少 daily_weights 或 daily_intervals。请确保阶段 6 凸优化分配已成功执行。")
            
        self.q_low_df = intervals['q_low']
        self.q_high_df = intervals['q_high']
        
        # 强类型日期轴转换清洗
        self.daily_weights.index = pd.DatetimeIndex(self.daily_weights.index)
        self.q_low_df.index = pd.DatetimeIndex(self.q_low_df.index)
        self.q_high_df.index = pd.DatetimeIndex(self.q_high_df.index)
        
        # 有限状态机运行期内部寄存状态簿
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}
        self.impairment_factor = {sym: 1.0 for sym in self.assets}
        self.impairment_applied = {sym: False for sym in self.assets}
        self.price_cache = {}
        
        self.daily_returns_list = []
        self.violations_list = []

    def _load_config_layer(self):
        """将分散的全局常数及硬编码静态因子无缝融入配置字典"""
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
        
        # 彻底废除一切基于 AUM 放大估计的迭代基础
        self.impact_kappa_base = STATIC_KAPPA_IMPACT

    def calc_nav(self):
        """100%纯现货无负债资产负债清算计算器"""
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
        """物理卖出订单清退指令执行 (采用固定硬编码冲击常数)"""
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
        """根据数据总线点状检索生存者偏差判定"""
        delisted_date = self.bus.query_by_pit(sym, self.current_date, "delisting_date")
        if delisted_date is not None and self.current_date >= pd.Timestamp(delisted_date):
            return True
        return False

    def run_engine_pipeline(self):
        """时间步进器大循环，驱动交易引擎（Flow-Pro 8.5 纯静态容量边界终审路径）"""
        test_dates = self.daily_weights.index
        if len(test_dates) == 0:
            raise ValueError("❌ 凸优化后的 Test 测试日期序列为空!")
            
        logger.info(f"⏳ 现货FSM交易引擎启动。执行个体静态容量边界终审。当前本金规模: {self.cash:.0f} 元。")
        
        prev_nav = None
        for t, date in enumerate(test_dates):
            self.current_date = date
            
            # 1. 拦截目标权重
            target_w_series = self.daily_weights.loc[date]
            raw_target_weights = {sym: target_w_series[sym] if sym in target_w_series else 0.0 for sym in self.assets}
            
            # 2. 动态风控风控层：个人流动性与资金量双重硬约束防线（替代机构举牌线）
            current_nav = self.calc_nav() if t > 0 else self.cash
            adjusted_target_weights = {}
            for sym in self.assets:
                w = max(0.0, raw_target_weights.get(sym, 0.0))  # 刚性排除一切做空毒素
                if w > 0.0:
                    upper_limit = compute_individual_position_limit(sym, current_nav, date, self.bus, self.config)
                    if w > upper_limit:
                        logger.debug(f"[{date.strftime('%Y-%m-%d')}] 🛡️ 个人流动性过低，限制持仓比例 {sym}: {w:.4f} -> {upper_limit:.4f}")
                        w = upper_limit
                adjusted_target_weights[sym] = w
                
            q_low_series = self.q_low_df.loc[date]
            q_high_series = self.q_high_df.loc[date]
            prices_snapshot = get_prices_for_date(self.assets, date, self.bus, self.price_cache)
            
            # 🚀 有限状态机顺序执行流推进
            process_state_4_execution(self, adjusted_target_weights, prices_snapshot)
            process_state_5_equity(self)
            process_state_6_reconciliation(self, prices_snapshot, prev_nav)
            
            # 3. 核算收盘后净值并归档
            nav_after_close = self.calc_nav()
            self.nav_series.append(nav_after_close)
            
            daily_ret = 0.0 if t == 0 else (nav_after_close - self.nav_series[-2]) / self.nav_series[-2]
            self.daily_returns_list.append((date, daily_ret))
            
            # 4. 统计学符合性区间穿透核算 (Conformal Interval Verification)
            w_vec = np.array([adjusted_target_weights.get(sym, 0.0) for sym in self.assets])
            q_low_vec = np.array([q_low_series.get(sym, 0.0) for sym in self.assets])
            q_high_vec = np.array([q_high_series.get(sym, 0.0) for sym in self.assets])
            
            active_mask = w_vec > 1e-6
            if active_mask.sum() == 0:
                violation = False
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


def execute_step7_with_cache(context, force_update=False):
    """
    [README 挂起任务完美闭环核心入口]
    分阶段会话缓存与双格式持久化（.parquet / .feather）生命周期管理。
    """
    cache_dir_parquet = os.path.join("Split-Up", "Phase Result", "parquet", "Phase_7")
    cache_dir_feather = os.path.join("Split-Up", "Phase Result", "feather", "Phase_7")
    
    os.makedirs(cache_dir_parquet, exist_ok=True)
    os.makedirs(cache_dir_feather, exist_ok=True)
    
    file_parquet = os.path.join(cache_dir_parquet, "step7_output.parquet")
    file_feather = os.path.join(cache_dir_feather, "step7_output.feather")
    
    # 1. 检查缓存就绪状态
    if not force_update and os.path.exists(file_parquet):
        logger.info("🎯 [Cache Hit] 侦测到本地合规阶段七时序回测缓存，直接启动极速加载，免去重复回归。")
        df_cache = pd.read_parquet(file_parquet)
        context['daily_nav'] = df_cache['daily_nav']
        context['daily_returns'] = df_cache['daily_returns']
        context['violations'] = df_cache['violations'].astype(bool)
        context['final_nav'] = float(df_cache['daily_nav'].iloc[-1])
        return
        
    # 2. 缓存失效或强制刷新时，物理呼叫确定性 FSM 状态机进行前向计算
    engine = FSMEngine(context)
    engine.run_engine_pipeline()
    
    # 3. 整合面板结构矩阵
    df_save = pd.DataFrame({
        'daily_nav': context['daily_nav'],
        'daily_returns': context['daily_returns'],
        'violations': context['violations'].astype(int)
    })
    
    # 4. 双格式本地物理隔离转储固化
    df_save.to_parquet(file_parquet, index=True)
    df_save.reset_index().to_feather(file_feather)
    logger.info("💾 [Cache Saved] 阶段七确定性状态机运算完毕，双格式数据高保真固化落盘。")