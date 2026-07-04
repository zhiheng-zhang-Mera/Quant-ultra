"""
Quant-Ultra Flow - Phase_7 Finite State Machine Backtest Core Engine
"""
import logging
import numpy as np
import pandas as pd
from Phase_7.config import DEFAULT_HANDLING_FEE, DEFAULT_MANAGEMENT_FEE, DEFAULT_STAMP_TAX, DEFAULT_SLIPPAGE_BPS
from Phase_7.market_utils import get_prices_for_date, get_previous_close_price
from Phase_7.risk_guard import compute_individual_position_limit
from Phase_7.execution_fsm import process_state_4_execution, process_state_5_equity, process_state_6_reconciliation

logger = logging.getLogger("FSMBacktest.Engine")

class FSMEngine:
    def __init__(self, context: dict):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        self.config = context.get('config', {}).copy()
        self._load_config_layer()
        self.cash = float(self.config.get('individual_account_equity', 10000000.0))
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.nav_series = []
        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}
        self.daily_weights = context['daily_weights']
        self.price_cache = {}
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}
        self.violations_list = []
        self.daily_returns_list = []

    def _load_config_layer(self):
        self.config.setdefault('handling_fee', DEFAULT_HANDLING_FEE)
        self.config.setdefault('management_fee', DEFAULT_MANAGEMENT_FEE)
        self.config.setdefault('stamp_tax', DEFAULT_STAMP_TAX)
        self.config.setdefault('slippage_bps', DEFAULT_SLIPPAGE_BPS)

    def calc_nav(self) -> float:
        mv = 0.0
        prices = get_prices_for_date(self.assets, self.current_date, self.bus, self.price_cache)
        for sym in self.assets:
            p = prices.get(sym) or 0.0
            if self.halt_status[sym]:
                mv += self.holdings[sym] * p * (1.0 - self.config.get('impairment_rate', 0.1))
            else:
                mv += self.holdings[sym] * p
        return float(self.cash + mv)

    def _check_is_delisted(self, asset: str) -> bool:
        if hasattr(self.bus, 'manager') and asset in getattr(self.bus.manager, '_failed_symbols', set()):
            return True
        return False

    def _sell_asset_action(self, sym: str, shares: float, price: float):
        cost = shares * price
        fee = (self.config.get('handling_fee', 0.0000487) + self.config.get('management_fee', 0.00002)) * cost
        tax = self.config.get('stamp_tax', 0.0005) * cost
        self.cash += (cost - fee - tax)
        self.holdings[sym] -= shares

    def run_engine_pipeline(self):
        test_dates_str = self.daily_weights.index.tolist()
        test_dates = [pd.Timestamp(d).to_pydatetime() for d in test_dates_str]
        prev_nav = self.cash

        for t_idx, date in enumerate(test_dates):
            self.current_date = date
            date_str = test_dates_str[t_idx]
            
            # State 1: 行情接入与点时对齐价格捕获
            prices = get_prices_for_date(self.assets, date, self.bus, self.price_cache)
            
            # State 2: 目标权重输入与自愈合账
            t_weights_raw = self.daily_weights.loc[date_str].to_dict()
            adjusted_target_weights = {}
            violation_flag = 0

            # State 3: 个体流动性限额与追高防御双重过滤器校验
            for sym in self.assets:
                w_target = t_weights_raw.get(sym, 0.0)
                if w_target > 0:
                    p_curr = prices.get(sym)
                    p_prev_close = get_previous_close_price(sym, date, self.bus, self.price_cache)
                    
                    if p_curr and p_prev_close and (p_curr - p_prev_close) / p_prev_close >= GAP_UP_THRESHOLD:
                        w_target = 0.0; violation_flag = 1
                        
                    limit_line = compute_individual_position_limit(sym, prev_nav, date, self.bus, self.config)
                    if w_target > limit_line: w_target = limit_line; violation_flag = 1
                    
                adjusted_target_weights[sym] = w_target

            # 重新归一化分配多头空闲资产仓位
            tot_w = sum(adjusted_target_weights.values())
            if tot_w > 1.0:
                adjusted_target_weights = {k: v / tot_w for k, v in adjusted_target_weights.items()}

            # State 4-6: 撮合、权益、强行对账
            process_state_4_execution(self, adjusted_target_weights, prices)
            process_state_5_equity(self)
            process_state_6_reconciliation(self, prices, prev_nav)

            nav_after_close = self.calc_nav()
            self.nav_series.append(nav_after_close)
            
            daily_return = np.log(nav_after_close / prev_nav) if prev_nav > 0 else 0.0
            self.daily_returns_list.append((date_str, daily_return))
            self.violations_list.append((date_str, violation_flag))
            
            self.prev_weights = adjusted_target_weights.copy()
            prev_nav = nav_after_close

        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates_str)
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.cash
        self.context['daily_returns'] = pd.Series(dict(self.daily_returns_list))
        self.context['violations'] = pd.Series(dict(self.violations_list))

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[OP] Deploy Phase_7 Finite State Backtester | [SOURCE] Unified Pipeline Control Deck | [RESULT] FSM Engine initialized successfully | [SIGNIFICANCE] Maps continuous allocation weights to audited spot capital performance curves")
    logger.info("[操作] 部署 Phase_7 状态机回测核 | [来源] 统一工作流中央控制台 | [结果] FSM 引擎流水线成功激活 | [意义] 将离线凸优化生成的权重序列落地为经得起审计的真实资金现货净值变动曲线")
    logger.info("=" * 60)
    
    engine = FSMEngine(pipeline_context)
    engine.run_engine_pipeline()
    
    return {
        'daily_nav': pipeline_context['daily_nav'],
        'daily_returns': pipeline_context['daily_returns'],
        'violations': pipeline_context['violations'].astype(int),
        'final_nav': float(pipeline_context['final_nav']),
        'backtest_ready': True
    }
```

### 6. `Phase_7/__init__.py` (约 5 行)
```python
# -*- coding: utf-8 -*-
# 位于 Phase_7/__init__.py
from .step7_fsm_backtest import execute

__all__ = ['execute']