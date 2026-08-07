# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Finite State Machine Backtest Core Engine
"""
import logging
import numpy as np
import pandas as pd
from Phase_7.config import (
    DEFAULT_HANDLING_FEE, DEFAULT_MANAGEMENT_FEE, DEFAULT_STAMP_TAX,
    DEFAULT_SLIPPAGE_BPS, GAP_UP_THRESHOLD, INITIAL_CASH,
    STAR_MARKET_LOT, MAIN_BOARD_LOT, MAX_SINGLE_TICKET_PROP
)
from Phase_7.market_utils import get_prices_for_date, get_previous_close_price
from Phase_7.risk_guard import compute_individual_position_limit
from Phase_7.execution_fsm import process_state_4_execution, process_state_5_equity, process_state_6_reconciliation
from Main.trading_costs import explicit_order_fees, is_etf, merged_cost_config

logger = logging.getLogger("FSMBacktest.Engine")

class FSMEngine:
    def __init__(self, context: dict):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        self.config = context.get('config', {}).copy()
        self._load_config_layer()

        # 账户状态
        self.cash = float(self.config.get('initial_cash', INITIAL_CASH))
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.price_cache = {}
        self.nav_series = []
        self.daily_returns_list = []
        self.violations_list = []
        self.cost_ledger = {"commission": 0.0, "exchange_fee": 0.0, "regulatory_fee": 0.0, "stamp_tax": 0.0, "slippage": 0.0, "management_fee": 0.0}

        # 停牌相关
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}
        self.impairment_factor = {sym: 1.0 for sym in self.assets}
        self.impairment_applied = {sym: False for sym in self.assets}

        # ==============================================================================
        # 🛡️ 注入时序轴时区自适应动态锚定引擎，彻底消除 tz-naive 与 tz-aware 冲突
        # ==============================================================================
        # 主动提取 DataBus 内部期望的物理时区令牌，默认兜底为 UTC
        target_tz = getattr(self.bus, '_tz', 'UTC')

        self.daily_weights = context['daily_weights'].copy()
        self.daily_weights.index = pd.to_datetime(self.daily_weights.index)
        if self.daily_weights.index.tz is None:
            self.daily_weights.index = self.daily_weights.index.tz_localize(target_tz)
        else:
            self.daily_weights.index = self.daily_weights.index.tz_convert(target_tz)
        
        self.daily_intervals = context.get('daily_intervals')
        if self.daily_intervals is not None:
            self.daily_intervals = self.daily_intervals.copy()
            self.daily_intervals.index = pd.to_datetime(self.daily_intervals.index)
            if self.daily_intervals.index.tz is None:
                self.daily_intervals.index = self.daily_intervals.index.tz_localize(target_tz)
            else:
                self.daily_intervals.index = self.daily_intervals.index.tz_convert(target_tz)
        # ==============================================================================
        
        # ==============================================================================
        # 🛡️ 自适应截面维度对齐引擎，彻底封杀 MultiIndex 错位与上游命名漂移
        # ==============================================================================
        self.q_low = None
        self.q_high = None
        
        if self.daily_intervals is not None:
            cols = self.daily_intervals.columns
            is_multi = isinstance(cols, pd.MultiIndex)

            # 声明模糊容灾令牌候选池（完美抵抗 CQR 标定层的契约脱靶）
            LOW_CANDIDATES = ['q_low', 'q_lower', 'lower', 'low', 'q_0.05', 'q_0.025']
            HIGH_CANDIDATES = ['q_high', 'q_upper', 'upper', 'high', 'q_0.95', 'q_0.975']

            target_low_attr = None
            target_high_attr = None

            if is_multi:
                # 【多重索引自愈】遍历所有列层级，精准定位分位数 Metric 所在的物理轴
                for level in range(cols.nlevels):
                    level_tags = cols.get_level_values(level).unique()
                    matched_low = [c for c in LOW_CANDIDATES if c in level_tags]
                    if matched_low:
                        target_low_attr = matched_low[0]
                        matched_high = [c for c in HIGH_CANDIDATES if c in level_tags]
                        target_high_attr = matched_high[0] if matched_high else None
                        
                        # 使用横截面交叉切片算子 .xs() 完美提取并重组为 (Date, Asset) 维度的纯净矩阵
                        self.q_low = self.daily_intervals.xs(target_low_attr, level=level, axis=1)
                        if target_high_attr:
                            self.q_high = self.daily_intervals.xs(target_high_attr, level=level, axis=1)
                        break
            else:
                # 【单层索引兜底】常规列名模糊寻优
                for c in LOW_CANDIDATES:
                    if c in cols:
                        target_low_attr = c
                        break
                for c in HIGH_CANDIDATES:
                    if c in cols:
                        target_high_attr = c
                        break
                
                if target_low_attr:
                    self.q_low = self.daily_intervals[target_low_attr]
                if target_high_attr:
                    self.q_high = self.daily_intervals[target_high_attr]

            # 打印对齐遥测监控日志
            if target_low_attr:
                logger.info("[ALIGN] Successfully aligned conformal interval low axis via: '%s'", target_low_attr)
            else:
                available_info = list(cols)[:5] if not is_multi else [list(cols.get_level_values(i)[:3]) for i in range(cols.nlevels)]
                logger.warning("[ALIGN] Conformal Interval Missing or Unrecognized. Structure: %s. Disabling Violation Check.", available_info)
        # ==============================================================================

        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}

    def _load_config_layer(self):
        """加载默认配置，补全缺失项"""
        self.config.setdefault('handling_fee', DEFAULT_HANDLING_FEE)
        self.config.setdefault('management_fee', DEFAULT_MANAGEMENT_FEE)
        self.config.setdefault('stamp_tax', DEFAULT_STAMP_TAX)
        self.config.setdefault('slippage_bps', DEFAULT_SLIPPAGE_BPS)
        self.config.setdefault('commission_rate', 0.00025)
        self.config.setdefault('minimum_commission', 5.0)
        self.config.setdefault('exchange_fee_rate', 0.0000341)
        self.config.setdefault('regulatory_fee_rate', 0.00002)
        self.config.setdefault('slippage_rate', DEFAULT_SLIPPAGE_BPS)
        self.config.setdefault('etf_annual_management_fee', 0.005)
        self.config.setdefault('gap_up_threshold', GAP_UP_THRESHOLD)
        self.config.setdefault('max_single_ticket_prop', MAX_SINGLE_TICKET_PROP)
        self.config.setdefault('star_market_lot', STAR_MARKET_LOT)
        self.config.setdefault('main_board_lot', MAIN_BOARD_LOT)
        self.config.setdefault('halt_days_limit', 20)
        self.config.setdefault('impairment_rate', 0.1)
        self.config.setdefault('default_residual_rate', 0.0)

    def _get_lot_size(self, sym):
        """根据股票代码返回整手股数"""
        if sym.startswith('688'):
            return self.config.get('star_market_lot', 200)
        else:
            return self.config.get('main_board_lot', 100)

    def calc_nav(self) -> float:
        """计算当前净值，考虑停牌减值"""
        mv = 0.0
        prices = get_prices_for_date(self.assets, self.current_date, self.bus, self.price_cache)
        for sym in self.assets:
            p = prices.get(sym) or 0.0
            factor = self.impairment_factor.get(sym, 1.0)
            mv += self.holdings[sym] * p * factor
        return float(self.cash + mv)

    def _check_is_delisted(self, sym) -> bool:
        """退市判断"""
        del_date = self.bus.query_by_pit(sym, self.current_date, "delisting_date")
        if del_date is not None and pd.Timestamp(del_date).replace(tzinfo=None) <= pd.Timestamp(self.current_date).replace(tzinfo=None):
            return True
        return False

    def _sell_asset_action(self, sym, shares, price):
        """执行卖出，包含冲击、税费"""
        adv = self.bus.query_by_pit(sym, self.current_date, "adv")
        adv = adv if (adv is not None and adv > 0) else 1e7
        turnover = (shares * price) / adv
        impact = 0.001 * (turnover ** 0.5)
        slippage = float(merged_cost_config(self.config)['slippage_rate'])
        exec_price = price * (1.0 - impact - slippage)
        cost = shares * exec_price
        fees = explicit_order_fees(cost, "sell", symbol=sym, config=self.config)
        self.cash += cost - fees['total']
        self.holdings[sym] -= shares
        for key in ("commission", "exchange_fee", "regulatory_fee", "stamp_tax"):
            self.cost_ledger[key] += fees[key]
        self.cost_ledger["slippage"] += shares * price * (impact + slippage)
        logger.debug("[SELL] %s %d shares @ %.4f (impact %.4f)", sym, shares, exec_price, impact)

    def run_engine_pipeline(self):
        test_dates = self.daily_weights.index
        if len(test_dates) == 0:
            raise ValueError("Empty test dates.")

        # logger.info("FSM Engine started. Initial cash: %.2f", self.cash)
        logger.info("FSM回测引擎启动 | 初始现金: %.2f | 回测周期: 起始 %s ~ 终止 %s", self.cash, test_dates[0], test_dates[-1])
        prev_nav = self.cash

        for t_idx, date in enumerate(test_dates):
            self.current_date = date
            date_str = date.strftime('%Y-%m-%d')

            # ---- 1. 获取价格 ----
            prices = get_prices_for_date(self.assets, date, self.bus, self.price_cache)
            daily_management = sum(
                self.holdings[sym] * (prices.get(sym) or 0.0) * self.config['etf_annual_management_fee'] / 252.0
                for sym in self.assets if is_etf(sym)
            )
            self.cash -= daily_management
            self.cost_ledger['management_fee'] += daily_management

            # ---- 2. 原始目标权重 ----
            raw_weights = self.daily_weights.loc[date].to_dict()
            adjusted_weights = {}

            # ---- 3. 风险过滤：追高防御 + 单票限额 ----
            for sym in self.assets:
                w = raw_weights.get(sym, 0.0)
                if w > 0:
                    # 追高防御
                    prev_close = get_previous_close_price(sym, date, self.bus, self.price_cache)
                    p_curr = prices.get(sym)
                    if p_curr is not None and prev_close is not None and prev_close > 0:
                        gap = (p_curr - prev_close) / prev_close
                        if gap >= self.config['gap_up_threshold']:
                            logger.debug("[GAP] %s gap up %.2f%% > threshold, weight set to 0", sym, gap*100)
                            w = 0.0
                    # 单票限额
                    if w > 0:
                        limit = compute_individual_position_limit(sym, prev_nav, date, self.bus, self.config)
                        if w > limit:
                            logger.debug("[LIMIT] %s weight %.4f > %.4f, clipped", sym, w, limit)
                            w = limit
                adjusted_weights[sym] = w

            # 重归一化（如果总权重 > 1）
            total_w = sum(adjusted_weights.values())
            if total_w > 1.0:
                adjusted_weights = {k: v / total_w for k, v in adjusted_weights.items()}

            # ---- 4. 执行状态机 ----
            logger.info("由FSM回测引擎执行Phase 4 ~ 6 | 日期: %s | 终点: %s", date_str, test_dates[-1].strftime('%Y-%m-%d'))
            logger.info("FSM回测启动phase4")
            process_state_4_execution(self, adjusted_weights, prices)
            logger.info("FSM回测启动phase5")
            process_state_5_equity(self)
            logger.info("FSM回测启动phase6")
            process_state_6_reconciliation(self, prices, prev_nav)

            # ---- 5. 收盘净值与收益 ----
            nav_after = self.calc_nav()
            self.nav_series.append(nav_after)
            ret = (nav_after - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            self.daily_returns_list.append((date_str, ret))

            # ---- 6. 置信区间违规检测（若成功提炼出 q_low/q_high） ----
            violation = 0
            if getattr(self, 'q_low', None) is not None and getattr(self, 'q_high', None) is not None:
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    w_vec = np.array([adjusted_weights.get(s, 0.0) for s in self.assets])
                    # 此时的 q_low 已经是解耦开的以 Asset 为列、Date 为行的纯净 DataFrame，.get 完美无缝运行
                    low_vec = np.array([self.q_low.loc[date].get(s, 0.0) for s in self.assets])
                    high_vec = np.array([self.q_high.loc[date].get(s, 0.0) for s in self.assets])
                    port_low = np.average(low_vec, weights=w_vec)
                    port_high = np.average(high_vec, weights=w_vec)
                    if not (port_low <= ret <= port_high):
                        violation = 1
                        logger.debug("[VIOL] %s ret %.4f outside [%.4f, %.4f]", date_str, ret, port_low, port_high)
            self.violations_list.append((date_str, violation))

            self.prev_weights = adjusted_weights.copy()
            prev_nav = nav_after

            # 每季度打印一次进度
            if t_idx % 60 == 0:
                # logger.info("Progress: %s, NAV=%.2f", date_str, nav_after)
                logger.info("回测期间季度打印: 总进程 %s / %s | 当前净值: %.2f", date_str, test_dates[-1].strftime('%Y-%m-%d'), nav_after)
        # logger.info("FSM Engine finished. Final NAV: %.2f", self.nav_series[-1] if self.nav_series else self.cash)
        logger.info("FSM回测完成 | 最终净值: %.2f | 回测周期: 起始%s ~ 终止%s", self.nav_series[-1] if self.nav_series else self.cash, test_dates[0], test_dates[-1])
        
        # 保存结果到 context
        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates)
        # Use the same Timestamp index as daily_nav so downstream audit phases
        # (e.g. Christoffersen coverage) can intersect the series correctly.
        self.context['daily_returns'] = pd.Series([r for _, r in self.daily_returns_list], index=test_dates)
        self.context['violations'] = pd.Series([v for _, v in self.violations_list], index=test_dates)
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.cash
        self.context['transaction_costs'] = {**self.cost_ledger, "total": float(sum(self.cost_ledger.values()))}
        self.context['backtest_ready'] = True


def execute(pipeline_context: dict) -> dict:
    """
    标准管道入口。原地更新 pipeline_context 并返回。
    """
    logger.info("=" * 60)
    # logger.info("[PHASE-7] Deploying Finite State Machine Backtester")
    # logger.info("[PHASE-7] Backtest Period: %s ~ %s", pipeline_context['daily_weights'].index[0], pipeline_context['daily_weights'].index[-1])
    logger.info("[Phase-7]运行 Finite State Machine 回测引擎启动")
    logger.info("=" * 60)

    engine = FSMEngine(pipeline_context)
    engine.run_engine_pipeline()

    # 更新上下文
    pipeline_context.update({
        'daily_nav': pipeline_context['daily_nav'],
        'daily_returns': pipeline_context['daily_returns'],
        'violations': pipeline_context['violations'],
        'final_nav': float(pipeline_context['final_nav']),
        'transaction_costs': pipeline_context['transaction_costs'],
        'fsm_engine': engine,          # 供后续阶段使用
        'nav_history': pipeline_context['daily_nav'].to_dict(),
    })

    # logger.info("[PHASE-7] Completed. Final NAV: %.2f", pipeline_context['final_nav'])
    logger.info("[Phase-7] 运行完成 | 最终净值: %.2f", pipeline_context['final_nav'])
    logger.info("=" * 60)
    return pipeline_context
