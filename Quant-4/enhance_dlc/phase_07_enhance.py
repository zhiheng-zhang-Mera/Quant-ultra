# -*- coding: utf-8 -*-
"""
Phase-07 增强模块：延迟执行、非线性滑点与做空管理器
用于替换/增强 execution_fsm 中的原始执行逻辑。
设计为纯函数式注入，不修改原执行函数签名，通过 context 传递增强配置。
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger("FSMBacktest.Enhancement")

# ------------------- 1. 延迟执行适配器 -------------------
class ExecutionDelayAdapter:
    """
    将当日信号缓存，并在次日以指定价格类型执行。
    支持 'open'（开盘价）或 'twap'（前30分钟均价，此处简化为 close 近似，实际需数据支持）。
    """
    def __init__(self, price_type: str = 'open'):
        self.price_type = price_type
        self.pending_signals = None  # 存储上一日的 adjusted_weights
        self.last_date = None

    def process(self, engine, adjusted_weights: Dict, current_date, prices: Dict):
        """
        返回应当在本日执行的权重（即前日信号），同时将本日信号缓存。
        """
        # 如果有前日信号，则执行它们；否则首次运行，不执行任何交易（或使用初始信号）
        if self.pending_signals is not None and self.last_date is not None:
            # 获取执行日的价格（若 price_type 为 open，需从 data_bus 获取当日开盘价）
            exec_prices = self._get_execution_prices(engine, current_date)
            # 执行前日信号（替代原有 process_state_4_execution 的 prices）
            self._execute_signals(engine, self.pending_signals, exec_prices)
            logger.debug(f"[Delay] Executed signals from {self.last_date} on {current_date}")

        # 缓存当前信号为下日执行
        self.pending_signals = adjusted_weights.copy()
        self.last_date = current_date
        # 返回 None 表示信号已在本次执行，无需再额外执行（原流程会跳过执行）
        return None

    def _get_execution_prices(self, engine, date):
        """根据 price_type 获取执行价，目前仅支持 'open'，若缺失则回退 total_return_price"""
        prices = {}
        for sym in engine.assets:
            if self.price_type == 'open':
                p = engine.bus.query_by_pit(sym, date, "open")
            else:  # twap 或 fallback
                p = engine.bus.query_by_pit(sym, date, "total_return_price")
            if p is None or np.isnan(p) or p <= 0:
                # 停牌穿透回退
                p = engine.bus.query_by_pit(sym, date, "total_return_price")
            prices[sym] = p
        return prices

    def _execute_signals(self, engine, weights, prices):
        """调用原始执行函数（process_state_4_execution），但传入缓存的 weights 和次日 prices"""
        # 注意：原 process_state_4_execution 会自行计算目标市值，我们只需传入目标权重
        # 但原始函数内部使用 engine.assets 和目标权重，这里的 weights 是前日信号
        from Phase_7.execution_fsm import process_state_4_execution
        process_state_4_execution(engine, weights, prices)


# ------------------- 2. 高级市场冲击模型 -------------------
def calculate_market_impact(symbol: str, order_value: float, daily_volume: float,
                            daily_volatility: float, alpha: float = 0.5,
                            beta: float = 0.02) -> float:
    """
    Barra 平方根冲击变体：
        Impact = alpha * sigma * (order_value / daily_volume)^beta
    其中 sigma 为日波动率，daily_volume 为日成交金额。
    返回冲击率（例如 0.001 表示 10bp）。
    """
    if daily_volume <= 0 or order_value <= 0:
        return 0.0
    turnover_fraction = order_value / daily_volume
    # 限制极端值，防止冲击过度
    turnover_fraction = min(turnover_fraction, 1.0)
    impact = alpha * daily_volatility * (turnover_fraction ** beta)
    # 合理范围 [0, 0.2]
    return np.clip(impact, 0.0, 0.2)

def apply_impact_to_price(price: float, is_buy: bool, impact: float) -> float:
    """买入加滑点，卖出减滑点"""
    if is_buy:
        return price * (1.0 + impact)
    else:
        return price * (1.0 - impact)

# 增强版执行函数（可替换 process_state_4_execution 中的买卖逻辑）
def execute_order_with_advanced_impact(engine, sym, shares, price, is_buy):
    """
    使用高级冲击模型进行单笔买卖，并更新 cash/holdings。
    返回执行价格（含冲击）和实际成交股数。
    """
    if shares <= 0 or price is None or price <= 0:
        return 0, 0

    # 获取该标的日成交金额（ADV）和日波动率
    adv = engine.bus.query_by_pit(sym, engine.current_date, "adv")
    if adv is None or adv <= 0:
        adv = 1e7  # 默认安全值
    # 日波动率（可用最近20日历史波动率，此处简化：从 data_bus 获取 annualized_volatility 折算）
    vol = engine.bus.query_by_pit(sym, engine.current_date, "volatility")  # 假设存在
    if vol is None or np.isnan(vol) or vol <= 0:
        vol = 0.02  # 默认 2% 日波动
    # 订单价值
    order_val = shares * price
    impact = calculate_market_impact(sym, order_val, adv, vol,
                                     alpha=engine.config.get('impact_alpha', 0.5),
                                     beta=engine.config.get('impact_beta', 0.02))
    exec_price = apply_impact_to_price(price, is_buy, impact)

    # 税费
    fee_rate = engine.config.get('handling_fee', 0.0000487) + engine.config.get('management_fee', 0.00002)
    stamp = engine.config.get('stamp_tax', 0.0005) if not is_buy else 0.0  # 卖出才收印花税

    cost = shares * exec_price
    fee = fee_rate * cost
    stamp_cost = stamp * cost if not is_buy else 0.0
    total_cost = cost + fee + stamp_cost

    if is_buy:
        if engine.cash < total_cost:
            # 资金不足，可部分成交（简化，此处直接拒绝）
            logger.warning(f"[Impact] Insufficient cash for {sym}, order skipped.")
            return 0, 0
        engine.cash -= total_cost
        engine.holdings[sym] += shares
    else:
        # 卖出
        if engine.holdings[sym] < shares:
            # 仓位不足，全清
            shares = engine.holdings[sym]
            if shares <= 0:
                return 0, 0
            cost = shares * exec_price
            fee = fee_rate * cost
            stamp_cost = stamp * cost
            total_cost = cost + fee + stamp_cost
        engine.cash += (cost - fee - stamp_cost)
        engine.holdings[sym] -= shares

    logger.debug(f"[AdvancedImpact] {sym} {'BUY' if is_buy else 'SELL'} {shares} @ {exec_price:.4f} impact={impact:.4f}")
    return shares, exec_price


# ------------------- 3. 借券管理器（做空支持） -------------------
class BorrowManager:
    """
    管理做空头寸的借券成本、额度与强平。
    """
    def __init__(self, config: Dict):
        self.annual_interest_rate = config.get('borrow_interest_rate', 0.08)  # 年化 8%
        self.max_short_ratio = config.get('max_short_ratio', 0.20)           # 单票最大做空市值/股票市值
        self.maintenance_margin = config.get('maintenance_margin', 1.30)     # 维持担保比例 130%
        self.initial_margin = config.get('initial_margin', 1.50)             # 初始保证金 150%
        # 记录每只股票的做空数量（负持仓）
        self.short_holdings = {}  # sym -> shares (正数表示做空股数)
        self.collateral = 0.0      # 抵押物（现金）
        self.borrow_fee_accrued = 0.0  # 已累计利息

    def can_short(self, engine, sym, shares, price) -> Tuple[bool, str]:
        """
        检查是否允许做空：额度、是否可融、维持保证金。
        """
        # 1. 是否可融（模拟：使用外部数据或随机）
        if not self._is_borrowable(sym, engine.current_date):
            return False, "Symbol not borrowable"
        # 2. 额度限制：做空市值不超过该股总市值的 max_short_ratio
        # 总市值需从 data_bus 获取（此处假设有 "market_cap" 字段）
        market_cap = engine.bus.query_by_pit(sym, engine.current_date, "market_cap")
        if market_cap is None or market_cap <= 0:
            market_cap = 1e9  # 默认
        short_value = shares * price
        if short_value > market_cap * self.max_short_ratio:
            return False, f"Short value {short_value:.2f} exceeds {self.max_short_ratio*100:.1f}% of market cap"
        # 3. 维持保证金（现有资产+抵押物）/（空头市值） >= maintenance_margin
        total_equity = engine.calc_nav() + self.collateral  # 假设全部资产可抵押
        if total_equity / (short_value + 1e-6) < self.maintenance_margin:
            return False, "Insufficient collateral for maintenance margin"
        return True, "OK"

    def _is_borrowable(self, sym, date):
        """模拟借券池：90% 的股票可融，或使用外部数据"""
        # 实际可通过 data_bus 查询 "borrowable" 标记
        return True  # 简化

    def open_short(self, engine, sym, shares, price):
        """开仓做空，记录 short_holdings，并冻结保证金"""
        if shares <= 0:
            return
        self.short_holdings[sym] = self.short_holdings.get(sym, 0) + shares
        # 冻结保证金：初始保证金比例 * 市值
        margin_required = shares * price * self.initial_margin
        if engine.cash < margin_required:
            logger.warning(f"[Borrow] Insufficient cash for margin, short order rejected.")
            return
        engine.cash -= margin_required
        self.collateral += margin_required
        # 实际做空头寸在 holdings 中体现为负数（但现有 holdings 为正，我们选择用单独字典记录）
        # 为了兼容净值计算，需调整 engine.holdings 为负（或另计）
        # 这里我们修改 holdings 为负（表示做空）
        engine.holdings[sym] = engine.holdings.get(sym, 0) - shares
        logger.info(f"[Borrow] Short {sym} {shares} shares at {price}, margin locked {margin_required:.2f}")

    def close_short(self, engine, sym, shares, price):
        """平仓做空，释放保证金，结算盈亏"""
        if sym not in self.short_holdings or self.short_holdings[sym] <= 0:
            return
        close_shares = min(shares, self.short_holdings[sym])
        # 买入平仓，需支付买入金额
        buy_cost = close_shares * price
        # 释放保证金：原先冻结的 margin 按比例释放
        locked_margin = close_shares * price * self.initial_margin
        # 计算平仓盈亏（卖空开仓价与平仓价的差）
        # 由于我们未记录开仓价，简化：仅按市价平仓，盈亏体现在现金中
        # 更真实：应记录每笔开仓成本，此处略
        # 更新 holdings
        engine.holdings[sym] += close_shares  # 负值减少
        # 现金变化：支付买入成本，释放保证金
        engine.cash -= buy_cost
        engine.cash += locked_margin  # 返还保证金
        self.collateral -= locked_margin
        self.short_holdings[sym] -= close_shares
        logger.info(f"[Borrow] Close short {sym} {close_shares} shares at {price}")

    def accrue_interest(self, engine, date):
        """每日计提融券利息（年化/365 * 做空市值）"""
        total_short_value = 0.0
        prices = {}  # 需要当前价格
        for sym, shares in self.short_holdings.items():
            if shares > 0:
                p = engine.bus.query_by_pit(sym, date, "total_return_price")
                if p is not None and p > 0:
                    total_short_value += shares * p
        daily_interest = total_short_value * self.annual_interest_rate / 365.0
        # 从现金中扣除利息
        if engine.cash >= daily_interest:
            engine.cash -= daily_interest
            self.borrow_fee_accrued += daily_interest
        else:
            # 现金不足，形成负债或强平
            logger.warning(f"[Borrow] Insufficient cash for interest {daily_interest:.2f}, forced liquidation may occur")
            # 此处可触发强平
        return daily_interest

    def force_liquidation(self, engine, date):
        """维持担保比例低于阈值时强平部分做空头寸"""
        # 计算担保比例
        total_equity = engine.calc_nav() + self.collateral
        total_short_value = 0.0
        for sym, shares in self.short_holdings.items():
            if shares > 0:
                p = engine.bus.query_by_pit(sym, date, "total_return_price")
                if p is not None and p > 0:
                    total_short_value += shares * p
        if total_short_value <= 0:
            return
        ratio = total_equity / total_short_value
        if ratio < self.maintenance_margin:
            # 强平部分头寸直到 ratio >= maintenance_margin
            target_short_value = total_equity / self.maintenance_margin
            reduce_value = total_short_value - target_short_value
            # 按比例平仓
            for sym, shares in self.short_holdings.items():
                if shares > 0 and reduce_value > 0:
                    p = engine.bus.query_by_pit(sym, date, "total_return_price")
                    if p is not None and p > 0:
                        val = shares * p
                        if val <= reduce_value:
                            self.close_short(engine, sym, shares, p)
                            reduce_value -= val
                        else:
                            close_shares = int(reduce_value / p)
                            if close_shares > 0:
                                self.close_short(engine, sym, close_shares, p)
                                reduce_value -= close_shares * p
            logger.info(f"[Borrow] Forced liquidation triggered, remaining short value {total_short_value - reduce_value:.2f}")


# ------------------- 4. 统一增强入口（供 FSMEngine 调用） -------------------
def apply_enhancements(engine, execution_context):
    """
    在 engine 初始化或每日循环前调用，注入增强功能。
    根据 execution_context 中的配置决定启用哪些增强。
    """
    enhancement_cfg = engine.config.get('enhancement', {})
    # 1. 延迟执行
    if enhancement_cfg.get('enable_delay', False):
        price_type = enhancement_cfg.get('delay_price_type', 'open')
        engine._delay_adapter = ExecutionDelayAdapter(price_type)
        # 修改 process_state_4_execution 的调用方式，在 run_engine_pipeline 中拦截
        # 我们通过 monkey-patch 或提供新的执行方法，此处演示在引擎中新增属性
        engine._delay_enabled = True

    # 2. 高级冲击模型
    if enhancement_cfg.get('enable_advanced_impact', False):
        engine._advanced_impact_enabled = True

    # 3. 做空管理
    if enhancement_cfg.get('enable_borrow', False):
        engine._borrow_manager = BorrowManager(enhancement_cfg)
        engine._borrow_enabled = True

    logger.info("[Enhance] Applied enhancements: delay=%s, impact=%s, borrow=%s",
                enhancement_cfg.get('enable_delay', False),
                enhancement_cfg.get('enable_advanced_impact', False),
                enhancement_cfg.get('enable_borrow', False))
    return engine