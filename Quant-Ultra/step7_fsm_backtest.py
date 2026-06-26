"""
Phase 7: Chronologically-Aligned State-Machine Engine and Asset Liability Backtesting
Now with daily weights and intervals from step6, records violations and daily returns.
完善停牌处理：使用最近有效价格并应用减值因子。
"""
import numpy as np
import pandas as pd
from datetime import timedelta
import logging

logger = logging.getLogger("FSMBacktest")

CONFIG = {
    "STAMP_TAX": 0.0005,
    "HANDLING_FEE": 0.0000487,
    "MANAGEMENT_FEE": 0.00002,
    "MARGIN_INTEREST": 0.06 / 252,
    "SHORT_INTEREST": 0.08 / 252,
    "MAINTENANCE_RATIO": 1.3,
    "HALT_DAYS": 20,
    "IMPAIRMENT_RATE": 0.1,          # 停牌超期减值比例
    "DELISTING_PENALTY": 0.05,
    "BANKRUPTCY_RESIDUAL": 0.1,
    "AUCTION_SLIPPAGE_BPS": 0.0002,
    "IMPACT_ALPHA": 0.5,
    "IMPACT_KAPPA": 0.05,
    "LIMIT_ORDER_FILL_RATIO": 0.5,
    "LIQUIDITY_DECAY": 0.001,
    "CLEARING_THRESHOLD": 0.001,
}

class FSMEngine:
    def __init__(self, context):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        self.cash = 1_000_000.0
        self.holdings = {sym: 0.0 for sym in self.assets}
        self.nav_series = []
        self.current_date = None
        self.prev_weights = {sym: 0.0 for sym in self.assets}
        # 从上下文读取预计算的权重和区间
        self.daily_weights = context.get('daily_weights')
        if self.daily_weights is None:
            raise ValueError("缺少 daily_weights，请确保 step6 已执行并预计算。")
        intervals = context.get('daily_intervals')
        if intervals is None:
            raise ValueError("缺少 daily_intervals，请确保 step6 已执行。")
        self.q_low_df = intervals['q_low']
        self.q_high_df = intervals['q_high']
        self.daily_weights.index = pd.DatetimeIndex(self.daily_weights.index)
        self.q_low_df.index = pd.DatetimeIndex(self.q_low_df.index)
        self.q_high_df.index = pd.DatetimeIndex(self.q_high_df.index)

        # 停牌相关状态
        self.halt_counter = {sym: 0 for sym in self.assets}
        self.halt_status = {sym: False for sym in self.assets}          # 当日是否停牌
        self.impairment_factor = {sym: 1.0 for sym in self.assets}      # 减值系数
        self.impairment_applied = {sym: False for sym in self.assets}   # 是否已减值（避免重复扣减）
        self.price_cache = {}
        # 用于记录每日收益率和 violations
        self.daily_returns_list = []
        self.violations_list = []

    def run(self):
        test_dates = self.daily_weights.index
        if len(test_dates) == 0:
            raise ValueError("Test 集为空。")
        for t, date in enumerate(test_dates):
            self.current_date = date
            # 获取当日目标权重
            target_w_series = self.daily_weights.loc[date]
            target_weights = {sym: target_w_series[sym] if sym in target_w_series else 0.0 for sym in self.assets}
            # 获取当日预测区间
            q_low_series = self.q_low_df.loc[date]
            q_high_series = self.q_high_df.loc[date]

            # State 4: 执行撮合（含停牌处理）
            self.state_4_execution(target_weights)
            # State 5: 权益处理
            self.state_5_equity_processing()
            # State 6: 信用计提
            self.state_6_credit_accrual(target_weights)
            # 记录当日净值（在清算前）
            nav_before = self.calc_nav()
            # State 7: 对账清算
            self.state_7_reconciliation()
            nav_after = self.calc_nav()
            self.nav_series.append(nav_after)
            # 计算当日实际收益率
            if t == 0:
                daily_ret = 0.0
            else:
                daily_ret = (nav_after - self.nav_series[-2]) / self.nav_series[-2]
            self.daily_returns_list.append((date, daily_ret))

            # 判断违约（组合收益率是否落在预测区间外）
            w_vec = np.array([target_weights.get(sym, 0.0) for sym in self.assets])
            q_low_vec = np.array([q_low_series.get(sym, 0.0) for sym in self.assets])
            q_high_vec = np.array([q_high_series.get(sym, 0.0) for sym in self.assets])
            mask = np.abs(w_vec) > 1e-6
            if mask.sum() == 0:
                violation = False
            else:
                portfolio_low = np.sum(w_vec[mask] * q_low_vec[mask]) / np.sum(np.abs(w_vec[mask]))
                portfolio_high = np.sum(w_vec[mask] * q_high_vec[mask]) / np.sum(np.abs(w_vec[mask]))
                violation = not (portfolio_low <= daily_ret <= portfolio_high)
            self.violations_list.append((date, violation))

            # 更新上一期权重
            self.prev_weights = target_weights.copy()

        # 回测结束后，将结果存入上下文
        self.context['nav_series'] = self.nav_series  # 保留（可选）
        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates)  # 新增
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.calc_nav()
        if self.daily_returns_list:
            dates, rets = zip(*self.daily_returns_list)
            daily_returns_series = pd.Series(rets, index=pd.DatetimeIndex(dates))
            self.context['daily_returns'] = daily_returns_series
        else:
            self.context['daily_returns'] = pd.Series()
        if self.violations_list:
            dates, viols = zip(*self.violations_list)
            violations_series = pd.Series(viols, index=pd.DatetimeIndex(dates))
            self.context['violations'] = violations_series
        else:
            self.context['violations'] = pd.Series()

    def state_4_execution(self, target_weights):
        """撮合执行，含整手截断、冲击滑点、停牌处理"""
        prices = self.get_prices(self.current_date)
        total_nav = self.calc_nav()
        target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in self.assets}
        for sym in self.assets:
            price = prices.get(sym)
            # 停牌处理
            if price is None or np.isnan(price) or price <= 0:
                self.halt_counter[sym] += 1
                self.halt_status[sym] = True
                # 检查是否达到停牌减值阈值且未减值
                if self.halt_counter[sym] >= CONFIG["HALT_DAYS"] and not self.impairment_applied[sym]:
                    self._apply_halt_impairment(sym)
                continue
            else:
                # 恢复交易，重置停牌计数和状态
                self.halt_counter[sym] = 0
                self.halt_status[sym] = False
                # 若之前有减值，但价格恢复，我们保持减值因子不变（规范未明确是否恢复），
                # 这里保持减值，除非另行处理（可设计为恢复后撤销减值，但保守起见保留）

            current_shares = self.holdings[sym]
            target_shares = target_values[sym] / price if price > 0 else 0
            diff = target_shares - current_shares
            lot_unit = 200 if sym.startswith("688") else 100
            if abs(diff) < CONFIG["CLEARING_THRESHOLD"] * total_nav / price:
                if current_shares != 0:
                    self._sell(sym, current_shares, price)
                continue
            if diff > 0:
                exec_shares = np.floor(diff / lot_unit) * lot_unit
                if exec_shares >= lot_unit:
                    adv = self.bus.query_by_pit(sym, self.current_date, "adv")
                    if adv is None or adv <= 0:
                        adv = 1e7
                    impact_factor = CONFIG["IMPACT_KAPPA"] * ((exec_shares * price / adv) ** CONFIG["IMPACT_ALPHA"])
                    exec_price = price * (1 + impact_factor) + CONFIG["AUCTION_SLIPPAGE_BPS"] * price
                    fee = (CONFIG["HANDLING_FEE"] + CONFIG["MANAGEMENT_FEE"]) * exec_shares * exec_price
                    self.cash -= exec_shares * exec_price + fee
                    self.holdings[sym] += exec_shares
            elif diff < 0:
                sell_shares = min(abs(diff), current_shares)
                if current_shares - sell_shares < lot_unit and current_shares - sell_shares > 0:
                    sell_shares = current_shares
                else:
                    sell_shares = np.floor(sell_shares / lot_unit) * lot_unit
                if sell_shares > 0:
                    self._sell(sym, sell_shares, price)

    def _sell(self, sym, shares, price):
        adv = self.bus.query_by_pit(sym, self.current_date, "adv")
        if adv is None or adv <= 0:
            adv = 1e7
        impact_factor = CONFIG["IMPACT_KAPPA"] * ((shares * price / adv) ** CONFIG["IMPACT_ALPHA"])
        exec_price = price * (1 - impact_factor) - CONFIG["AUCTION_SLIPPAGE_BPS"] * price
        stamp = CONFIG["STAMP_TAX"] * shares * exec_price
        fees = (CONFIG["HANDLING_FEE"] + CONFIG["MANAGEMENT_FEE"]) * shares * exec_price
        self.cash += shares * exec_price - stamp - fees
        self.holdings[sym] -= shares

    def _apply_halt_impairment(self, sym):
        """停牌超期减值，按固定比例降低减值因子"""
        if self.impairment_applied[sym]:
            return
        rate = CONFIG["IMPAIRMENT_RATE"]
        self.impairment_factor[sym] *= (1 - rate)
        self.impairment_applied[sym] = True
        logger.warning(f"{sym} 停牌超 {CONFIG['HALT_DAYS']} 日，减值 {rate*100:.1f}%，当前因子 {self.impairment_factor[sym]:.3f}")

    def state_5_equity_processing(self):
        pass

    def state_6_credit_accrual(self, target_weights):
        total_debt = 0
        nav = self.calc_nav()
        for sym in self.assets:
            price = self.get_prices(self.current_date).get(sym, 0)
            if price is None or price <= 0:
                continue
            # 应用减值因子计算市值
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            mv = self.holdings[sym] * adj_price
            w = target_weights.get(sym, 0)
            if w > 1.0 and self.holdings[sym] > 0:
                debt = (w - 1.0) * nav
                total_debt += debt
                self.cash -= CONFIG["MARGIN_INTEREST"] * debt
            elif self.holdings[sym] < 0:
                short_value = -self.holdings[sym] * adj_price
                self.cash -= CONFIG["SHORT_INTEREST"] * short_value
        if total_debt > 0:
            ratio = nav / total_debt
            if ratio < CONFIG["MAINTENANCE_RATIO"]:
                logger.warning(f"维持担保比例 {ratio:.2f} 低于阈值，强制平仓")
                for sym in self.assets:
                    if self.holdings[sym] != 0:
                        price = self.get_prices(self.current_date).get(sym, 0)
                        if price is not None and price > 0:
                            self._sell(sym, self.holdings[sym], price)
                            if self.holdings[sym] < 0:
                                buy_shares = -self.holdings[sym]
                                self.cash -= buy_shares * price * (1 + CONFIG["AUCTION_SLIPPAGE_BPS"])
                                self.holdings[sym] = 0

    def state_7_reconciliation(self):
        nav_calc = self.calc_nav()
        total_mv = 0.0
        prices = self.get_prices(self.current_date)
        for sym in self.assets:
            price = prices.get(sym)
            if price is None:
                continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            total_mv += self.holdings[sym] * adj_price
        diff = abs((self.cash + total_mv) - nav_calc)
        if diff > 0.01:
            raise AssertionError(f"对账不平，差值 {diff:.4f}")

    def get_prices(self, date):
        """
        返回当日有效价格字典。
        若当日无价格，向前追溯最多 10 个交易日，取最近有效价格。
        若仍无，则返回 None。
        """
        prices = {}
        for sym in self.assets:
            cache_key = (sym, date)
            if cache_key in self.price_cache:
                prices[sym] = self.price_cache[cache_key]
                continue
            p = self.bus.query_by_pit(sym, date, "total_return_price")
            if p is None:
                # 向前查找
                found = False
                for delta in range(1, 11):
                    alt_date = date - timedelta(days=delta)
                    p = self.bus.query_by_pit(sym, alt_date, "total_return_price")
                    if p is not None:
                        found = True
                        break
                if not found:
                    p = None
            self.price_cache[cache_key] = p
            prices[sym] = p
        return prices

    def calc_nav(self):
        """计算净资产，对停牌资产应用减值因子"""
        total_mv = 0.0
        prices = self.get_prices(self.current_date)
        for sym in self.assets:
            price = prices.get(sym)
            if price is None or np.isnan(price):
                # 若无价格，则忽略该持仓（理论上前向查找应能找到，但为安全忽略）
                continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            total_mv += self.holdings[sym] * adj_price
        return self.cash + total_mv


def execute(pipeline_context: dict):
    logger.info("=" * 60)
    logger.info("Phase 7: 确定性 FSM 回测（含停牌减值处理）")
    logger.info("=" * 60)
    engine = FSMEngine(pipeline_context)
    engine.run()
    pipeline_context['fsm_engine'] = engine
    final_nav = pipeline_context.get('final_nav', 0)
    logger.info(f"回测结束，最终净值: {final_nav:.2f}")
    return pipeline_context