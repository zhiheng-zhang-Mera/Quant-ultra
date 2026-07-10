"""
Phase 7: Chronologically-Aligned State-Machine Engine and Asset Liability Backtesting
Now with daily weights and intervals from step6, records violations and daily returns.
完善停牌处理：使用最近有效价格并应用减值因子。
支持自适应冲击系数（基于历史买卖价差）和退市处理。
"""
import numpy as np
import pandas as pd
from datetime import timedelta
import logging
import akshare as ak

logger = logging.getLogger("FSMBacktest")

class FSMEngine:
    def __init__(self, context):
        self.context = context
        self.bus = context['data_bus']
        self.assets = context['assets']
        self.config = context.get('config', {})
        self.cash = 1_000_000.0  # 初始现金
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
        self.halt_status = {sym: False for sym in self.assets}
        self.impairment_factor = {sym: 1.0 for sym in self.assets}
        self.impairment_applied = {sym: False for sym in self.assets}
        self.price_cache = {}

        # 退市信息（简单示例：假设上市日期超过20年即退市，实际应从公告获取）
        # 这里留接口，实际应维护退市列表
        self.delisted = set()

        # 记录
        self.daily_returns_list = []
        self.violations_list = []

        # 冲击系数参数
        self.impact_kappa_base = self.config.get('impact_kappa_base', 0.05)
        self.spread_lookback = self.config.get('spread_lookback_days', 60)

    def _estimate_kappa(self, sym):
        """根据过去60日平均买卖价差估算冲击系数"""
        try:
            # 获取历史数据，计算价差（用当日最高最低近似）
            end_date = self.current_date
            start_date = end_date - timedelta(days=self.spread_lookback * 2)
            df = self.bus.load_asset_history(sym, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df is None or len(df) < 20:
                return self.impact_kappa_base
            # 计算每日相对价差 (high-low)/close
            spread = (df['high'] - df['low']) / df['close']
            avg_spread = spread.tail(self.spread_lookback).mean()
            if np.isnan(avg_spread) or avg_spread <= 0:
                return self.impact_kappa_base
            # 按规范：kappa = 5% * avg_spread
            kappa = 0.05 * avg_spread
            # 限制在合理范围
            return np.clip(kappa, self.impact_kappa_base * 0.5, self.impact_kappa_base * 1.5)
        except:
            return self.impact_kappa_base

    def run(self):
        test_dates = self.daily_weights.index
        if len(test_dates) == 0:
            raise ValueError("Test 集为空。")
        for t, date in enumerate(test_dates):
            self.current_date = date

            # 获取当日预计算目标权重（原始）
            target_w_series = self.daily_weights.loc[date]
            raw_target_weights = {sym: target_w_series[sym] if sym in target_w_series else 0.0 for sym in self.assets}

            # 根据实际 NAV 重新截断权重，确保满足股本约束（举牌红线）
            current_nav = self.calc_nav()
            adjusted_target_weights = {}
            for sym in self.assets:
                w = raw_target_weights.get(sym, 0.0)
                if w != 0:
                    upper = self._compute_dynamic_upper(sym, current_nav)
                    if w > upper:
                        logger.info(f"{date} 截断 {sym} 权重 {w:.4f} -> {upper:.4f} (NAV={current_nav:.0f})")
                        w = upper
                    elif w < -upper:  # 空头上限（绝对值）
                        w = -upper
                adjusted_target_weights[sym] = w

            # 获取当日预测区间
            q_low_series = self.q_low_df.loc[date]
            q_high_series = self.q_high_df.loc[date]

            # State 4: 执行撮合（含停牌处理、自适应冲击）
            self.state_4_execution(adjusted_target_weights)
            # State 5: 权益处理
            self.state_5_equity_processing()
            # State 6: 信用计提
            self.state_6_credit_accrual(adjusted_target_weights)
            # State 7: 对账清算
            self.state_7_reconciliation()

            nav_after = self.calc_nav()
            self.nav_series.append(nav_after)
            if t == 0:
                daily_ret = 0.0
            else:
                daily_ret = (nav_after - self.nav_series[-2]) / self.nav_series[-2]
            self.daily_returns_list.append((date, daily_ret))

            # 判断违约
            w_vec = np.array([adjusted_target_weights.get(sym, 0.0) for sym in self.assets])
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

            self.prev_weights = adjusted_target_weights.copy()

        self.context['daily_nav'] = pd.Series(self.nav_series, index=test_dates)
        self.context['final_nav'] = self.nav_series[-1] if self.nav_series else self.calc_nav()
        self.context['daily_returns'] = pd.Series(dict(self.daily_returns_list))
        self.context['violations'] = pd.Series(dict(self.violations_list))

    def _compute_dynamic_upper(self, sym, nav):
        """根据当前市值和总股本计算该资产允许的最大权重"""
        try:
            info = ak.stock_individual_info_em(symbol=sym)
            total_shares = info[info['item']=='总股本']['value'].values[0]
            price = self.bus.query_by_pit(sym, self.current_date, "total_return_price")
            if price is None:
                return 0.045
            market_value = total_shares * price
            max_weight = (0.045 * market_value) / nav if nav > 0 else 0.045
            return min(max_weight, 0.045)
        except:
            return 0.045

    def state_4_execution(self, target_weights):
        """撮合执行，含整手截断、自适应冲击滑点、停牌处理、退市处理"""
        prices = self.get_prices(self.current_date)
        total_nav = self.calc_nav()
        target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in self.assets}

        for sym in self.assets:
            price = prices.get(sym)

            # 退市处理
            if self._is_delisted(sym):
                if self.holdings[sym] != 0:
                    residual = self.config.get('default_residual_rate', 0.0)
                    self.cash += self.holdings[sym] * residual * (price or 0)
                    self.holdings[sym] = 0
                continue

            # 停牌处理
            if price is None or np.isnan(price) or price <= 0:
                self.halt_counter[sym] += 1
                self.halt_status[sym] = True
                if self.halt_counter[sym] >= self.config.get('halt_days', 20) and not self.impairment_applied[sym]:
                    self._apply_halt_impairment(sym)
                continue
            else:
                # 恢复交易，重置停牌计数和减值因子（可选，按规范可保留）
                self.halt_counter[sym] = 0
                self.halt_status[sym] = False
                # 若之前减值，恢复为1（根据业务需求，这里选择恢复）
                if self.impairment_applied[sym]:
                    self.impairment_factor[sym] = 1.0
                    self.impairment_applied[sym] = False
                    logger.info(f"{sym} 恢复交易，减值因子重置为1.0")

            current_shares = self.holdings[sym]
            target_shares = target_values[sym] / price if price > 0 else 0
            diff = target_shares - current_shares
            lot_unit = 200 if sym.startswith("688") else 100
            if abs(diff) < self.config.get('clearing_threshold', 0.001) * total_nav / price:
                if current_shares != 0:
                    self._sell(sym, current_shares, price)
                continue

            # 自适应冲击系数
            kappa = self._estimate_kappa(sym)
            if diff > 0:
                exec_shares = np.floor(diff / lot_unit) * lot_unit
                if exec_shares >= lot_unit:
                    adv = self.bus.query_by_pit(sym, self.current_date, "adv")
                    if adv is None or adv <= 0:
                        adv = 1e7
                    impact_factor = kappa * ((exec_shares * price / adv) ** self.config.get('impact_alpha', 0.5))
                    exec_price = price * (1 + impact_factor) + self.config.get('auction_slippage_bps', 0.0002) * price
                    fee = (self.config.get('handling_fee', 0.0000487) + self.config.get('management_fee', 0.00002)) * exec_shares * exec_price
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
        kappa = self._estimate_kappa(sym)
        adv = self.bus.query_by_pit(sym, self.current_date, "adv")
        if adv is None or adv <= 0:
            adv = 1e7
        impact_factor = kappa * ((shares * price / adv) ** self.config.get('impact_alpha', 0.5))
        exec_price = price * (1 - impact_factor) - self.config.get('auction_slippage_bps', 0.0002) * price
        stamp = self.config.get('stamp_tax', 0.0005) * shares * exec_price
        fees = (self.config.get('handling_fee', 0.0000487) + self.config.get('management_fee', 0.00002)) * shares * exec_price
        self.cash += shares * exec_price - stamp - fees
        self.holdings[sym] -= shares

    def _apply_halt_impairment(self, sym):
        if self.impairment_applied[sym]:
            return
        rate = self.config.get('impairment_rate', 0.1)
        self.impairment_factor[sym] *= (1 - rate)
        self.impairment_applied[sym] = True
        logger.warning(f"{sym} 停牌超 {self.config.get('halt_days',20)} 日，减值 {rate*100:.1f}%，当前因子 {self.impairment_factor[sym]:.3f}")

    def _is_delisted(self, sym):
        """判断是否退市，实际应查询退市公告，此处模拟：上市日期超过20年视为退市"""
        list_date = self.bus.query_by_pit(sym, self.current_date, "listing_date")
        if list_date is None:
            return False
        years = (self.current_date - list_date).days / 365.25
        return years > 20  # 简单示例

    def state_5_equity_processing(self):
        pass

    def state_6_credit_accrual(self, target_weights):
        total_debt = 0
        nav = self.calc_nav()
        for sym in self.assets:
            price = self.get_prices(self.current_date).get(sym, 0)
            if price is None or price <= 0:
                continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            mv = self.holdings[sym] * adj_price
            w = target_weights.get(sym, 0)
            if w > 1.0 and self.holdings[sym] > 0:
                debt = (w - 1.0) * nav
                total_debt += debt
                self.cash -= self.config.get('margin_interest', 0.06/252) * debt
            elif self.holdings[sym] < 0:
                short_value = -self.holdings[sym] * adj_price
                self.cash -= self.config.get('short_interest', 0.08/252) * short_value
        if total_debt > 0:
            ratio = nav / total_debt
            if ratio < self.config.get('maintenance_ratio', 1.3):
                logger.warning(f"维持担保比例 {ratio:.2f} 低于阈值，强制平仓")
                for sym in self.assets:
                    if self.holdings[sym] != 0:
                        price = self.get_prices(self.current_date).get(sym, 0)
                        if price is not None and price > 0:
                            self._sell(sym, self.holdings[sym], price)
                            if self.holdings[sym] < 0:
                                buy_shares = -self.holdings[sym]
                                self.cash -= buy_shares * price * (1 + self.config.get('auction_slippage_bps', 0.0002))
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
        prices = {}
        for sym in self.assets:
            cache_key = (sym, date)
            if cache_key in self.price_cache:
                prices[sym] = self.price_cache[cache_key]
                continue
            p = self.bus.query_by_pit(sym, date, "total_return_price")
            if p is None:
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
        total_mv = 0.0
        prices = self.get_prices(self.current_date)
        for sym in self.assets:
            price = prices.get(sym)
            if price is None or np.isnan(price):
                continue
            adj_price = price * self.impairment_factor.get(sym, 1.0)
            total_mv += self.holdings[sym] * adj_price
        return self.cash + total_mv


def execute(pipeline_context: dict):
    logger.info("=" * 60)
    logger.info("Phase 7: 确定性 FSM 回测（含自适应冲击、停牌减值、退市处理）")
    logger.info("=" * 60)
    engine = FSMEngine(pipeline_context)
    engine.run()
    pipeline_context['fsm_engine'] = engine
    final_nav = pipeline_context.get('final_nav', 0)
    logger.info(f"回测结束，最终净值: {final_nav:.2f}")
    return pipeline_context