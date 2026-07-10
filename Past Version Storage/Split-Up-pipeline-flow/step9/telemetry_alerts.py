"""
Quant-Ultra Flow - Step 9.3: Telemetry Dashboard & Pre-emptive Nested System Alarms
Computes crowding indexes and residual volatility parameters to safeguard liquidity.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .config import CROWDED_CORR_THRESHOLD, VOL_COMPRESS_QUANTILE, VOLATILITY_WINDOW

logger = logging.getLogger("MLOps.TelemetryAlerts")

def process_nested_risk_telemetry(context: dict) -> dict:
    """
    因子拥挤度与低波泡沫分层嵌套预警。
    当条件A与条件B同时交汇触发时，强制执行生产前置物理熔断，锁定现金，绝对禁运信用。
    """
    logger.info("[Step 9.3] 运维体系常态化追踪风格拥挤度与低波泡沫分位数。")
    
    data_bus = context.get('data_bus')
    nav_history = context.get('nav_history', [])
    
    if data_bus is None:
        logger.warning("数据总线未就绪，跳过嵌套遥测指标审计。")
        return context

    # ---------- 1. 核验条件 B：低波泡沫（滚动残差波动率压缩至最低 10% 分位数） ----------
    condition_b_triggered = False
    if len(nav_history) >= VOLATILITY_WINDOW + 10:
        try:
            nav_series = pd.Series(nav_history)
            returns = nav_series.pct_change().dropna()
            
            # 计算滚动年化残差波动率序列
            rolling_vols = returns.rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)
            if not rolling_vols.dropna().empty:
                current_vol = rolling_vols.iloc[-1]
                historical_window = rolling_vols.tail(252)
                vol_quantile = float((historical_window < current_vol).mean())
                
                context['current_volatility_quantile'] = vol_quantile
                logger.info(f"策略当前组合净值滚动波动率所处历史分位数: {vol_quantile:.4f}")
                
                if vol_quantile < VOL_COMPRESS_QUANTILE:
                    condition_b_triggered = True
                    logger.warning(f"⚠️ [条件B触发] 低波泡沫产生！滚动波动率分位数 {vol_quantile:.4f} < 下沿红线 {VOL_COMPRESS_QUANTILE}")
        except Exception as e:
            logger.error(f"测算账户滚动波动率分位数失败: {str(e)}")
    else:
        logger.info("净值历史序列过短，无法提取平稳的波动率滚动分位数分布。")
        context['current_volatility_quantile'] = 0.5

    # ---------- 2. 核验条件 A：因子拥挤度（与基准沪深300指数的相关性突破 95% 分位数） ----------
    condition_a_triggered = False
    try:
        benchmark_code = data_bus.get_benchmark_code()
        end_date_str = context.get('current_date')
        if isinstance(end_date_str, datetime):
            end_date_str = end_date_str.strftime("%Y-%m-%d")
        elif end_date_str is None:
            end_date_str = datetime.now().strftime("%Y-%m-%d")
            
        start_date_str = (datetime.strptime(end_date_str, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        bench_df = data_bus.manager.fetch_historical(benchmark_code, start_date_str, end_date_str)
        
        if bench_df is not None and not bench_df.empty and len(nav_history) >= 21:
            bench_df = bench_df.sort_values('date')
            bench_df.set_index('date', inplace=True)
            bench_returns = np.log(bench_df['close'] / bench_df['close'].shift(1)).dropna()
            
            portfolio_returns = pd.Series(nav_history).pct_change().dropna()
            # 严格时序序号硬对齐
            portfolio_returns.index = bench_returns.index[-len(portfolio_returns):]
            common_dates = portfolio_returns.index.intersection(bench_returns.index)
            
            if len(common_dates) >= 15:
                corr_coeff = float(portfolio_returns.loc[common_dates].corr(bench_returns.loc[common_dates]))
                context['benchmark_crowding_correlation'] = corr_coeff
                logger.info(f"策略组合当前与大盘基准 ({benchmark_code}) 的截面相关系数: {corr_coeff:.4f}")
                
                if corr_coeff > CROWDED_CORR_THRESHOLD:
                    condition_a_triggered = True
                    logger.warning(f"⚠️ [条件A触发] 风格极其拥挤！相关系数 {corr_coeff:.4f} > 顶层红线 {CROWDED_CORR_THRESHOLD}")
    except Exception as e:
        logger.error(f"测算基准因子拥挤度相关系数失败: {str(e)}")
        context['benchmark_crowding_correlation'] = 0.0

    # ---------- 3. 分层嵌套联合决策控制 ----------
    context['condition_a_active'] = condition_a_triggered
    context['condition_b_active'] = condition_b_triggered
    
    if condition_a_triggered and condition_b_triggered:
        logger.critical("🚨 🚨 🚨 [最高级别生产前置物理熔断] 风格拥挤突破95% 且 波动率压缩至最低10%！双重重叠交汇！")
        logger.critical("刚性下调个体账户纯现货总持仓上限至 20%，其余 80% 资产进入强行无条件纯现金持币锁死状态。")
        context['enforce_crowded_allocation_cap'] = True
        context['max_allowed_portfolio_exposure'] = 0.20
        # 彻底物理封杀一切两融毒素，拒绝融券、借贷与垫资
        context['leverage_allowed'] = False
        context['lock_remaining_cash_pure_spot'] = True
    else:
        context['enforce_crowded_allocation_cap'] = False
        context['max_allowed_portfolio_exposure'] = 1.0
        context['lock_remaining_cash_pure_spot'] = False
        
    return context