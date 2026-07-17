'''
附加增强模块：EVT 极值理论黑天鹅压力生成器 (Phase 08)
设计模式：场景注入混入(Mixin)
使用方法：附加到现有的 Monte Carlo 蒙特卡洛引擎中。

扩展功能：
1. 参数化风险因子冲击测试（Beta 冲击、行业波动率飙升、融资利率骤增）
2. 基于 Block Bootstrap 的策略覆盖度测试（生成合成净值曲线，计算 Expected Shortfall）
3. 统一入口 run_enhanced_stress_tests()，集成上述测试并更新 context 审计标志
'''
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm, chi2
from typing import Dict, Any, Optional
from Phase_8.utils import safe_get_shadow, compute_max_drawdown
from Phase_8.config import DEFAULT_CONFIG

logger = logging.getLogger("AuditStressTest.Enhanced")

# ============================================================================
# 原有黑天鹅注入器（保留，可作为 Bootstrap 后处理增强）
# ============================================================================
class BlackSwanInjectorExt:
    def __init__(self, tail_percentile=1, multiplier=3.0):
        self.tail_percentile = tail_percentile
        self.multiplier = multiplier
        
    def apply_stress(self, simulated_paths, historical_returns):
        logging.info("增强注入：利用 EVT 极值理论向演化路径注入局部黑天鹅冲击...")
        tail_events = np.percentile(historical_returns, self.tail_percentile)
        mask = np.random.rand(*simulated_paths.shape) < 0.005
        simulated_paths[mask] *= (1 + tail_events * self.multiplier)
        return simulated_paths


# ============================================================================
# 新增功能 1：参数化因子冲击测试
# ============================================================================
def run_parametric_shock_test(context: Dict) -> None:
    """
    参数化风险因子冲击：
    - Beta 冲击：模拟大盘瞬间下跌 5%, 10%, 15%，测算组合净值跌幅
    - 行业冲击：假设核心行业波动率飙升 300%，并导致该行业收益率偏离
    - 资金断流：融资/融券利率跳升至 20%，计算杠杆成本对净值的侵蚀
    所有冲击均基于原始净值序列进行改造，计算冲击后的最大回撤。
    """
    logger.info("[操作] 启动参数化风险因子冲击测试 | [来源] 合成压力场景生成器")
    
    try:
        nav = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav, pd.Series) or len(nav) < 2:
            raise ValueError("daily_nav 无效或长度不足")

        # 获取持仓权重（用于行业冲击模拟），若无则使用均匀权重
        weights = safe_get_shadow(context, "daily_weights", None)
        if weights is None:
            # 若无权重，生成等权模拟
            assets = context.get("assets", [])
            if assets:
                weights = pd.DataFrame(1.0/len(assets), index=nav.index, columns=assets)
            else:
                weights = None

        # ---------- 1. Beta 冲击 ----------
        # 若无 Beta 估计，默认 Beta = 1.0，直接对净值乘以冲击因子
        beta = context.get("portfolio_beta", 1.0)
        shock_levels = [0.05, 0.10, 0.15]   # 5%, 10%, 15%
        beta_drawdowns = {}
        for lev in shock_levels:
            # 模拟大盘瞬间下跌 lev，组合净值下跌 beta * lev
            shock_factor = 1 - beta * lev
            shocked_nav = nav * shock_factor
            dd = compute_max_drawdown(shocked_nav)
            beta_drawdowns[f"beta_shock_{int(lev*100)}pct"] = dd
            logger.info("[操作] Beta 冲击 %d%% 完成，最大回撤 %.4f", int(lev*100), dd)

        # ---------- 2. 行业冲击 ----------
        # 简化：若无行业暴露数据，跳过；若有，则选取权重最大的行业冲击其收益率
        if weights is not None and hasattr(weights, 'columns') and len(weights.columns) > 0:
            # 假设每个证券的行业标签在 context 的 "industry_map" 中（模拟）
            industry_map = context.get("industry_map", {})
            if industry_map:
                # 按行业聚合权重
                industry_weights = pd.DataFrame(index=weights.index)
                for sec, ind in industry_map.items():
                    if sec in weights.columns:
                        industry_weights[ind] = industry_weights.get(ind, 0) + weights[sec]
                # 选总权重最大的行业作为冲击目标
                avg_industry_w = industry_weights.mean()
                if not avg_industry_w.empty:
                    target_ind = avg_industry_w.idxmax()
                    # 模拟该行业所有证券收益率发生 3 倍波动率偏移
                    # 我们用净值乘数模拟：该行业权重部分波动，其他不变
                    # 简单做法：对组合净值施加一个 -10% 的额外冲击（模拟板块暴跌）
                    # 更精确应基于个股收益率回放，此处仅作示意
                    shock_factor = 0.90  # 该行业整体下跌 10%
                    # 计算该行业权重占比
                    weight_series = industry_weights[target_ind]
                    # 对每个时点，组合净值变化 = 原始净值 * (1 - weight * (1 - shock_factor))
                    # 即行业部分下跌，其他不变
                    # 但需按时间序列逐点应用，因为权重随时间变化
                    # 为简化，取平均权重，施加一个恒定的冲击
                    avg_w = weight_series.mean()
                    shocked_nav = nav * (1 - avg_w * (1 - shock_factor))
                    dd = compute_max_drawdown(shocked_nav)
                    industry_drawdowns = {f"industry_shock_{target_ind}": dd}
                    logger.info("[操作] 行业冲击 '%s' 完成，最大回撤 %.4f", target_ind, dd)
                else:
                    industry_drawdowns = {}
            else:
                industry_drawdowns = {}
        else:
            industry_drawdowns = {}

        # ---------- 3. 资金断流冲击 ----------
        # 假设组合有融资融券，融资成本上升至 20% 年化
        # 需要融资金额，若无则跳过
        leverage_ratio = context.get("leverage_ratio", 0.0)  # 负债/净资产
        if leverage_ratio > 0:
            # 年化利率冲击从 5% 升至 20%，增加 15% 成本
            cost_increase = 0.15  # 15% 增量
            # 对每日净值扣除额外利息：利息 = 负债 * 日利率增量
            # 负债 = 净资产 * leverage_ratio
            # 每日额外成本 = nav * leverage_ratio * (cost_increase / 252)
            daily_extra_cost = nav * leverage_ratio * (cost_increase / 252)
            # 累计成本影响净值：从第一天开始累积扣除
            cumulative_cost = daily_extra_cost.cumsum()
            shocked_nav = nav - cumulative_cost
            dd = compute_max_drawdown(shocked_nav)
            liquidity_drawdown = {"funding_shock": dd}
            logger.info("[操作] 融资利率冲击完成，最大回撤 %.4f", dd)
        else:
            liquidity_drawdown = {}

        # 汇总所有冲击结果
        all_shock_results = {**beta_drawdowns, **industry_drawdowns, **liquidity_drawdown}
        context["parametric_shock_drawdowns"] = all_shock_results

        # 设定硬性阈值：任何冲击下最大回撤不得超过 20%（可配置）
        threshold = context.get("parametric_shock_threshold", 0.20)
        shock_pass = all(v <= threshold for v in all_shock_results.values()) if all_shock_results else True
        context["parametric_shock_pass"] = bool(shock_pass)

        logger.info("[操作] 参数化冲击测试完成，通过状态: %s", shock_pass)

    except Exception as e:
        logger.error("[操作] 参数化冲击测试异常: %s", str(e))
        context["parametric_shock_pass"] = False
        context["parametric_shock_drawdowns"] = {}


# ============================================================================
# 新增功能 2：Block Bootstrap 策略覆盖度测试
# ============================================================================
def block_bootstrap_sample(returns: pd.Series, block_size: int = 20, n_bootstrap: int = 1000) -> np.ndarray:
    """
    块重采样生成合成收益率序列。
    参数:
        returns: 日收益率 Series (带时间索引)
        block_size: 块长度（默认 20 个交易日）
        n_bootstrap: 生成序列数量
    返回:
        (n_bootstrap, len(returns)) 的 numpy 数组，每行是一条合成收益率序列
    """
    n = len(returns)
    if n <= block_size:
        # 若样本太短，降级为普通 Bootstrap（有放回抽取单点）
        block_size = 1
    # 计算可选的块起始索引
    max_start = n - block_size
    if max_start < 1:
        max_start = 1
    block_starts = np.arange(0, max_start + 1)
    
    boot_returns = np.zeros((n_bootstrap, n))
    for i in range(n_bootstrap):
        idx = 0
        while idx < n:
            start = np.random.choice(block_starts)
            end = min(start + block_size, n)
            block_len = end - start
            remaining = n - idx
            take = min(block_len, remaining)
            boot_returns[i, idx:idx+take] = returns.iloc[start:start+take].values
            idx += take
    return boot_returns


def run_bootstrap_es_test(context: Dict) -> None:
    """
    基于 Block Bootstrap 生成 1000 条合成净值曲线，计算最大回撤分布，
    并提取 99% 置信度下的 Expected Shortfall (ES) 作为尾部风险指标。
    若 ES 超过阈值（默认 15%），则测试不通过。
    """
    logger.info("[操作] 启动 Block Bootstrap 覆盖度测试 | [来源] 历史收益率块重采样")
    
    try:
        nav = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav, pd.Series) or len(nav) < 50:
            raise ValueError("daily_nav 长度不足 (需要至少50个交易日)")

        # 计算对数收益率
        returns = np.log(nav / nav.shift(1)).dropna()
        if len(returns) < 20:
            raise ValueError("有效收益率样本不足")

        # 获取配置参数
        config = context.get("config", {})
        n_bootstrap = config.get("bootstrap_iterations", 1000)
        block_size = config.get("bootstrap_block_size", 20)
        es_confidence = config.get("bootstrap_es_confidence", 0.99)
        es_threshold = config.get("bootstrap_es_threshold", 0.15)  # 15%

        # 生成合成收益率矩阵
        boot_returns = block_bootstrap_sample(returns, block_size, n_bootstrap)

        # 计算每条路径的净值（初始净值 1）
        boot_nav = np.exp(np.cumsum(boot_returns, axis=1))
        # 计算每条路径的最大回撤
        max_drawdowns = []
        for i in range(n_bootstrap):
            path = boot_nav[i, :]
            dd = compute_max_drawdown(pd.Series(path))
            max_drawdowns.append(-dd)  # 转为正数表示回撤幅度（正值）
        
        max_drawdowns = np.array(max_drawdowns)
        # 计算 VaR（百分位数）
        var = np.percentile(max_drawdowns, es_confidence * 100)
        # 计算 ES：超过 VaR 的平均回撤
        es = max_drawdowns[max_drawdowns >= var].mean() if np.any(max_drawdowns >= var) else var

        # 存入 context
        context["bootstrap_max_drawdowns"] = max_drawdowns.tolist()
        context["bootstrap_var"] = float(var)
        context["bootstrap_es"] = float(es)

        # 判断是否通过
        bootstrap_pass = es <= es_threshold
        context["bootstrap_pass"] = bootstrap_pass

        logger.info("[操作] Bootstrap 测试完成: ES=%.4f, 阈值=%.4f, 通过=%s", es, es_threshold, bootstrap_pass)

    except Exception as e:
        logger.error("[操作] Bootstrap 测试异常: %s", str(e))
        context["bootstrap_pass"] = False
        context["bootstrap_es"] = np.nan


# ============================================================================
# 统一入口：运行所有增强测试
# ============================================================================
def run_enhanced_stress_tests(context: Dict) -> None:
    """
    增强压力测试总入口，依次执行参数化冲击和 Bootstrap 测试，
    并将各自的 pass 标志写入 context。
    """
    logger.info("=" * 60)
    logger.info("[操作] 启动 Phase-8 增强压力测试套件")
    logger.info("=" * 60)

    run_parametric_shock_test(context)
    run_bootstrap_es_test(context)

    # 综合增强测试通过条件：两个子测试均通过
    overall_enhanced_pass = (
        context.get("parametric_shock_pass", False) and
        context.get("bootstrap_pass", False)
    )
    context["enhanced_stress_pass"] = overall_enhanced_pass

    logger.info("[操作] 增强压力测试综合结论: %s", overall_enhanced_pass)
    logger.info("=" * 60)