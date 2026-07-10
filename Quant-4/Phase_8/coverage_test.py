# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 8.3: Christoffersen Conditional Coverage LR Test
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from Phase_8.config import DEFAULT_CONFIG
from Phase_8.utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.Coverage")

def christoffersen_lr_core(viol_series: pd.Series) -> float:
    """条件独立性似然比检验计算核心（含一阶马尔可夫转移矩阵计数）"""
    if len(viol_series) < 10:
        return 1.0
    
    n00 = n01 = n10 = n11 = 0
    prev = viol_series.iloc[0]
    
    for v in viol_series.iloc[1:]:
        if prev == 0 and v == 0: n00 += 1
        elif prev == 0 and v == 1: n01 += 1
        elif prev == 1 and v == 0: n10 += 1
        elif prev == 1 and v == 1: n11 += 1
        prev = v

    if n01 == 0 and n11 == 0:
        return 1.0

    pi_01 = n01 / ((n00 + n01) if (n00 + n01) > 0 else 1)
    pi_11 = n11 / ((n10 + n11) if (n10 + n11) > 0 else 1)
    pi_val = (n01 + n11) / (n - 1)

    # 极大似然连续乘积比值求取 (Likelihood Ratio under independence null)
    ln_null = n00 * np.log(1 - pi_val + 1e-9) + (n01 + n10) * np.log(pi_val + 1e-9) + n11 * np.log(pi_val + 1e-9)
    ln_alt = n00 * np.log(1 - pi_01 + 1e-9) + n01 * np.log(pi_01 + 1e-9) + n10 * np.log(1 - pi_11 + 1e-9) + n11 * np.log(pi_11 + 1e-9)
    
    lr_statistic = -2 * (ln_null - ln_alt)
    p_value = 1.0 - chi2.cdf(max(0.0, lr_statistic), df=1)
    return float(p_value)

def run_christoffersen_test(context: dict) -> None:
    """Christoffersen 联合无条件与条件独立覆盖审计总入口"""
    logger.info("[OP] Initialize Christoffersen Safety Probe | [SOURCE] FSM Risk Violation Logs | [RESULT] Triggering multi-regime LR sweeps | [SIGNIFICANCE] Confirms whether strategy drawdown anomalies cluster in time")
    logger.info("[操作] 初始化 Christoffersen 安全探针 | [来源] 状态机交易超限违规记录日志 | [结果] 正在启动多轨体制似然比扫描 | [意义] 证明极值回撤事件在时间轴上是否呈现聚集性，防止雪崩式爆仓")
    
    config = context.get('config', {})
    min_coverage = config.get('min_coverage', DEFAULT_CONFIG['min_coverage'])
    pval_threshold = config.get('christoffersen_pval_threshold', DEFAULT_CONFIG['christoffersen_pval_threshold'])

    try:
        violations = safe_get_shadow(context, "violations")
        nav_series = safe_get_shadow(context, "daily_nav")
        if not isinstance(violations, pd.Series) or not isinstance(nav_series, pd.Series):
            raise TypeError("violations and daily_nav must be pandas Series.")

        returns = np.log(nav_series / nav_series.shift(1)).dropna()
        common_idx = violations.index.intersection(returns.index)
        
        violations = violations.loc[common_idx]
        returns = returns.loc[common_idx]

        # 三等分滑动波动率体制划分 (Volatility Regime Shunting)
        vol = returns.rolling(20, min_periods=10).std()
        vol_quantiles = vol.quantile([0.33, 0.67]).values
        
        regimes = {
            "full": violations,
            "low_vol": violations[vol <= vol_quantiles[0]],
            "mid_vol": violations[(vol > vol_quantiles[0]) & (vol <= vol_quantiles[1])],
            "high_vol": violations[vol > vol_quantiles[1]],
        }

        emp_cov = float(1.0 - violations.mean())
        context["empirical_coverage"] = emp_cov
        unconditional_pass = emp_cov >= min_coverage

        regime_pvals = {}
        for name, ser in regimes.items():
            if len(ser) >= 10:
                regime_pvals[name] = christoffersen_lr_core(ser)
            else:
                logger.warning("[OP] Evaluate Local Independent Regime | [SOURCE] Slice: %s | [RESULT] Length too short (%s) | [SIGNIFICANCE] Safely skip LR calculation for this subspace", name, len(ser))
                logger.warning("[操作] 评估局部局部独立体制 | [来源] 体制子切片: %s | [结果] 样本点过少 (%s) | [意义] 安全跳过该局部子空间的似然比解算，防范高维稀疏误差")
                regime_pvals[name] = 1.0

        context["christoffersen_regime_pvals"] = regime_pvals
        all_pvals_ok = all(p >= pval_threshold for p in regime_pvals.values())
        context["christoffersen_pass"] = bool(unconditional_pass and all_pvals_ok)

        logger.info("[OP] Finish Christoffersen Risk Audit | [SOURCE] Multi-Regime LR Results | [RESULT] Emp Coverage: %.4f (Required: %.4f), Regime p-vals: %s | [SIGNIFICANCE] Guards portfolio against systematic clustered tail events", emp_cov, min_coverage, regime_pvals)
        logger.info("[操作] 终结 Christoffersen 联合风险审计 | [来源] 多体制似然比汇总大表 | [结果] 无条件覆盖率: %.4f (保底线: %.4f), 各体制下独立性p值: %s | [意义] 严防系统性回撤在特定高波动或低流动性极端体制下发生链式聚集爆发")

    except Exception as e:
        logger.error("[OP] Execute Coverage Audit | [SOURCE] LR Estimator Engine | [RESULT] Failure: %s | [SIGNIFICANCE] Veto filter triggered to protect master pipeline", str(e))
        logger.error("[操作] 执行条件覆盖率审计 | [来源] 似然比估值核心引擎 | [结果] 算法失败: %s | [意义] 触发风控强熔断一票否决，阻止该策略进入投产阶段")
        context["christoffersen_pass"] = False