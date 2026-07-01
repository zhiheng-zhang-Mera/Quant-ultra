import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .config import DEFAULT_CONFIG
from .utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.Coverage")

def christoffersen_lr(viol_series: pd.Series) -> float:
    """核心似然比检验算子，带完备的转移矩阵计数监控"""
    if len(viol_series) < 10:
        return 1.0
    
    n = len(viol_series)
    n00 = n01 = n10 = n11 = 0
    prev = viol_series.iloc[0]
    
    for v in viol_series.iloc[1:]:
        if prev == 0 and v == 0: n00 += 1
        elif prev == 0 and v == 1: n01 += 1
        elif prev == 1 and v == 0: n10 += 1
        elif prev == 1 and v == 1: n11 += 1
        prev = v

    # 打印详细转移计数矩阵，便于穿透定位
    logger.debug(f"[Markov Matrix] n00:{n00}, n01:{n01}, n10:{n10}, n11:{n11}")

    if n01 == 0 and n11 == 0:
        return 1.0

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_val = (n01 + n11) / n

    try:
        ln_null = (n00 + n10) * np.log(1 - pi_val + 1e-12) + (n01 + n11) * np.log(pi_val + 1e-12)
        ln_alt = n00 * np.log(1 - pi_01 + 1e-12) + n01 * np.log(pi_01 + 1e-12) + \
                 n10 * np.log(1 - pi_11 + 1e-12) + n11 * np.log(pi_11 + 1e-12)
        lr_stat = -2 * (ln_null - ln_alt)
        p_val = float(1 - chi2.cdf(lr_stat, df=1))
        return p_val
    except Exception as e:
        logger.error(f"⚠️ 似然比对数空间映射失败 (可能由于极度断层引发): {str(e)}")
        return 0.0

def run_christoffersen_test(context: dict) -> None:
    """Christoffersen conditional coverage test partitioned across endogenous volatility regimes."""
    logger.info("[Step 8.3] 启动 Christoffersen 违约条件覆盖及波动率体制独立性检验.")
    
    config = context.get('config', {})
    min_coverage = config.get('min_coverage', DEFAULT_CONFIG['min_coverage'])
    pval_threshold = config.get('christoffersen_pval_threshold', DEFAULT_CONFIG['christoffersen_pval_threshold'])

    try:
        violations = safe_get_shadow(context, "violations")
        returns = safe_get_shadow(context, "daily_returns")
        
        if not isinstance(violations, pd.Series) or not isinstance(returns, pd.Series):
            raise TypeError("violations 与 daily_returns 均必须为 pandas.Series 类型。")

        if violations.empty or returns.empty:
            logger.warning("⚠️ 违约向量或基础收益率序列为空，跳过 Christoffersen 条件检验。")
            context["christoffersen_pass"] = False
            return

        common_idx = violations.index.intersection(returns.index)
        if len(common_idx) < 50:
            logger.warning(f"⚠️ 时空对齐后的有效样本数 ({len(common_idx)}) 不足 50 天，无法提供稳健统计学推断。")
            context["christoffersen_pass"] = False
            return
            
        violations = violations.loc[common_idx]
        returns = returns.loc[common_idx]

        # 动态波动率体制三等分切片
        vol = returns.rolling(20, min_periods=10).std()
        vol_quantiles = vol.quantile([1/3, 2/3]).values
        
        regimes = {
            "full": violations,
            "low_vol": violations[vol <= vol_quantiles[0]],
            "mid_vol": violations[(vol > vol_quantiles[0]) & (vol <= vol_quantiles[1])],
            "high_vol": violations[vol > vol_quantiles[1]],
        }

        emp_cov = float(1 - violations.mean())
        context["empirical_coverage"] = emp_cov
        unconditional_pass = emp_cov >= min_coverage

        regime_pvals = {}
        for name, ser in regimes.items():
            if len(ser) >= 10:
                regime_pvals[name] = christoffersen_lr(ser)
            else:
                logger.warning(f"⚠️ 体制分流 [{name}] 包含交易日少于 10 天，该局部独立性测试安全跳过。")
                regime_pvals[name] = 1.0

        context["christoffersen_regime_pvals"] = regime_pvals
        all_pvals_ok = all(p >= pval_threshold for p in regime_pvals.values())
        context["christoffersen_pass"] = bool(unconditional_pass and all_pvals_ok)

        logger.info(f"📊 VaR 覆盖核验 -> 实际无条件覆盖率: {emp_cov:.4f} (硬放行准则: >= {min_coverage}) -> 达标: {unconditional_pass}")
        logger.info(f"📊 体制马尔可夫独立性检验 p-values: {regime_pvals} (准入线: >= {pval_threshold}) -> 达标: {all_pvals_ok}")

    except Exception as e:
        logger.error(f"❌ [条件覆盖校验崩溃] 错误详情: {str(e)}", exc_info=True)
        context["christoffersen_pass"] = False