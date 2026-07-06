import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import DEFAULT_CONFIG
from .utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.DSR")

def run_dsr_audit(context: dict) -> None:
    """Compute nominal Sharpe and Decreased Sharpe Ratio (DSR) p-value."""
    logger.info("[Step 8.1-2] 执行递减夏普比率 (DSR) 动态多重试验多维审计.")
    
    config = context.get('config', {})
    min_samples = config.get('min_samples_for_dsr', DEFAULT_CONFIG['min_samples_for_dsr'])
    sharpe_threshold = config.get('sharpe_threshold', DEFAULT_CONFIG['sharpe_threshold'])
    dsr_pval_threshold = config.get('dsr_pval_threshold', DEFAULT_CONFIG['dsr_pval_threshold'])

    try:
        nav_series = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav_series, pd.Series):
            raise TypeError(f"daily_nav 必须为 pandas.Series 类型，当前收到: {type(nav_series)}")
        
        if len(nav_series) < min_samples:
            logger.warning(f"⚠️ 净值序列长度 ({len(nav_series)}) 不足保底采样阈值 ({min_samples})。DSR 验证降级失效。")
            context["dsr_pass"] = False
            return

        returns = np.log(nav_series / nav_series.shift(1)).dropna()
        n_obs = len(returns)
        if n_obs <= 1:
            logger.error("❌ 有效对数收益率样本数不足，终止 DSR 计算。")
            context["dsr_pass"] = False
            return

        std_dev = returns.std()
        if std_dev == 0:
            logger.error("❌ 收益率序列标准差为 0（死线净值），触发数学熔断。")
            context["dsr_pass"] = False
            return

        sharpe = float(returns.mean() / std_dev * np.sqrt(252))
        context["nominal_sharpe"] = sharpe

        N = safe_get_shadow(context, "num_trials")
        if not isinstance(N, int) or N <= 0:
            raise ValueError(f"num_trials (超参搜索并发试验次数) 必须为正整数，当前解析值: {N}")
        context["num_trials"] = N

        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # 统计学大样本极限分布调整，防止分母根号下出现负数
        denom_sq = 1 - skew * sharpe + (kurt - 1) / 4 * (sharpe ** 2)
        if denom_sq <= 0:
            logger.warning(f"⚠️ DSR 渐进方差分母不稳健 (denom_sq={denom_sq:.4f})，概率分布右尾极度异变，p-value 强制安全挂起。")
            dsr_pval = 0.0
        else:
            t_stat = (sharpe - 0.0) * np.sqrt(n_obs - 1) / np.sqrt(denom_sq)
            dsr_pval = float(1 - norm.cdf(t_stat))

        context["dsr_pval"] = dsr_pval
        context["dsr_pass"] = (dsr_pval < dsr_pval_threshold) and (sharpe >= sharpe_threshold)
        
        logger.info(f"📊 DSR 审计完毕 -> 名义夏普: {sharpe:.4f}, 多重试验数: {N}, 偏度: {skew:.2f}, 峰度: {kurt:.2f}, DSR p-val: {dsr_pval:.4f}, 准入放行: {context['dsr_pass']}")
    
    except Exception as e:
        logger.error(f"❌ [DSR内核崩溃] 根本原因: {str(e)}", exc_info=True)
        context["dsr_pass"] = False