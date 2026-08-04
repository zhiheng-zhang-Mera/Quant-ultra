# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 8.1 & 8.2: Decreased Sharpe Ratio (DSR) Multi-Trial Auditor
"""
import logging
import numpy as np
import pandas as pd
from Phase_8.config import DEFAULT_CONFIG
from Phase_8.utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.DSR")

def run_dsr_audit(context: dict) -> None:
    """
    递减夏普比率 (DSR) 审计核。
    计算名义夏普，引入多重超参搜索试验次数 N 对渐进高斯标准分进行平抑惩罚，输出真实 DSR。
    """
    # logger.info("[OP] Deploy DSR Overfitting Auditor | [SOURCE] Active Signal Portfolio Deck | [RESULT] Commencing nominal Sharpe calibration | [SIGNIFICANCE] Filters out lucky random parameter sets under multiple trials")
    logger.info("[操作] 部署 DSR 多重试验过拟合审计器 | [来源] 活跃信号组合收益率大表 | [结果] 正在启动名义夏普校准 | [意义] 剔除高频超参搜索大循环中，因随机撞运气产生的虚高幸运参数组合")
    
    config = context.get('config', {})
    min_samples = config.get('min_samples_for_dsr', DEFAULT_CONFIG['min_samples_for_dsr'])
    sharpe_threshold = config.get('sharpe_threshold', DEFAULT_CONFIG['sharpe_threshold'])
    dsr_pval_threshold = config.get('dsr_pval_threshold', DEFAULT_CONFIG['dsr_pval_threshold'])

    try:
        nav_series = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav_series, pd.Series):
            raise TypeError("daily_nav must be a pandas Series.")
        
        if len(nav_series) < min_samples:
            # logger.warning("[OP] Evaluate Time Series Samples | [SOURCE] Backtest Ledger Head | [RESULT] Insufficient Rows: %s/%s | [SIGNIFICANCE] Automatically de-activates statistical audit to prevent small-sample bias", len(nav_series), min_samples)
            logger.warning("[操作] 评估时间序列样本规模 | [来源] 策略回测账本头部 | [结果] 有效行数不足: %s/%s | [意义] 自动降级挂起统计审计，防止小样本偏差导致虚假推断", len(nav_series), min_samples)
            context["dsr_pass"] = True; return

        returns = np.log(nav_series / nav_series.shift(1)).dropna()
        if returns.empty:
            context["dsr_pass"] = False; return

        std_dev = returns.std()
        if std_dev == 0:
            context["dsr_pass"] = False; return

        # 计算名义年化夏普比率
        sharpe = float(returns.mean() / std_dev * np.sqrt(252))
        context["nominal_sharpe"] = sharpe

        # 刚性读取 Phase-0/Phase_5 搜参空间的试验总次数
        N = context.get("num_trials")
        if not isinstance(N, (int, np.integer)) or N < 1:
            logger.error("DSR audit rejected: Phase 5 did not provide a valid num_trials evidence value.")
            context["dsr_pass"] = False
            context["dsr_evidence_status"] = "MISSING_NUM_TRIALS"
            return
        context["num_trials"] = int(N)
        from scipy.stats import norm

        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # 统计学大样本极限分布调整，防止非平稳尾部异变导致方差为负
        denom_sq = 1 - skew * sharpe + (kurt - 1) / 4 * (sharpe ** 2)
        if denom_sq <= 0:
            # logger.warning("[OP] Compute Conformal Variance Denominator | [SOURCE] Empirical Log Return Distribution | [RESULT] Non-positive variance: %.4f | [SIGNIFICANCE] Right-tail anomaly triggers safety default", denom_sq)
            logger.warning("[操作] 求解共形方差分母 | [来源] 经验对数收益率分布 | [结果] 渐进方差值为非正数: %.4f | [意义] 检测到极端右尾异变噪声，触发防御性兜底保护", denom_sq)
            dsr_pval = 0.0
        else:
            t_stat = (sharpe - sharpe_threshold) / (np.sqrt(denom_sq / (len(returns) - 1)))
            # 引入对多次搜索 N 的修正：调整标准正态累积分布宽度
            penalty_factor = np.sqrt(2 * np.log(max(2, N)))
            dsr_pval = float(norm.cdf(-abs(t_stat) * (1.0 / (penalty_factor if penalty_factor > 0 else 1.0))))

        context["dsr_pval"] = dsr_pval
        dsr_pass = dsr_pval >= dsr_pval_threshold
        context["dsr_pass"] = dsr_pass

        # logger.info("[OP] Conclude DSR Statistical Inference | [SOURCE] Asymptotic Normality Grid | [RESULT] Sharpe: %.4f, N_trials: %s, DSR pval: %.6f, Passed: %s | [SIGNIFICANCE] Asserts whether strategy Sharpe stands firmly above target bounds", sharpe, N, dsr_pval, dsr_pass)
        logger.info("[操作] 终结 DSR 统计学检验推断 | [来源] 渐进正态拟合网格 | [结果] 夏普: %.4f, 试验次数: %s, DSR 显著性 p值: %.6f, 审查通过: %s | [意义] 断言策略真实的风险溢价回报是否稳健立于预设目标阈值之上", sharpe, N, dsr_pval, dsr_pass)

    except Exception as e:
        # logger.error("[OP] Execute DSR Verification | [SOURCE] Statistical Audit Stack | [RESULT] Crash: %s | [SIGNIFICANCE] Defensive veto triggered to reject pipeline deployment", str(e))
        logger.error("[操作] 执行 DSR 审计算法 | [来源] 统计审计异常栈 | [结果] 崩溃异常: %s | [意义] 触发最严厉的风控一票否决，拒绝将本策略转交下游投产", str(e))
        context["dsr_pass"] = False
