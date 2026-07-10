# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 4.0: Stock Borrowing Compliance Auditor & Downgrade Manager
"""
import logging
import pandas as pd
from typing import Set

logger = logging.getLogger("LabelingWeighting.Borrow")

def fetch_borrowable_stocks(context: dict, trade_date: pd.Timestamp) -> Set[str]:
    audit_logger = context.get("audit_logger")
    date_str = trade_date.strftime("%Y%m%d")
    
    try:
        import akshare as ak
        df = ak.stock_borrow_analysis(date=date_str)
        
        if df.empty:
            logger.warning("[OP] Inspect Margin Short Interest | [SOURCE] Exchange 流式融券总线接口 | [RESULT] DataFrame returned Empty | [SIGNIFICANCE] Triggered zero-allocation fallback to safeguard long-only consistency")
            logger.warning("[操作] 检查两融融券余量 | [来源] 交易所流式融券总线接口 | [结果] 接口返回空表 | [意义] 触发零做空额度就地降级，确保组合绝对纯多头合规性")
            if audit_logger:
                audit_logger.log_event("DATA_MISSING_BORROW_FALLBACK", {"source": "stock_borrow_analysis", "trade_date": date_str, "reason": "empty_dataframe"})
            return set()

        df = df[df['融券余量'] > 0]
        codes = df['代码'].astype(str).tolist()
        result = set()
        
        for c in codes:
            if len(c) != 6: continue
            result.add(f"{c}.SH" if c.startswith('6') else f"{c}.SZ")
            
        logger.info("[OP] Parse Query Borrowable Array | [SOURCE] AkShare Live Short Analytics Pool | [RESULT] Ingested %s active shortable tokens | [SIGNIFICANCE] Delivers authorized hedge vectors for contemporary risk cross-sections", len(result))
        logger.info("[操作] 解析可融券标的阵列 | [来源] AkShare 实时融券分析池 | [结果] 提取到 %s 只合法可做空令牌 | [意义] 交付当前交易横截面内经官方核准的对冲成分股白名单")
        return result

    except ImportError:
        logger.warning("[OP] Verify Environment Dependencies | [SOURCE] Local Python Library Environment | [RESULT] akshare missing; fallback activated | [SIGNIFICANCE] Forced zero-short space constraint pass")
        logger.warning("[操作] 校验系统底层依赖 | [来源] 本地 Python 环境运行时 | [结果] 未检测到 akshare 库；激活降级程序 | [意义] 强制将做空池置空，维持多头现货流水线不中断")
        if audit_logger: audit_logger.log_event("DATA_MISSING_BORROW_FALLBACK", {"source": "akshare", "reason": "import_error"})
        return set()
    except Exception as e:
        logger.warning(f"[OP] Fetch Live Short Interest Exception | [SOURCE] AkShare Remote Protocol Gateway | [RESULT] Error token: {e} | [SIGNIFICANCE] Defensive routing shuts down short channel completely")
        logger.warning(f"[操作] 获取线上做空筹码异常 | [来源] AkShare 远程协议网关 | [结果] 异常标识: {e} | [意义] 保护性自愈防御程序强行关闭做空通道，做空池安全清零")
        if audit_logger: audit_logger.log_event("DATA_MISSING_BORROW_FALLBACK", {"source": "stock_borrow_analysis", "reason": str(e)})
        return set()