# -*- coding: utf-8 -*-
"""
step4/borrow_manager.py
融券可用性合规审计与降级机制
"""

import logging
import pandas as pd
from typing import Set

logger = logging.getLogger("LabelingWeighting.Borrow")

def fetch_borrowable_stocks(context: dict, trade_date: pd.Timestamp) -> Set[str]:
    """
    从 AkShare 获取指定交易日有融券余量的股票代码集合。
    若接口失败或数据缺失，按规范降级为"全部不可用"，并记录审计事件。
    """
    audit_logger = context.get("audit_logger")
    try:
        import akshare as ak
        date_str = trade_date.strftime("%Y%m%d")
        df = ak.stock_borrow_analysis(date=date_str)
        
        if df.empty:
            logger.warning(f"融券数据为空（日期: {date_str}），降级为全部不可用")
            if audit_logger:
                audit_logger.log_event("数据缺失-默认残值", {
                    "source": "stock_borrow_analysis",
                    "trade_date": date_str,
                    "reason": "返回空DataFrame"
                })
            return set()

        df = df[df['融券余量'] > 0]
        codes = df['代码'].astype(str).tolist()
        result = set()
        
        for c in codes:
            if len(c) != 6:
                continue
            if c.startswith('6'):
                result.add(f"{c}.SH")
            elif c.startswith('0') or c.startswith('3'):
                result.add(f"{c}.SZ")
                
        logger.info(f"获取融券可用标的: {len(result)} 只（日期: {date_str}）")
        return result

    except ImportError:
        logger.warning("AkShare 未安装，无法获取融券数据，降级为全部不可用")
        if audit_logger:
            audit_logger.log_event("数据缺失-默认残值", {
                "source": "akshare",
                "reason": "import_error"
            })
        return set()
    except Exception as e:
        logger.warning(f"融券数据查询异常: {e}，降级为全部不可用")
        if audit_logger:
            audit_logger.log_event("数据缺失-默认残值", {
                "source": "stock_borrow_analysis",
                "reason": str(e)
            })
        return set()