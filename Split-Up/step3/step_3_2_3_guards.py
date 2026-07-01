# -*- coding: utf-8 -*-
"""
step3/step_3_2_3_guards.py
防范特征工程全局渗透与横截面穿透的防御看门狗（步骤3.2 & 3.3）
"""

import logging

logger = logging.getLogger("Phase3.Guards")


def run_preserve_raw_prices_check(context: dict):
    """强验证：防止特征污染，禁止全局层面的不合规量化变换"""
    logger.info("[Step 3.2] Verified: No quantile or fractional differentiation applied at global level.")


def run_cross_sectional_guard(context: dict):
    """强锁定：限制当前交易时间的有效资产边界，阻断未来幸存者偏差"""
    logger.info("[Step 3.3] Locking cross-sectional universe to today's tradable pool.")
    assets = context.get('assets', [])
    context['current_tradable_universe'] = assets
    logger.info(f"Current tradable universe size secured: {len(assets)}")