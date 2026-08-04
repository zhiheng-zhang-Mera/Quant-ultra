# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 9.1: Deterministic Shadow Reconciliation & Emergency Hard Kill Switch
"""
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pandas as pd
from Phase_9.config import DEFAULT_MLOPS_CONFIG

logger = logging.getLogger("MLOps.ShadowRecon")

def run_shadow_reconciliation(context: dict) -> dict:
    """
    高精度影子对账快速重放协议。
    """
    # logger.info("[OP] Initiate Shadow Reconciliation Pipeline | [SOURCE] Master Portfolio Target weights | [RESULT] Activating bi-track routing | [SIGNIFICANCE] Validates physical execution matching to prevent accounting leaks")
    logger.info("[操作] 启动确定性影子对账管线 | [来源] 主控制流目标分配权重 | [结果] 正在激活双轨模态分流 | [意义] 校对模型目标与物理成交之间的一致性，防止隐性滑点资产流失")

    target_weights = context.get('target_weights', {})
    if not target_weights:
        daily_weights = context.get('daily_weights')
        if isinstance(daily_weights, pd.DataFrame) and not daily_weights.empty:
            target_weights = daily_weights.iloc[-1].dropna().astype(float).to_dict()
            context['target_weights'] = target_weights
    if not target_weights:
        daily_weights = context.get('daily_weights')
        if isinstance(daily_weights, pd.DataFrame) and not daily_weights.empty:
            target_weights = daily_weights.iloc[-1].dropna().astype(float).to_dict()
            context['target_weights'] = target_weights
    executed_weights = {}
    is_live = context.get('is_live', False)
    mae_ceiling = context.get('config', {}).get('reconciliation_mae_ceiling', DEFAULT_MLOPS_CONFIG['reconciliation_mae_ceiling'])

    if not target_weights:
        context['reconciliation_mae'] = float('inf')
        context['recon_passed'] = False
        context['reconciliation_evidence_status'] = 'MISSING_TARGET_WEIGHTS'
        logger.critical("Reconciliation rejected because no target portfolio evidence was available.")
        return context

    if not target_weights:
        context['reconciliation_mae'] = float('inf')
        context['recon_passed'] = False
        context['reconciliation_evidence_status'] = 'MISSING_TARGET_WEIGHTS'
        logger.critical("Reconciliation rejected because no target portfolio evidence was available.")
        return context

    if is_live:
        gateway = context.get('counterparty_gateway')
        if gateway is None:
            # logger.critical("[OP] Extract Broker Gateway Interface | [SOURCE] Live Trade Routing Tunnel | [RESULT] Connection Failed: Gateway Vacuum | [SIGNIFICANCE] Directs account freeze to avoid executing under dark state")
            logger.critical("[操作] 提取券商柜台交互网关 | [来源] 实盘柜台网络通路 | [结果] 网关连接对象为空真空 | [意义] 检测到实盘连接断裂异常，强制冻结系统，防止盲目下单")
            raise RuntimeError("MLOps Catastrophe: Physical execution gateway absent under live mode.")
        
        try:
            live_portfolio = gateway.get_portfolio_snapshot()
            total_equity = float(live_portfolio.get('total_equity', 1.0))
            for asset, mv in live_portfolio.get('holdings_value', {}).items():
                executed_weights[asset] = float(mv) / total_equity
        except Exception as e:
            logger.error(f"Failed to fetch real-time broker holdings: {e}")
            executed_weights = {k: 0.0 for k in target_weights.keys()}
    else:
        # 回测态：无缝对比 Phase_7 FSM 实际模拟持仓，完美修复类型覆盖 Bug
        fsm_engine = context.get('fsm_engine')
        if fsm_engine is not None:
            total_val = fsm_engine.calc_nav()
            for asset in context.get('assets', []):
                p = fsm_engine.bus.query_by_pit(asset, fsm_engine.current_date, "total_return_price") or 0.0
                executed_val = fsm_engine.holdings.get(asset, 0.0) * p
                executed_weights[asset] = float(executed_val / (total_val if total_val > 0 else 1.0))
        else:
            executed_weights = {k: 0.0 for k in target_weights.keys()}

    # 计算平均绝对误差 MAE
    all_keys = set(target_weights.keys()) | set(executed_weights.keys())
    diffs = [abs(target_weights.get(k, 0.0) - executed_weights.get(k, 0.0)) for k in all_keys]
    mae = float(np.mean(diffs)) if diffs else 0.0

    context['reconciliation_mae'] = mae
    recon_passed = mae <= mae_ceiling
    context['recon_passed'] = recon_passed
    context['reconciliation_evidence_status'] = 'VERIFIED' if recon_passed else 'POSITION_MISMATCH'
    context['reconciliation_evidence_status'] = 'VERIFIED' if recon_passed else 'POSITION_MISMATCH'

    # logger.info("[OP] Run Deterministic MAE Calculation | [SOURCE] Comparative Asset Portfolios | [RESULT] Current MAE: %.8f, Limits Ceiling: %.8f, Passed: %s | [SIGNIFICANCE] Determines whether execution drift triggers trade hold locks", mae, mae_ceiling, recon_passed)
    logger.info("[操作] 计算确定性持仓 MAE 偏离度 | [来源] 理论与执行双端持仓明细对账 | [结果] 截面 MAE 误差: %.8f, 允许限额: %.8f, 对账通过: %s | [意义] 以高精度数学均值衡量实盘调仓损耗，不平账则立刻对系统下单实施封锁锁死", mae, mae_ceiling, recon_passed)
    return context

def trigger_physical_hard_kill_switch(fsm_engine, counterparty_gateway) -> dict:
    """紧急特权一键清仓看门狗程序"""
    # logger.critical("[OP] Trigger Emergency Hard Kill Switch | [SOURCE] Telemetry System Command Desk | [RESULT] Executing force liquidation | [SIGNIFICANCE] Bypasses models to instantly liquidate positions into cash baseline")
    logger.critical("[操作] 激活最高级紧急硬杀伤开关 | [来源] 生产看门狗应急控制台指令 | [结果] 正在执行全额地毯式强制平仓 | [意义] 强行超越一切优化器推演逻辑，瞬间变现全部资产，最大程度回流现金保障资金本金")
    
    report = {
        "status": "TOTAL_LIQUIDATION_EXECUTED",
        "timestamp": datetime.now().isoformat(),
        "liquidated_assets": [],
        "returned_cash": 0.0
    }
    
    if counterparty_gateway is not None:
        try:
            counterparty_gateway.close_all_market_positions()
            report["liquidated_assets"].append("ALL_LIVE_PORTFOLIO_LIQUIDATED")
        except Exception as e:
            logger.error(f"Live broker liquidation failed: {e}")
            
    if fsm_engine is not None:
        for asset in list(fsm_engine.holdings.keys()):
            shares = fsm_engine.holdings.get(asset, 0.0)
            if shares > 0:
                fsm_engine.holdings[asset] = 0.0
                report["liquidated_assets"].append({"asset": asset, "shares": shares})
        
        final_nav = fsm_engine.calc_nav()
        fsm_engine.cash = final_nav
        report["returned_cash"] = final_nav
        
    return report

def enforce_reconciliation_gate(context: dict) -> dict:
    """Freeze order generation on reconciliation failure; liquidate only in live mode."""
    if context.get("recon_passed") is True:
        context["trading_halted"] = False
        context["kill_switch_report"] = None
        return context

    context["trading_halted"] = True
    if context.get("is_live") is True and context.get("config", {}).get("analysis_only") is not True:
        context["kill_switch_report"] = trigger_physical_hard_kill_switch(
            context.get("fsm_engine"), context.get("counterparty_gateway")
        )
    else:
        context["kill_switch_report"] = {
            "status": "ORDER_GENERATION_FROZEN",
            "reason": "RECONCILIATION_FAILED",
            "liquidation_attempted": False,
        }
    return context
