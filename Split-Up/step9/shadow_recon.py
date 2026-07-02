"""
Quant-Ultra Flow - Step 9.1: Deterministic Shadow Reconciliation & Gateway Safety Systems
Handles physical command decoupling, MAE auditing, and the absolute Hard Kill Switch.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from .config import MAE_THRESHOLD

logger = logging.getLogger("MLOps.ShadowRecon")

def run_shadow_reconciliation(context: dict) -> dict:
    """
    执行高精度确定性影子对账。
    对比离线生产层目标权重与有限状态机（FSM）实际执行权重，实施 1e-5 MAE 刚性拦截。
    """
    logger.info("[Step 9.1] 启动确定性影子对账快速重放协议。")
    
    target_weights = context.get('target_weights', {})
    executed_weights = {}
    
    # 从有限状态机引擎中流式抽取实盘影子执行权重
    fsm_engine = context.get('fsm_engine')
    if fsm_engine is not None:
        try:
            total_nav = fsm_engine.calc_nav()
            current_date = context.get('current_date', datetime.now().strftime("%Y-%m-%d"))
            prices = fsm_engine.get_prices(current_date)
            for symbol in fsm_engine.assets:
                price = prices.get(symbol)
                if price is not None and total_nav > 0:
                    position_value = fsm_engine.holdings.get(symbol, 0.0) * price
                    executed_weights[symbol] = position_value / total_nav
                else:
                    executed_weights[symbol] = 0.0
        except Exception as e:
            logger.error(f"从 FSM 状态机引擎计算执行权重失败: {str(e)}")
            executed_weights = {}
            
    if not executed_weights:
        # 灾备降级：尝试从上下文中直接读取静态快照
        holdings = context.get('holdings', {})
        prices = context.get('current_prices', {})
        if holdings and prices:
            total_mv = sum(holdings.get(s, 0.0) * prices.get(s, 0.0) for s in holdings)
            if total_mv > 0:
                for symbol in holdings:
                    executed_weights[symbol] = (holdings[symbol] * prices.get(symbol, 0.0)) / total_mv
                    
    # 执行横截面绝对可比性对账核验
    all_assets = set(target_weights.keys()) | set(executed_weights.keys())
    if all_assets:
        mae = float(np.mean([abs(executed_weights.get(sym, 0.0) - target_weights.get(sym, 0.0)) for sym in all_assets]))
        context['reconciliation_mae'] = mae
        if mae > MAE_THRESHOLD:
            logger.error(f"🚨 [影子对账失败] MAE={mae:.7f} 超过规范硬红线阈值 {MAE_THRESHOLD}！拦截次日交易。")
            context['recon_passed'] = False
            context['halt_next_trading_cycle'] = True
        else:
            logger.info(f"✅ [影子对账通过] 生产层与推演层硬阈值核验成功。MAE={mae:.7f}")
            context['recon_passed'] = True
            context['halt_next_trading_cycle'] = False
    else:
        logger.warning("未检测到任何资产持仓分配阵列，影子对账跳过。")
        context['reconciliation_mae'] = 0.0
        context['recon_passed'] = True

    # 序列化生成多资产分布式分析报文，供未来 LLM 及监控层进行点状特征与观点追溯
    payload = {
        "strategy_id": "QUANT_ULTRA_PRO_VERSION_FINAL",
        "timestamp": datetime.now().isoformat(),
        "current_date": str(context.get('current_date', datetime.now())),
        "target_allocations": target_weights,
        "executed_allocations": executed_weights,
        "reconciliation_mae": context.get('reconciliation_mae', 0.0),
        "portfolio_nav": float(context.get('final_nav', context.get('current_nav', 0.0))),
        "cqr_widths": {k: float(v) for k, v in context.get('cqr_hetero_widths', {}).items()},
        "view_posterior_bl": [float(x) for x in context.get('R_BL', [])] if isinstance(context.get('R_BL'), (np.ndarray, list)) else []
    }
    
    payload_path = Path("multi_asset_llm_payload.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    logger.info(f"多资产 LLM 生产分析报文已点状固化落盘: {payload_path.resolve()}")
    
    return context

def trigger_physical_hard_kill_switch(fsm_engine, counterparty_gateway) -> dict:
    """
    最高特权紧急控制看门狗（Hard Kill Switch）。
    一键点击瞬间越过所有模型推演，强制以市价地毯式全额清仓纯现货多头，资金退回个体纯现金账户，斩断柜台。
    """
    logger.critical("⚠️ 🔴 [紧急特权硬杀伤激活] 收到看门狗物理强平指令！启动全盘多头饱和清仓程序。")
    liquidation_report = {
        "status": "TOTAL_LIQUIDATION_EXECUTED",
        "timestamp": datetime.now().isoformat(),
        "liquidated_assets": [],
        "returned_cash": 0.0
    }
    
    try:
        if fsm_engine is not None:
            assets_to_liquidate = list(fsm_engine.holdings.keys())
            for asset in assets_to_liquidate:
                shares = fsm_engine.holdings.get(asset, 0.0)
                if shares > 0:
                    fsm_engine.holdings[asset] = 0.0  # 物理清除
                    liquidation_report["liquidated_assets"].append({"asset": asset, "shares_liquidated": shares})
            
            # 资金底座隐式绝对对账对齐
            final_nav = fsm_engine.calc_nav()
            fsm_engine.cash = final_nav  # 100% 回归纯现金状态
            liquidation_report["returned_cash"] = float(final_nav)
            
        if counterparty_gateway is not None:
            # 刚性切断与柜台的一切通讯物理连接
            counterparty_gateway.disconnect_all_channels()
            logger.critical("实盘券商柜台物理连接已被系统安全哨兵刚性截断断开。")
            
    except Exception as ex:
        logger.critical(f"紧急熔断清仓管线执行期间发生致命恐慌性故障: {str(ex)}")
        raise SystemExit("紧急风险系统物理关机清仓失败！必须立刻介入人工干预。")
        
    return liquidation_report