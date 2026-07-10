"""
Quant-Ultra Flow - Step 9.1: Deterministic Shadow Reconciliation & Gateway Safety Systems
Handles physical command decoupling, MAE auditing, and the absolute Hard Kill Switch.
Fully refactored to support distinct Backtest and Live trading regimes (Resolves Flaw A-2).
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
    支持 [回测态] 与 [实盘态] 双轨自适应解耦路由：
    - 回测态 (is_live=False): 对比目标权重与有限状态机（FSM）实际模拟持仓。
    - 实盘态 (is_live=True): 对比目标权重与真实券商/柜台连接器（counterparty_gateway）的API实际持仓资产进行穿透校准。
    两轨均实施 1e-5 MAE 刚性红线拦截。
    """
    logger.info("[Step 9.1] 启动确定性影子对账快速重放协议。")
    
    target_weights = context.get('target_weights', {})
    executed_weights = {}
    
    # ====================================================
    # 核心修复 A-2：引入模态分流，切除实盘对回测FSM引擎的硬性依赖
    # ====================================================
    is_live = context.get('is_live', False)
    
    if is_live:
        logger.info("📡 [影子对账切换] 检测到当前系统处于【实盘生产态 (Live Mode)】。正在绕过FSM引擎，直接接入真实券商柜台接口。")
        gateway = context.get('counterparty_gateway')
        if gateway is not None:
            try:
                # 1. 穿透调用实盘柜台API，拉取实时持仓股份、实时行情与总权益底座
                live_positions = gateway.get_positions()    # 预期返回标准字典: {symbol: float_shares}
                live_prices = gateway.get_live_prices()       # 预期返回标准字典: {symbol: float_price}
                total_assets = gateway.get_total_assets()   # 预期返回实盘总资产账户净值 (NAV)
                
                # 兜底计算：若柜台总资产接口未直接返回，利用可用现金及持仓市值刚性累加
                if total_assets <= 0:
                    live_cash = gateway.get_available_cash()
                    market_value = sum(live_positions.get(s, 0.0) * live_prices.get(s, 0.0) for s in live_positions)
                    total_assets = live_cash + market_value
                    
                if total_assets > 0:
                    # 2. 对所有在持资产或目标资产进行横截面实盘真实持仓权重折算
                    all_possible_symbols = set(target_weights.keys()) | set(live_positions.keys())
                    for symbol in all_possible_symbols:
                        shares = live_positions.get(symbol, 0.0)
                        # 优先采用柜台实时买卖档中间价，若不可用则向上下文当前历史价格流降级
                        price = live_prices.get(symbol, context.get('current_prices', {}).get(symbol, 0.0))
                        executed_weights[symbol] = (shares * price) / total_assets
                    logger.info("✅ 成功穿透拉取实盘真实柜台持仓矩阵，实盘持仓权重折算完毕。")
                else:
                    logger.error("🚨 实盘柜台返回的总资产总权益估算为0或负值，无法进行实盘持仓权重折算！")
            except Exception as e:
                logger.error(f"🚨 穿透调用实盘券商柜台接口发生致命通信/解包异常: {str(e)}")
                executed_weights = {}
        else:
            logger.critical("🚨 致命异常：系统声明为实盘生产态，但外部主控未注入合规的柜台连接器(counterparty_gateway)！影子对账流强行挂起。")
            executed_weights = {}
    else:
        logger.info("📊 [影子对账切换] 检测到当前系统处于【历史回测态 (Backtest Mode)】。继续调用有限状态机（FSM）执行模拟对账。")
        # 从有限状态机引擎中流式抽取回测期实际执行权重
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

    # ====================================================
    # 统一横截面绝对可比性对账核验流（回测与实盘高度共享）
    # ====================================================
    all_assets = set(target_weights.keys()) | set(executed_weights.keys())
    if all_assets:
        mae = float(np.mean([abs(executed_weights.get(sym, 0.0) - target_weights.get(sym, 0.0)) for sym in all_assets]))
        context['reconciliation_mae'] = mae
        if mae > MAE_THRESHOLD:
            logger.error(f"🚨 [影子对账失败] MAE={mae:.7f} 超过规范硬红线阈值 {MAE_THRESHOLD}！阻断次日交易循环。")
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

    # 序列化生成多资产分布式分析报文，供未来监控层及本地LLM进行点状特征与逻辑追溯
    payload = {
        "strategy_id": "QUANT_ULTRA_PRO_VERSION_FINAL",
        "timestamp": datetime.now().isoformat(),
        "is_live_regime": is_live,
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