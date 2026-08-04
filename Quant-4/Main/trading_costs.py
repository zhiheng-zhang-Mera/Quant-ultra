"""Shared, configurable transaction-friction model for optimization and execution."""
from __future__ import annotations
from typing import Mapping

DEFAULT_COST_CONFIG = {
    "commission_rate": 0.00025,
    "minimum_commission": 5.0,
    "exchange_fee_rate": 0.0000341,
    "regulatory_fee_rate": 0.00002,
    "stamp_tax": 0.0005,
    "slippage_rate": 0.0002,
    "etf_annual_management_fee": 0.005,
}

def is_etf(symbol: str = "", asset_type: str = "") -> bool:
    kind = str(asset_type).upper()
    code = str(symbol).split(".")[0]
    return kind in {"ETF", "FUND", "基金"} or code.startswith(("15", "16", "50", "51", "56", "58"))

def merged_cost_config(config: Mapping | None = None) -> dict:
    result = DEFAULT_COST_CONFIG.copy()
    if config:
        result.update({key: value for key, value in config.items() if value is not None})
        if "handling_fee" in config and "exchange_fee_rate" not in config:
            result["exchange_fee_rate"] = config["handling_fee"]
        if "management_fee" in config and "regulatory_fee_rate" not in config:
            result["regulatory_fee_rate"] = config["management_fee"]
        if "slippage_bps" in config and "slippage_rate" not in config:
            result["slippage_rate"] = config["slippage_bps"]
    return result

def explicit_order_fees(notional: float, side: str, symbol: str = "", asset_type: str = "", config: Mapping | None = None) -> dict:
    cfg = merged_cost_config(config)
    notional = max(float(notional), 0.0)
    if notional == 0:
        return {"commission": 0.0, "exchange_fee": 0.0, "regulatory_fee": 0.0, "stamp_tax": 0.0, "total": 0.0}
    commission = max(notional * float(cfg["commission_rate"]), float(cfg["minimum_commission"]))
    exchange_fee = notional * float(cfg["exchange_fee_rate"])
    regulatory_fee = notional * float(cfg["regulatory_fee_rate"])
    stamp = notional * float(cfg["stamp_tax"]) if side.lower() == "sell" and not is_etf(symbol, asset_type) else 0.0
    total = commission + exchange_fee + regulatory_fee + stamp
    return {"commission": commission, "exchange_fee": exchange_fee, "regulatory_fee": regulatory_fee, "stamp_tax": stamp, "total": total}

def round_trip_friction_rate(notional: float, symbol: str = "", asset_type: str = "", holding_days: int = 20, config: Mapping | None = None) -> float:
    cfg = merged_cost_config(config)
    base = max(float(notional), 1.0)
    buy = explicit_order_fees(base, "buy", symbol, asset_type, cfg)["total"]
    sell = explicit_order_fees(base, "sell", symbol, asset_type, cfg)["total"]
    slippage = 2.0 * float(cfg["slippage_rate"]) * base
    management = base * float(cfg["etf_annual_management_fee"]) * max(int(holding_days), 0) / 365.0 if is_etf(symbol, asset_type) else 0.0
    return float((buy + sell + slippage + management) / base)
