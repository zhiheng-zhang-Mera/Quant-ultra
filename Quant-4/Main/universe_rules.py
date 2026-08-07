"""Shared universe eligibility rules used by the orchestrator and backtests.

Kept in a dedicated module so the pipeline orchestrator never imports from a
backtest driver, and so eligibility logic has one auditable home.
"""
from __future__ import annotations

DEFAULT_EXCLUDED_PREFIXES = ("200", "300", "301", "4", "8", "92", "688", "689", "900")


def symbol_purchase_eligibility(symbol: str, excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES) -> tuple[bool, str]:
    """Return (eligible, reason) for a research-only purchase filter.

    ETFs and common A-share codes are eligible; excluded account-permission
    prefixes and unsupported boards are rejected with a stable reason code.
    """
    code, dot, exchange = str(symbol).upper().partition(".")
    if not dot or exchange not in {"SH", "SZ"} or len(code) != 6 or not code.isdigit():
        return False, "NOT_STANDARD_A_SHARE"
    is_etf = code.startswith(("15", "16", "50", "51", "56", "58"))
    if not is_etf and any(code.startswith(prefix) for prefix in excluded_prefixes):
        return False, "EXCLUDED_ACCOUNT_PERMISSION_PREFIX"
    if not code.startswith(("0", "6")) and not is_etf:
        return False, "UNSUPPORTED_PURCHASE_CODE"
    return True, "ELIGIBLE_CODE"


def asset_type_for_symbol(symbol: str) -> str:
    """Classify a symbol as ETF or stock using the same prefix rules."""
    return "ETF" if symbol.split(".")[0].startswith(("15", "16", "50", "51", "56", "58")) else "stock"
