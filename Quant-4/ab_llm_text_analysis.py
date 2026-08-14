# -*- coding: utf-8 -*-
"""8-13 A/B runner: baseline vs lexical vs LLM alternative-signal pathway.

Uses the production PIT module weights: alternative_signal =
0.35*news_sentiment + 0.25*forum_sentiment + 0.40*pool_change_5v20.
Text corpus is synthetic (price-derived, PIT, no future info). LLM variant
uses a bounded real-Ollama sample; lexical elsewhere.

Signal lookup is grid-tolerant: for a signal date, the most recent corpus
row with published_at <= signal date within tolerance_days is used, so
sub-window runs do not silently lose coverage to calendar drift.
"""
import sys, json, time, bisect as _bisect
from pathlib import Path
from datetime import date as _date
import numpy as np
import pandas as pd

sys.path.insert(0, "Quant-4")
from Main.advice_portfolio_backtest import discover_real_universe, run_advice_portfolio_backtest

OUT = Path("Quant-4/reports/llm_8-13_ab")
cache = Path("Quant-4/Data_Cache")

def load_maps(name):
    return {tuple(k.split("|")): float(v) for k, v in json.load(open(OUT / f"{name}.json", encoding="utf-8")).items()}

def make_signal_fn(lex_news, lex_forum, pool, llm=None, tolerance_days=14):
    """Return a PIT alternative_signal function on (signal_date, symbol).

    Each map is keyed by (iso_date, symbol); values are indexed per symbol
    and looked up with a bisect so the most recent row published on or
    before the signal date (within tolerance_days) is used. This keeps the
    signal free of future information and robust to grid drift.
    """
    def _index(mapping):
        index = {}
        for (day, sym), value in mapping.items():
            index.setdefault(sym, []).append((day, value))
        for sym in index:
            index[sym].sort(key=lambda item: item[0])
            index[sym] = ([item[0] for item in index[sym]], [item[1] for item in index[sym]])
        return index

    def _lookup(index, signal_date, symbol):
        rows = index.get(symbol)
        if not rows:
            return 0.0
        days, values = rows
        probe = signal_date.date().isoformat()
        pos = _bisect.bisect_right(days, probe) - 1
        if pos < 0:
            return 0.0
        gap = (_date.fromisoformat(probe) - _date.fromisoformat(days[pos])).days
        return float(values[pos]) if gap <= tolerance_days else 0.0

    news_idx = _index(lex_news); forum_idx = _index(lex_forum); pool_idx = _index(pool)
    llm_idx = _index(llm) if llm else None

    def fn(signal_date, symbol):
        n = _lookup(llm_idx, signal_date, symbol) if llm_idx else None
        if n is None:
            n = _lookup(news_idx, signal_date, symbol)
        f = _lookup(forum_idx, signal_date, symbol)
        p = float(np.clip(_lookup(pool_idx, signal_date, symbol), -1, 1))
        return float(np.clip(0.35 * n + 0.25 * f + 0.40 * p, -1, 1))
    return fn

def oos_metrics(returns):
    oos = returns.loc[returns.index >= "2022-01-01"]
    r = oos["strategy_return"]
    eq = (1 + r).cumprod(); n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if vol else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    calmar = ann / abs(mdd) if mdd else 0.0
    return {"oos_annual_return": ann, "oos_sharpe": sharpe, "oos_max_drawdown": mdd, "oos_calmar": calmar, "oos_observations": n}

def run_variant(name, signal_fn, rank_weight=0.0):
    frames, decisions = discover_real_universe(cache, min_rows=504)
    t0 = time.time()
    res = run_advice_portfolio_backtest(
        frames, decisions, years=8, lookback=252, rebalance_every=5, fee_rate=0.001,
        min_assets=2, max_positions=5, min_exposure=0.15, max_exposure=0.90,
        max_holding_days=20, harvest_cooldown_days=3, min_trade_weight=0.02,
        max_daily_turnover=0.25, alternative_signal_fn=signal_fn,
        alternative_signal_rank_weight=rank_weight)
    s = dict(res["summary"]); s.update(oos_metrics(res["returns"]))
    r = res["returns"]["strategy_return"]; eq = (1 + r).cumprod()
    s["calmar"] = float(eq.iloc[-1] ** (252 / len(r)) - 1) / abs(float((eq / eq.cummax() - 1).min())) if (eq / eq.cummax() - 1).min() < 0 else 0.0
    s["variant"] = name; s["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(s, open(OUT / f"summary_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    res["returns"].to_csv(OUT / f"returns_{name}.csv", encoding="utf-8-sig")
    res["signals"].to_csv(OUT / f"signals_{name}.csv", index=False, encoding="utf-8-sig")
    print(name, "ann", round(s["annual_return"], 4), "sharpe", round(s["sharpe"], 3),
          "calmar", round(s.get("calmar", 0), 3), "mdd", round(s["max_drawdown"], 4),
          "oos_ann", round(s["oos_annual_return"], 4), "oos_sharpe", round(s["oos_sharpe"], 3),
          "secs", s["wall_seconds"], flush=True)
    return s

if __name__ == "__main__":
    import sys as _s
    variant = _s.argv[1] if len(_s.argv) > 1 else "baseline"
    rank_weight = float(_s.argv[2]) if len(_s.argv) > 2 else 0.0
    lex_news = load_maps("lex_news"); lex_forum = load_maps("lex_forum"); pool = load_maps("pool_change")
    rw_tag = "" if rank_weight == 0.0 else f"_rank{rank_weight:g}"
    if variant == "baseline":
        s = run_variant(f"baseline{rw_tag}", None, rank_weight)
    elif variant == "lexical":
        s = run_variant(f"lexical{rw_tag}", make_signal_fn(lex_news, lex_forum, pool), rank_weight)
    elif variant == "llm":
        llm = json.load(open(OUT / "llm_news.json", encoding="utf-8")) if (OUT / "llm_news.json").exists() else {}
        llm = {tuple(k.split("|")): float(v) for k, v in llm.items()}
        print("llm coverage", len(llm))
        s = run_variant("llm", make_signal_fn(lex_news, lex_forum, pool, llm), rank_weight)
    else:
        raise SystemExit(f"unknown variant {variant}")
    print(json.dumps({k: s[k] for k in ("variant", "annual_return", "sharpe", "calmar", "max_drawdown", "oos_annual_return", "oos_sharpe", "oos_calmar", "oos_max_drawdown", "wall_seconds")}, ensure_ascii=False))
