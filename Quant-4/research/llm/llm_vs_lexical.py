"""Round-19 Priority 1: LLM vs lexical sentiment on the REAL 38-name corpus.

Loads the DeepSeek-scored records (llm_scores_real_38.jsonl) and compares
against the dictionary scorer (score_text): correlation, sign disagreement,
strong-signal rates, distributions, per-symbol breakdown. This is the 8-13
hypothesis test - real news text is where an LLM should differentiate from
a keyword dictionary.

Outputs:
  reports/_iter/llm_deepseek/llm_vs_lex_real_38_deepseek.json
  reports/_iter/llm_deepseek/llm_vs_lex_real_38_deepseek.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "reports" / "_iter" / "llm_deepseek"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path,
                        default=OUT_DIR / "llm_scores_real_38.jsonl")
    parser.add_argument("--out-tag", default="real_38")
    parser.add_argument("--corpus-total", type=int, default=12066)
    args = parser.parse_args()

    scores_path = args.scores
    if not scores_path.exists():
        print(f"FATAL: {scores_path} not found - run _llm_deepseek_score_real.py first", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in scores_path.open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    print(f"scored records: {len(df)} | symbols: {df['symbol'].nunique()} | "
          f"coverage of corpus: {len(df)}/{args.corpus_total}")

    valid = df.dropna(subset=["lex", "llm"])
    n = len(valid)
    corr = valid["lex"].corr(valid["llm"])
    spearman = valid["lex"].corr(valid["llm"], method="spearman")
    sign_disagree = ((valid["lex"] > 0) != (valid["llm"] > 0)).mean()
    # strong-signal rates (magnitude >= 0.5)
    strong_lex = (valid["lex"].abs() >= 0.5).mean()
    strong_llm = (valid["llm"].abs() >= 0.5).mean()
    neutral_lex = (valid["lex"] == 0).mean()
    neutral_llm = (valid["llm"].abs() < 0.05).mean()
    mean_abs_diff = (valid["lex"] - valid["llm"]).abs().mean()
    lex_std, llm_std = valid["lex"].std(), valid["llm"].std()
    lex_mean, llm_mean = valid["lex"].mean(), valid["llm"].mean()

    def _corr(s: pd.Series) -> float:
        other = valid.loc[s.index, "llm"]
        if s.nunique() < 2 or other.nunique() < 2:
            return float("nan")
        return float(s.corr(other))

    per_symbol = valid.groupby("symbol").agg(
        n=("llm", "size"),
        lex_mean=("lex", "mean"), llm_mean=("llm", "mean"),
        lex_std=("lex", "std"), llm_std=("llm", "std"),
        corr=("lex", _corr),
        sign_disagree=("lex", lambda s: ((s > 0) != (valid.loc[s.index, "llm"] > 0)).mean()),
    ).round(4).sort_values("n", ascending=False)

    # distribution buckets
    def bucket(s: pd.Series) -> list[int]:
        return [int(((s > 0.5) & (s <= 1.0)).sum()), int(((s > 0.05) & (s <= 0.5)).sum()),
                int(((s >= -0.05) & (s <= 0.05)).sum()), int(((s >= -0.5) & (s < -0.05)).sum()),
                int(((s >= -1.0) & (s < -0.5)).sum())]

    summary = {
        "n": n, "corpus_total": args.corpus_total,
        "pearson_corr": round(corr, 4), "spearman_corr": round(spearman, 4),
        "sign_disagreement": round(float(sign_disagree), 4),
        "strong_lex_ge_0.5": round(float(strong_lex), 4),
        "strong_llm_ge_0.5": round(float(strong_llm), 4),
        "neutral_lex": round(float(neutral_lex), 4),
        "neutral_llm_abs_lt_0.05": round(float(neutral_llm), 4),
        "mean_abs_diff": round(float(mean_abs_diff), 4),
        "lex_mean_std": [round(float(lex_mean), 4), round(float(lex_std), 4)],
        "llm_mean_std": [round(float(llm_mean), 4), round(float(llm_std), 4)],
        "lex_buckets_pos_strong/pos/neutral/neg/neg_strong": bucket(valid["lex"]),
        "llm_buckets_pos_strong/pos/neutral/neg/neg_strong": bucket(valid["llm"]),
        "per_symbol_top": per_symbol.head(38).to_dict(orient="index"),
    }
    out = OUT_DIR / f"llm_vs_lex_{args.out_tag}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    valid[["record_id", "symbol", "published_at", "text", "lex", "llm"]].to_csv(
        OUT_DIR / f"llm_vs_lex_{args.out_tag}.csv", index=False, encoding="utf-8-sig")

    print(f"n={n} | pearson={corr:.3f} spearman={spearman:.3f} | sign_disagree={sign_disagree:.1%}")
    print(f"strong(>=0.5): lex {strong_lex:.0%} llm {strong_llm:.0%} | neutral: lex {neutral_lex:.0%} llm {neutral_llm:.0%}")
    print(f"mean|lex-llm|={mean_abs_diff:.3f} | lex {lex_mean:+.3f}/{lex_std:.3f} vs llm {llm_mean:+.3f}/{llm_std:.3f}")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
