"""Score the real 38-name news corpus (12,066 items) with the DeepSeek API.

Round-19 priority 1: run LLM sentiment over the REAL news corpus to test the
8-13 hypothesis that real text is where an LLM differentiates from the
dictionary (lexical) scorer. The local-GPU path is hardware-blocked (R17);
the DeepSeek fallback (R19-previous commit) removes that block.

Batched, resumable, fail-closed:
  - one chat-completions call per batch of BATCH records (default 30),
  - each record's score is appended to the output JSONL immediately, so a
    killed run resumes from the checkpoint (completed ids are re-read),
  - retries each batch 3x with backoff, then logs and continues (a second
    sweep can re-score the failed ids),
  - temperature 0, response_format json_object (same contract as the
    production DeepSeekClient.generate).

Usage:
  python reports/_iter/_llm_deepseek_score_real.py [--limit N] [--batch B]
        [--out NAME] [--min-lex] [--max-lex]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Phase_3.alternative_data import DeepSeekClient, score_text  # noqa: E402

NEWS_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "sina_news_real_38.jsonl"
OUT_DIR = PROJECT_ROOT / "reports" / "_iter" / "llm_deepseek"
MAX_CHARS = 200
RETRIES = 3
BACKOFF = (2.0, 5.0, 10.0)


def load_corpus(limit: int | None = None, corpus: Path | None = None) -> list[dict]:
    records = []
    path = corpus or NEWS_PATH
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    # most-recent-first so window-critical records are scored first
    records.sort(key=lambda r: r["published_at"], reverse=True)
    if limit:
        records = records[:limit]
    return records


def build_prompt(batch: list[dict]) -> str:
    payload = [{"id": r["id"], "symbol": r["symbol"], "text": r["text"],
                "lexical_score": r["lex"]} for r in batch]
    return (
        "Analyze financial sentiment. Return JSON object with key results, an "
        "array of objects: id, score (-1 to 1), confidence (0 to 1). "
        "lexical_score is a dictionary baseline for reference; your score may "
        "agree or disagree. No prose.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def parse_scores(raw: str) -> dict[str, float]:
    parsed = json.loads(raw)
    return {item["id"]: float(np.clip(item["score"], -1, 1))
            for item in parsed.get("results", []) if "id" in item and "score" in item}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--out", default="llm_scores_real_38.jsonl")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="contract-format JSONL to score (default: the 38-name Sina news corpus)")
    parser.add_argument("--min-lex", type=float, default=None,
                        help="only score records with lexical score >= this")
    parser.add_argument("--max-lex", type=float, default=None,
                        help="only score records with lexical score <= this")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("FATAL: DEEPSEEK_API_KEY not set in this process env", file=sys.stderr)
        return 2
    client = DeepSeekClient(timeout=180.0)
    print(f"provider: {client.model} @ {client.base_url}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.out

    records = load_corpus(args.limit, args.corpus)
    print(f"corpus loaded: {len(records)} records", flush=True)

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
        print(f"resume: {len(done)} already scored", flush=True)

    rows = []
    for rec in records:
        lex = float(score_text(rec["text"]))
        if args.min_lex is not None and lex < args.min_lex:
            continue
        if args.max_lex is not None and lex > args.max_lex:
            continue
        rows.append({**rec, "id": rec["record_id"], "lex": lex,
                     "text": str(rec["text"])[:MAX_CHARS]})
    print(f"after filter: {len(rows)} rows to score", flush=True)

    todo = [r for r in rows if r["id"] not in done]
    print(f"pending: {len(todo)} (already done {len(rows) - len(todo)})", flush=True)

    t0 = time.time()
    n_ok = n_fail = n_batches = 0
    fail_ids: list[str] = []
    batch = []
    for i, row in enumerate(todo):
        batch.append(row)
        if len(batch) < args.batch and i + 1 < len(todo):
            continue
        n_batches += 1
        prompt = build_prompt(batch)
        raw = None
        for attempt in range(RETRIES):
            try:
                raw = client.generate(client.model, prompt)
                scores = parse_scores(raw)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < RETRIES - 1:
                    wait = BACKOFF[attempt]
                    print(f"  batch {n_batches} attempt {attempt + 1} failed "
                          f"({type(exc).__name__}), retry in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  batch {n_batches} FAILED after {RETRIES} tries: "
                          f"{type(exc).__name__}", flush=True)
                    raw = None
        if raw is None:
            n_fail += len(batch)
            fail_ids.extend(r["id"] for r in batch)
        else:
            missing = [r["id"] for r in batch if r["id"] not in scores]
            if missing:
                print(f"  batch {n_batches}: {len(missing)}/{len(batch)} ids missing "
                      f"from response", flush=True)
            with out_path.open("a", encoding="utf-8") as handle:
                for r in batch:
                    score = scores.get(r["id"])
                    if score is None:
                        continue
                    handle.write(json.dumps({
                        "id": r["id"], "record_id": r["record_id"],
                        "symbol": r["symbol"], "published_at": r["published_at"],
                        "text": r["text"], "lex": r["lex"], "llm": score,
                    }, ensure_ascii=False) + "\n")
            n_ok += sum(1 for r in batch if r["id"] in scores)
        if n_batches % 10 == 0:
            elapsed = time.time() - t0
            print(f"  progress: {n_batches} batches, {n_ok} scored, {n_fail} failed, "
                  f"{elapsed:.0f}s elapsed", flush=True)
        batch = []
        time.sleep(0.4)  # gentle pacing

    summary = {
        "corpus": str(NEWS_PATH), "provider": client.model,
        "requested": len(rows), "scored": n_ok, "failed": n_fail,
        "batches": n_batches, "wall_seconds": round(time.time() - t0, 1),
        "failed_ids_sample": fail_ids[:20],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
