# Negative Results Report

| Field | Value |
|---|---|
| Report type | NEGATIVE_RESULT |
| Experiment ID | LEGACY-20260816-ALTERNATIVE-SIGNAL |
| Status | REJECT |
| Code version | git:81c5fe4-legacy-evidence |
| Data version | legacy-real-news-cache |
| Parameter version | alternative-signal-weight/v1 |
| Generated at | 2026-08-19T06:05:00+00:00 |
| Report SHA-256 | 364c4d0d1771448f48ca99be0a088ccadf04442bfae5f654ec8bbf009aa30284 |

## Summary

Real-news signal direction changed with universe size and full-window/OOS deltas were immaterial; additional parameter and risk-control mechanisms also failed their predefined evidence requirements.

## Metrics

| Metric | Value |
|---|---|
| alternative_signal_weight | 0 |
| decision_status | REJECT |
| formal_registry_migration | not retroactive |
| strategy_selector | disabled |

## Evidence

- The root README retains the 2026-08-14 to 2026-08-16 negative-result narrative.
- Quant-4/Main/experiment_registry.py now preserves all new ACCEPT, HOLD and REJECT outcomes.
- Alternative-event outputs remain research-only unless independent evidence passes.

## Limitations

- These experiments predate the formal EXP- registry and are labelled LEGACY rather than backfilled.
- Historical local report files and caches are not published.
- A rejected signal in this sample is not proof it can never work.

## Decision

REJECT production activation; keep alternative_signal_weight=0 and preserve the failed direction/stability evidence.
