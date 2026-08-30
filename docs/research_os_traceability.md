# Research OS implementation traceability — 8-30

This document maps the imported implementation plan to code and tests. It distinguishes implemented foundations from intentionally deferred production-scale integrations.

| Plan area | Implementation | Verification |
|---|---|---|
| W0 kernel stability | Existing Phase 1–11 unchanged; architecture documented | Existing 229-test baseline |
| W1 contracts | `Research_OS/contracts/` | deterministic JSON/hash, frozen spec, validation tests |
| W2 registry/trials | `registry/storage.py`, `experiment_registry.py` | tamper detection, append-only transitions, budget exhaustion |
| W3 memory | `memory/retrieval.py` | negative-result duplicate retrieval, offline deterministic ranking |
| W4 evidence | `evidence/graph.py`, `evidence/data_gate.py` | conflicts, diversity, PIT chronology, missing-state distinction |
| W5 agents/providers | `agents/`, `providers/` | permissions, blinding, malformed output, diversity policy |
| W6–W8 discovery/lock/data | shared contracts and DAG boundaries | validation, immutability and fail-closed data-gate tests |
| W9–W10 implementation verification | blinded context and static coordinator | planted future shift and label leak |
| W11 kernel execution | `adapters/quant_kernel.py` | isolated overlay, canonical main, default-config byte check |
| W12 reproduction | `verification/reproduction.py` | independent identity, hash and cache-contamination mismatch |
| W13 statistics | `adapters/statistics.py` | registered-trial field and missing-evidence preservation |
| W14 robustness | deterministic attack registry | randomized-placebo inheritance failure |
| W15 generalization | `adapters/generalization.py` | existing kernel invariants remain authoritative |
| W16 governance | `governance/policy.py` | majority cannot override a critical failure; missing evidence holds |
| W17 memory update | memory contracts/importer | accepted and failed records remain first-class |
| W18 DAG | `orchestration/` | R0–R20, resume, budget exhaustion and deterministic output |
| W19 CLI | `cli/` | offline executable demo |
| W20 reports | `reporting/workspace.py` | 21 evidence files plus hash manifest and summary |
| W21 performance | `registry/agent_performance.py` | allowed objective metrics exclude profitability |
| W22 migration | `migration/historical.py`, curated example YAML | missing hashes are not fabricated |
| W23 examples | three YAML workflows | offline synthetic integration path |
| W24 hardening | `security/policy.py`, registry hash validation | secrets, traversal, shell and invariant tests |
| W25 parallelism | DAG permits internal replacement handlers; deterministic ordering retained | parallel provider/source execution remains deferred until real connectors are configured |
| W26 documentation | README and architecture docs | documentation describes only implemented behavior |

## Deferred boundaries

The branch provides a complete safe Research OS foundation and offline reference lifecycle. Real provider connectors, production-scale parallel reconnaissance, automatic conversion of every historical update plan, and full real-market R0–R20 reference runs remain configuration/data work. They are not represented as complete and cannot silently promote evidence. Live brokerage execution, LLM-controlled orders, automatic production activation and a web UI remain out of scope.
