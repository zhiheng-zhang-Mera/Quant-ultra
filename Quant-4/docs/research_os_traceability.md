# 8-30 finalization traceability

| Finalization requirement | Implementation | Regression proof |
|---|---|---|
| Mode truth | `application/mode.py` | default demo, capability mode tests |
| Durable CREATED | `application/run_store.py` | restart hydration test |
| V1–V12 truth | `verification/evidence_store.py`, `matrix_builder.py` | empty HOLD, one-level-only change, restart |
| Composite governance | `governance/policy.py`, `decision_store.py` | V2/V5/V12 block, immutable re-evaluation |
| Snapshot provenance | `contracts/provenance.py`, `verification/environment.py` | Git-less clean snapshot and mutation tests |
| Risk counterfactual | `verification/composite.py` | missing counterfactual/capacity HOLD |
| Source lineage | `evidence/independence.py` | unresolved upstream excluded and HOLD |
| Typed L1–L6 | `verification/independent_implementation.py` | strict profile and accounting divergence |
| Live events | `orchestration/dag.py` | STARTED persisted while handler blocks |
| Nested DAG | `orchestration/subdag_executor.py` | critical substage failure blocks parent |
| Active cancellation | `orchestration/cancellation.py`, application service | downstream skipped, no false completion |
| Subscriber isolation | `application/event_bus.py` | subscriber crash leaves event durable |
| Process ownership | `application/process_manager.py` | timeout/cancel child recovery |
| Desktop truth | `Desktop/research_workbench/workbench.py` | mode/matrix/run hydration and event reducer |

Production defaults are outside this finalization implementation and are guarded by an unchanged-file hash check.
