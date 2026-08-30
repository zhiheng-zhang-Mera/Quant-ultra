# Research OS architecture

## Truth ownership

| Layer | Owns | Must not own |
|---|---|---|
| Quant Kernel | calculations, simulation, portfolio/risk outputs | research admission |
| Research OS | R0–R20, nested verification, evidence, memory, governance | UI presentation state |
| Application | commands, DTO queries, run/process orchestration, persisted events | verification calculations |
| Desktop | Qt models, interaction, rendering and preferences | PASS/REAL/admission decisions |

The application derives `DEMO_OFFLINE`, `RESEARCH_OFFLINE`, or `REAL_RESEARCH` from explicit capabilities. Legacy state is `UNKNOWN_LEGACY`, never inferred real.

## State reconstruction

`ResearchRunStore` writes atomic state v2. `PersistentEventBus`, `VerificationEvidenceStore`, and `GovernanceDecisionStore` are append-only hash chains. After restart the application hydrates run, lifecycle, verification and governance DTOs from those stores, then replays newer events. A persisted RUNNING state without a live owned process becomes ORPHANED.

## Execution

The canonical R0–R20 DAG emits events before and after handlers. R2/R12/R13/R16/R18/R19 execute nested DAGs through the same dependency engine. Cancellation is cooperative across stage boundaries and is acknowledged before `CANCELLED` is emitted. Managed subprocesses use terminate, bounded grace, then kill fallback.

## Admission

Only `VerificationMatrixBuilder` constructs V1–V12 matrices, using latest immutable records from `VerificationEvidenceStore`. Missing evidence is HOLD. `GovernancePolicy.decide_composite` applies a versioned profile, binds the matrix evidence set and policy hash, and never grants production activation; human authorization remains mandatory.
