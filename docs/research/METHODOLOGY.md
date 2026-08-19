# Methodology

## Research question and scope

Quant-Ultra is an analysis-only, long-only quantitative research platform. It separates data collection, PIT feature creation, labels, models, allocation, simulated execution, audit, governance, and advisory output into eleven explicit phases. It does not submit broker orders.

## Causal timing contract

Signals are formed only from information available at the declared `as_of`. Close-derived signals execute no earlier than the next eligible open. Embargoed train/validation/test windows and PIT data-bus queries guard against look-ahead. Missing required evidence fails closed; optional evidence is explicitly marked unavailable.

## Portfolio and execution contract

Gross exposure is capped at 1.0, shorting and leverage are disabled, and cash is an explicit residual. Board lots, minimum commission, exchange fees, stamp tax, participation limits, suspension rules, transaction costs and next-open/TWAP assumptions are encoded rather than subtracted informally after evaluation.

## Risk and accounting

The pipeline evaluates concentration, covariance, liquidity, capacity, turnover, stress, drawdown, PSI and research-evidence gates. NAV, cash, positions and costs have automated identities. A runnable strategy is not necessarily an accepted candidate: statistical, experimental and production-readiness gates are separate.

## Decision boundary

`PASS` means the specified engineering/research contract passed. `HOLD` means evidence is insufficient or mixed. `REJECT` preserves a failed formal hypothesis. None of these states is individualized investment advice or proof of future returns.
