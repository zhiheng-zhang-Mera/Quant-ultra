# Statistical Validation

Production candidates use the single entry in `Quant-4/Main/statistical_validation_v2.py`. The specification is code-owned and records sample range and total trials.

The suite combines annualized performance with Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, a multiple-testing correction, Probability of Backtest Overfitting, bootstrap confidence intervals, predefined parameter-neighborhood stability and explicit OOS checks. Thresholds must be fixed before observing candidate results.

The gate returns `PASS`, `HOLD`, or `REJECT`; it never promotes a candidate based only on return, Sharpe, or drawdown. Non-finite samples, too-short histories, missing trials, weak OOS evidence or unstable perturbations fail closed.

Interpretation is deliberately narrow: statistical evidence quantifies uncertainty under the supplied sample and assumptions. It does not correct survivorship errors, bad timestamps, vendor revisions, structural regime change, or execution-model misspecification; those belong to separate gates.
