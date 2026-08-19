# Reproducibility

Python 3.12 dependencies are exact in `Quant-4/requirements-lock-py312.txt`. Every governed run can record Python/platform details, all installed packages, Git commit and dirty state, lock-file hashes, random seeds, hardware and thread/runtime controls.

Run the representative external-market proof from the repository root:

```powershell
.\.venv-full\Scripts\python.exe Quant-4\research\run_reproducibility_smoke.py --download
```

The command retains raw and cleaned inputs locally, builds a data manifest, executes frozen parameters twice, compares annual return/Sharpe/max drawdown at tolerance `1e-12`, and verifies every input/output hash. Runtime data and reports are ignored by Git.

For performance regression evidence run `Quant-4/research/run_performance_gate.py`. Full-pool and alternative-event workloads have time/memory budgets plus financial-result fingerprints, so a speed change that changes results fails.

Reproducibility proves repeatability for specified bytes, software and tolerance. It does not prove vendor truth, future stability or profitability.
