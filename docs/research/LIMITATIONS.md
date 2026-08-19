# Limitations

- Engineering gates validate code contracts, not live broker connectivity or order safety.
- Backtests remain sensitive to data revisions, historical universe construction, fees, liquidity and unmodelled market impact.
- The external ETF validation uses a static liquid universe; it is not historical constituent research.
- Statistical corrections reduce but do not eliminate selection bias, researcher degrees of freedom or regime dependence.
- Free news/announcement/forum sources may have incomplete history, changing schemas and ambiguous licences.
- LLM event parsing can be confidently wrong; confidence is metadata, not ground truth.
- Exact Python 3.12 dependencies are locked, while Python 3.11 CI verifies compatibility using bounded runtime requirements rather than a full platform-specific lock.
- Reports generated from local manifests cannot be independently reproduced without access to the same input bytes.
- `PASS` is scoped to its named gate. Production and investment conclusions remain `HOLD_FOR_REVIEW` or `OBSERVATION_ONLY` when evidence is incomplete.
