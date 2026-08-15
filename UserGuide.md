# Quant-Ultra 用户指南 / User Guide

> 双语操作手册：安装、数据、运行、配置、报告解读与故障排查。 / Bilingual manual: installation, data, run, configuration, report reading and troubleshooting.

[中文](#中文) · [English](#english)

---

## 中文

### 1. 环境准备

要求：Windows 10/11，Python 3.11+，可访问网络（首次需要下载数据）。

```powershell
cd D:\Quant-Ultra
.\setup.ps1
```

`setup.ps1` 会创建或复用 `D:\Quant-Ultra\.venv-full`，安装主引擎与回测/优化/测试依赖，并依次执行：依赖导入 → `pip check` → Cython 编译 → 单元测试验收。结果写入 `Quant-4/reports/setup/setup_report.json`。

可选参数：

| 参数 | 说明 |
|---|---|
| `-Mirror <URL>` | 指定 PyPI 镜像 |
| `-SkipTests` | 跳过测试验收 |
| `-NoPipUpgrade` | 跳过 pip/setuptools/wheel 升级 |

### 2. 数据准备

历史日线缓存位于 `Quant-4/Data_Cache/`（gitignore，运行期生成），文件名形如 `600519.SH_history.parquet`。数据源支持 akshare 与 efinance 多通道切换，并带熔断与超时审计。

自适应周轮动的避险资产池（已在生产配置中启用，首次运行需联网抓取）：

| 代码 | 资产 | 年化（2016-2026 样本） | 波动 |
|---|---|---|---|
| `511010.SH` | 国债 ETF | ≈3.2% | ≈2.4% |
| `511260.SH` | 十年国债 ETF | ≈4.2% | ≈2.8% |
| `518880.SH` | 黄金 ETF | ≈12.4% | ≈15.3% |
| `511880.SH` | 货币 ETF | ≈2.8% | ≈0.4% |

缺失时可用数据源管理器补抓：

```python
from Main.datasource_manager import FreeDataSourceManager
mgr = FreeDataSourceManager()
df = mgr.fetch_historical("511010.SH", "2015-01-01", "2026-08-09")
```

### 3. 运行完整流水线

```powershell
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\Main\main.py --force-recompute --non-interactive
```

| 参数 | 说明 |
|---|---|
| `--config <path>` | 外部 YAML 配置路径 |
| `--only-phase 6` | 仅执行 Phase 6 及其依赖 |
| `--resume-from 6` | 从 Phase 6 断点恢复 |
| `--skip-phases 3,7` | 跳过指定阶段 |
| `--offline` | 全离线（仅缓存） |
| `--force-recompute` | 忽略缓存强制重算 |
| `--symbols 600519.SH,510300.SH` | 受限真实数据运行 |
| `--download-workers 1` | 限制 Phase 1 下载并发 |

报告输出：`Quant-4/reports/runs/<run-id>/`，含每阶段双语 Markdown/JSON、自包含 `execute_report.html` 与结构化 `execute_report_data.json`。

### 4. 运行自适应周轮动回测

```powershell
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\run_weekly_rotation.py
# 自定义起点
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\run_weekly_rotation.py --start 2020-01-01
# 稳健风险档（2026-08-14 离线推荐：固定 3% + z-score 3.0 冲击检测、8% 止损带；全池复验后采纳）
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\run_weekly_rotation.py --pit --profile robust --num-trials 81
```

输出到 `Quant-4/reports/weekly_rotation/`：

| 文件 | 内容 |
|---|---|
| `weekly_rotation_report.md` | 双语报告 + 门槛记分卡 + 第三视角审查 |
| `weekly_rotation_summary.json` | 结构化指标（含近3年恢复期） |
| `weekly_rotation_returns.csv` | 日频净值/收益/敞口/换手 |
| `weekly_rotation_monthly.csv` | 月度策略 vs 等权基准 |
| `weekly_rotation_equity.png` | 净值与回撤曲线 |
| `weekly_rotation_monthly_heatmap.png` | 月度收益热力图 |

**指标口径**：

| 指标 | 口径 |
|---|---|
| 夏普 | 日收益均值/标准差 × √252 |
| 卡玛 | 年化收益 / 最大回撤 |
| 回撤修复期（近3年） | 最近 3 年窗口内“峰值→新高”最大连续回撤天数（门槛口径，≤126） |
| 回撤修复期（全窗口） | 全历史同口径值（披露） |
| 季度超基准胜率 | 季度收益跑赢等权基准/沪深300 的季度占比 |

### 5. 按新数据迭代参数（自适应回测）

```powershell
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\run_adaptive_backtest.py 600519 --kind stock --years 8
# 诊断模式（禁用跨运行迭代）
... run_adaptive_backtest.py 600519 --kind stock --years 8 --disable-self-optimize
```

迭代只在数据截止日推进后扩展搜索网格；每个 walk-forward 折仍只使用折前数据选参，不产生交易授权。状态按标的存于 `reports/adaptive_backtests/parameter_state/`，含 SHA-256，篡改即失败关闭。

### 6. 配置参考

**`Quant-4/Main/default_param.yaml`（全流水线）** 关键项：

| 键 | 默认 | 说明 |
|---|---|---|
| `commission_rate` / `stamp_tax` | 0.00025 / 0.0005 | 佣金/印花税 |
| `slippage_rate` | 0.0002 | 滑点 |
| `cash_buffer_weight` / `max_daily_turnover` | 0.05 / 0.25 | 现金缓冲/单日换手上限 |
| `max_single_stock_weight` | 0.05 | 单票上限 |
| `embargo_min` / `holding_period` | 5 / 5 | 隔离带/持有期 |
| `local_llm_*` | — | 本地 Ollama 情绪增强限额 |

**`run_weekly_rotation.py::default_params`（周轮动）** 关键项：

| 参数 | 生产值 | 说明 |
|---|---|---|
| `regime_ma` / `regime_ma_fast` / `regime_confirmation_ma` | 40 / 10 / 200 | 市场状态均线 |
| `regime_model` / `ml_bear_override` | `logit` / True | ML 状态检测 |
| `ml_bear_floor` / `ml_bear_low` / `ml_bull_high` | 0.30 / 0.40 / 0.60 | 连续敞口映射 |
| `defensive_hold_assets` | 4 只避险 ETF | 防御态安全资产池 |
| `defensive_hold_exposure` / `defensive_hold_safe_frac` | 1.0 / 0.65 | 防御态总敞口/安全占比 |
| `event_shock_threshold` / `event_shock_exposure` | 0.025 / 0.70 | 事件冲击触发/避险占比 |
| `euphoria_threshold` | 0.10 | 亢奋过滤（20日涨幅） |
| `max_holding_days` | 63 | 3 个月持有上限 |
| `enable_intraweek_stops` / `stop_loss_pct` / `take_profit_pct` | True / 0.08 / 0.06 | 小额收割 |
| `confirm_leverage` / `max_gross_exposure` | 1.0 / 1.0 | 杠杆端到端禁用（个人资金不负债） |
| `vol_target` / `vol_scale_floor` | 0.20 / 0.90 | 高地板波动率目标 |
| `max_annual_vol` / `per_position_cap` | 0.40 / 0.30 | 波动率过滤/单票上限 |

### 7. 解读门槛记分卡

报告末尾的门槛记分卡按用户定义的四项目标判定：夏普 ≥0.9、卡玛 ≥1.2、回撤修复（近3年）≤126 交易日、季度超基准胜率（等权 ≥50% / 沪深300 ≥60%）；同时披露全窗口修复期、换手率、杠杆状态与样本外衰减。未达标项会如实标为“未达”，不做虚标。

### 8. 故障排查

| 现象 | 处理 |
|---|---|
| 网络下载失败 | 重试或 `--offline` 用缓存；数据源带熔断会自动切换通道 |
| 缺少避险 ETF 数据 | 用数据源管理器补抓（见第 2 节） |
| `git` 脏工作区校验失败 | 提交/暂存改动后再运行，或明确使用 `--no-git-check` |
| 测试报错 | 先 `python -m pytest tests -q` 定位；历史已知环境类错误与本次改动无关时核对版本 |
| 报告图片不显示 | 图片在 `docs/images/` 需受版本控制；运行期报告在 `reports/`（gitignore） |

### 9. FAQ

- **为什么防御状态不再空仓？** 修正前约 52% 时间空仓错过反弹；现在防御状态持有避险资产，让净值在长熊中持续增长。
- **为什么没有杠杆？** 按个人小资金原则杠杆端到端禁用（`confirm_leverage=1.0`、`max_gross_exposure=1.0`），只用自盘资金、无做空、无负债风险。
- **回撤修复期为什么有两个值？** 门槛采用近 3 年滚动窗口口径；全窗口口径受 2018 与 2021-2023 两段市场性长熊影响，如实披露。
- **可以用于实盘吗？** 不能。本项目为研究与工程验证，模拟成交不能替代券商回单，免费数据源可能限流/改版。

---

## English

### 1. Environment

Windows 10/11, Python 3.11+. Internet access required on first run (market data download).

```powershell
cd D:\Quant-Ultra
.\setup.ps1
```

`setup.ps1` creates/reuses `D:\Quant-Ultra\.venv-full`, installs engine/backtest/optimization/test dependencies, then runs import → `pip check` → Cython build → unit-test acceptance. Report: `Quant-4/reports/setup/setup_report.json`.

Options: `-Mirror <URL>`, `-SkipTests`, `-NoPipUpgrade`.

### 2. Data

Daily OHLCV cache lives in `Quant-4/Data_Cache/` (gitignored), e.g. `600519.SH_history.parquet`. Sources: akshare / efinance multi-channel with circuit breaker.

Safe-asset pool used by the adaptive rotation (fetched on first run):

| Code | Asset | Sample annual return | Volatility |
|---|---|---|---|
| `511010.SH` | Treasury ETF | ≈3.2% | ≈2.4% |
| `511260.SH` | 10Y Treasury ETF | ≈4.2% | ≈2.8% |
| `518880.SH` | Gold ETF | ≈12.4% | ≈15.3% |
| `511880.SH` | Money-market ETF | ≈2.8% | ≈0.4% |

Fetch missing data:

```python
from Main.datasource_manager import FreeDataSourceManager
mgr = FreeDataSourceManager()
df = mgr.fetch_historical("511010.SH", "2015-01-01", "2026-08-09")
```

### 3. Full Pipeline

```powershell
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\Main\main.py --force-recompute --non-interactive
```

Key args: `--only-phase`, `--resume-from`, `--skip-phases`, `--offline`, `--force-recompute`, `--symbols`, `--download-workers`.

Reports: `Quant-4/reports/runs/<run-id>/` with per-phase bilingual Markdown/JSON, `execute_report.html` and `execute_report_data.json`.

### 4. Weekly Rotation Backtest

```powershell
D:\Quant-Ultra\.venv-full\Scripts\python.exe Quant-4\run_weekly_rotation.py
```

Outputs to `Quant-4/reports/weekly_rotation/` (bilingual report, summary JSON, daily/monthly CSVs, equity chart, heatmap). Metric definitions: Sharpe = mean/std × √252; Calmar = annual return / max DD; recovery (last-3y) = max below-peak streak in the recent 3-year window (gate ≤126); full-window recovery disclosed; quarterly win rate vs equal-weight benchmark / CSI 300.

### 5. Adaptive Per-Symbol Backtest

```powershell
... run_adaptive_backtest.py 600519 --kind stock --years 8
... run_adaptive_backtest.py 600519 --kind stock --years 8 --disable-self-optimize
```

Search-grid iteration advances only when the data cutoff moves forward; every walk-forward fold selects parameters on pre-fold data only. State is hashed and tamper-evident.

### 6. Configuration

`Quant-4/Main/default_param.yaml` — pipeline-wide costs, slippage, cash buffer, turnover cap, concentration cap, embargo, optional local-LLM sentiment limits.

`run_weekly_rotation.py::default_params` — regime MAs (40/10/200), logit regime detector, continuous ML exposure (floor 0.30 / low 0.40 / high 0.60), safe-asset pool (4 ETFs), defensive hold (exposure 1.0, safe share 0.65), event shock (0.025 / 0.70), euphoria filter (0.10), 63-day max hold, 6% TP / 8% SL harvesting, no leverage (confirm_leverage=1.0, max_gross_exposure=1.0), vol target 0.20/floor 0.90, vol cap 0.40, position cap 0.30.

### 7. Reading the Gate Scorecard

The report's gate scorecard checks the four user-defined targets (Sharpe ≥0.9, Calmar ≥1.2, last-3y recovery ≤126 trading days, quarterly win ≥50% vs equal-weight / ≥60% vs CSI 300), and discloses full-window recovery, turnover, leverage state and OOS decay. Failures are reported honestly.

### 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Network download failure | Retry or `--offline`; sources auto-switch channels |
| Missing safe-asset data | Fetch via the data-source manager (Section 2) |
| Git dirty-worktree gate | Commit/stash changes, or explicitly `--no-git-check` |
| Test errors | Run `python -m pytest tests -q`; confirm version/environment issues are unrelated |
| Images not rendering | Tracked images live in `docs/images/`; runtime reports are gitignored |

### 9. FAQ

- **Why no more cash in defensive states?** The old model sat in cash ~52% of days; defensive states now hold safe assets so the equity keeps compounding through equity bears.
- **Why no leverage?** Per the small-personal-capital principle, leverage is disabled end-to-end (`confirm_leverage=1.0`, `max_gross_exposure=1.0`): own capital only, no shorting, no debt risk.
- **Why two recovery values?** The gate uses a rolling last-3-year window; the full-window value (affected by the 2018 and 2021-2023 market cycles) is disclosed.
- **Can this trade live?** No. This is a research/engineering system; simulated fills are not broker confirmations and free data sources may be rate-limited.

---

**Disclaimer** — Research and education only. No investment advice. Past performance does not guarantee future results.
