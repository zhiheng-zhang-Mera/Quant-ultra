# Quant-Ultra

面向 A 股、ETF 与美股研究的可审计量化流水线。本分支在原有 Phase 1–10 基础上增加 Cython 热路径、跨网络环境的数据源回退、逐阶段证据报告、数据质量门禁和指定持仓交互分析。

> 重要：本项目用于研究与工程验证，不构成投资建议。任何免费数据源或代理都无法承诺在所有国家、地区和网络中永久可用；系统通过多源回退、可配置代理、缓存和完整性证据降低风险，但不伪造“全球绝对可用”保证。

## 1. 系统要求

- Windows 10/11 或 Linux；Python 3.11–3.13 推荐
- 64 位 C/C++ 编译器（仅编译 Cython 扩展需要）
- 建议至少 16 GB 内存
- 本仓库放在 D 盘时，缓存、日志、阶段结果与报告默认也全部留在 D 盘

## 2. 安装

```powershell
Set-Location D:\Quant-Ultra\Quant-4
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

不要使用旧版 `install_deps.py` 的“升级全部系统包”方式。固定虚拟环境可减少依赖漂移。

## 3. Cython 加速

```powershell
.\.venv\Scripts\python setup.py build_ext --inplace
.\.venv\Scripts\python -c "from Main.fast_math import BACKEND; print(BACKEND)"
```

输出 `cython` 表示扩展已加载；无编译器时自动使用经过同一测试的 NumPy 回退，不影响正确性。当前扩展加速日志收益率和下行标准差等高频数值循环；Pandas/网络 I/O 不适合盲目 Cython 化。

## 4. 数据访问与代理

默认 A 股按 AkShare、BaoStock、Tushare（配置 token 时）、efinance 回退；美股使用 yfinance 与 AkShare 灾备。下载结果写入 `Quant-4/data_cache`，每次成功来源在 `data_cache/evidence` 记录：

- 来源与抓取时间
- 行数和日期边界
- OHLCV 结构、不重复、有限数值和价格区间校验
- 内容 SHA-256
- 是否使用代理

在受限网络中显式配置由你信任的 HTTP(S) 代理：

```powershell
$env:QUANT_ULTRA_PROXY='http://127.0.0.1:7890'
```

也兼容标准 `HTTPS_PROXY`。系统不自动采用互联网上的匿名免费代理：这类节点可能窃听、篡改数据、突然离线，不能作为可信金融数据通道。建议使用用户自行控制的免费开源客户端与合法可用节点。

## 5. 运行流水线

```powershell
Set-Location D:\Quant-Ultra\Quant-4
.\.venv\Scripts\python Main\main.py --force-recompute --portfolio-query
```

常用参数：

```text
--only-phase 6       运行指定阶段及其依赖
--resume-from 6      从指定阶段恢复
--skip-phases ...    跳过阶段
--offline            只使用已验证本地缓存
--force-recompute    忽略阶段缓存
```

每一阶段都会在 `Quant-4/reports/runs/<运行时间>/` 生成独立 `.md` 与 `.json` 报告。十阶段结束后还会生成 `post_pipeline_candidates.md/csv`，逐项给出筛选后的股票/ETF、理想买入区间、建议配置仓位和理想止盈比例。`--portfolio-query` 会在其后进入持仓查询。

## 6. 指定 A 股 / ETF 持仓分析

```powershell
.\.venv\Scripts\python analyze_cn_asset.py --end-token END
```

输入格式为 `总资金 代码 持仓数量 平均成本 [可选类型]`：

```text
1000000 600519 300 1450 stock
1000000 510300 20000 3.85 etf
END
```

输出包括年化收益与波动、Sharpe、Sortino、最大回撤、Calmar、95% VaR/CVaR、ATR、20/60 日均线、浮盈亏、账户实际仓位、模型目标仓位、目标持股数、应增减数量、入仓区间、止盈价、风险参考价、操作建议和触发理由。A股数量按100股整数批次计算，目标仓位带10%单标的上限；使用前仍需结合税费、流动性、停牌、涨跌停和个人风险约束。

## 7. 数学定义

- 对数收益：`r_t = ln(P_t / P_{t-1})`
- 年化波动：`sigma_a = std(r) sqrt(252)`
- Sharpe：`(R_a - R_f) / sigma_a`
- Sortino：`(R_a - R_f) / sigma_downside`
- Calmar：`R_a / |MDD|`
- 风险平价：各资产风险贡献 `w_i (Sigma w)_i / sqrt(w' Sigma w)` 相等
- Black–Litterman：以市场隐含收益和观点矩阵的精度加权获得后验收益

协方差矩阵在优化前投影到半正定空间，避免负特征值导致虚假的风险估计。

## 8. 验证

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python tests\run_acceptance.py
.\.venv\Scripts\python -m compileall Main cython analyze_cn_asset.py
git diff --check
```

## 8.1 无未来预知的参数自适应回测

```powershell
.\.venv\Scripts\python run_adaptive_backtest.py 600519 --kind stock
.\.venv\Scripts\python run_adaptive_backtest.py 510300 --kind etf --train-size 504 --test-size 63 --embargo 5
```

回测使用 expanding walk-forward：每一折只在测试期之前的训练窗选择快慢均线、波动窗口和目标波动率，参数随后在整段测试窗冻结。训练窗与测试窗之间强制保留 embargo；`t` 日收盘信号只能在 `t+1` 日开盘成交，收益使用 `t+1` 到 `t+2` 的开盘价格。任何 `signal_time >= execution_time` 或训练/测试重叠都会立即终止。报告位于 `Quant-4/reports/adaptive_backtests`，包含每折参数、时间边界、收益明细及防泄漏审计。

参数自适应不是用测试集反复调参：测试结果不反馈给同一折的参数选择。需要进一步避免研究者对全部历史测试结果的人为过拟合时，应另留从未查看的最终 holdout 数据。

正常 Python 发行版使用 pytest；缺少 `unittest` 的裁剪版/嵌入式 Python 可运行第二条独立验收命令。测试覆盖数据门禁的正反例、Cython/NumPy 数值等价接口、风险平价不变量，以及股票/ETF 代码归一化。联网数据正确性还需查看当次 `evidence` 清单；离线单元测试不能证明第三方实时行情本身真实。

## 9. 目录

```text
Quant-4/
├─ Main/                 编排、数据、报告与金融数学
├─ Phase_1 ... Phase_10/ 原流水线阶段
├─ cython/               Cython 热路径源码
├─ tests/                确定性测试
├─ analyze_cn_asset.py   A股/ETF 交互分析
├─ requirements.txt      隔离环境依赖
└─ setup.py              Cython 构建入口
```

## 10. 已知边界

- 免费第三方接口可能改版、限流或因当地政策不可达。
- 哈希和结构校验可证明“下载后未静默改变、格式和数值关系合理”，不能单独证明发行方数据绝对真实。
- 首次全市场运行耗时和存储占用较大；建议先运行单阶段及小范围标的。
- 交易信号是透明、可复算的规则模型，不是收益承诺。
