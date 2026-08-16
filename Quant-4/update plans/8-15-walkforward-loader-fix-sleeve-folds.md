# 8-15 全池折线一致性修复 + 套筒折线证据（Round 13）

> 背景：R12 全池套筒复验给出"组合优于单书"的全窗口证据（夏普 1.09 vs 1.01、回撤 -9.91% vs -10.29%）。
> 本轮按 R5/R9 的暖启动折线方法论（每配置一次全程运行后按 4 折切片，连续状态无冷启动噪声）
> 对套筒做折级验证；同时发现并修复 R9 折线脚本的**加载口径不一致**——原 `_walkforward_fullpool.py`
> 使用 scratch loader（glob 全部 parquet = 5503 只，含 25 只 B 股/非 master-list 标的），
> 而生产数字全部来自 `build_pit_universe`（5478 只，master-list 过滤）。本轮用规范 PIT loader 重跑。

## 1. 基线

- 测试套件：`pytest Quant-4/tests -q -p no:cacheprovider` → **174 passed**。
- 全池生产默认每次重跑逐位一致：6.54% / 1.01 / 0.64 / -10.29%，OOS 8.35% / 1.31。

## 2. 加载口径修复：R9 折线重跑（规范 PIT loader）

**发现**：`run_sweep.load_frames`（scratch）glob 全部 `*_history.parquet` = **5503** 只；
`build_pit_universe`（规范）按 master-list 过滤 = **5478** 只（差异 25 只：200xxx/0005xx B 股等，
不在 master list）。R9 的 `_walkforward_fullpool.py` 用了 scratch loader，因此其 2024-26 折
robust 年化 13.52% 是**含 25 只非池标的的伪口径**。规范重跑（`_walkforward_fullpool_pit.py`，
同一 5478 只、同一会计、同一 4 折）：

| 折 | base 年化/夏普/回撤 | robust 年化/夏普/回撤 | 年化胜者 |
|---|---|---|---|
| 2016-18 | 0.99% / 0.18 / -12.12% | 1.57% / 0.27 / -10.29% | robust |
| 2019-21 | 8.72% / 1.33 / -5.72% | 8.92% / 1.33 / -6.20% | robust |
| 2022-23 | 0.02% / 0.03 / -4.86% | 0.81% / 0.19 / -3.87% | robust |
| 2024-26 | 8.62% / 1.09 / -13.40% | **14.49% / 1.93 / -8.34%** | robust |

**结论**：robust **4/4 折年化胜出、3/4 夏普、3/4 卡玛** —— 与 R9 判决一致（胜折数不变），
但规范口径下 2024-26 折为 **14.49%**（R9 的 13.52% 因混入 25 只非池标的而偏低 0.97pp）。
R9 结论稳健性不受影响（判决未翻转），但折线数字以规范重跑为准
（`Quant-4/reports/_iter/walkforward_fullpool_pit_20260815.json`）。

## 3. 套筒暖启动折线（R12 运行切片，4 折）

直接切片 R12 已保存的两个全池运行（单书 `weekly_rotation_default`、套筒
`sleeve_portfolio_fullpool_20260815`，同一 5478 只、同一日历 2578 日，索引逐位对齐）：

| 折 | 单书 年化/夏普/卡玛/回撤 | 套筒 年化/夏普/卡玛/回撤 | 年化胜者 |
|---|---|---|---|
| 2016-18 | 1.57% / 0.27 / 0.15 / -10.29% | **1.77% / 0.31 / 0.18 / -9.91%** | 套筒 |
| 2019-21 | **8.92% / 1.33 / 1.44 / -6.20%** | 8.74% / **1.52 / 1.56** / **-5.61%** | 单书(年化) |
| 2022-23 | 0.81% / 0.19 / 0.21 / -3.87% | **3.46% / 0.73 / 0.85** / -4.10% | 套筒 |
| 2024-26 | **14.49% / 1.93 / 1.74 / -8.34%** | 12.24% / 1.71 / 1.27 / -9.62% | 单书(年化) |

**结论（诚实、细致）**：套筒 **3/4 折夏普、3/4 折卡玛胜出、2/4 年化、2/4 回撤**。
套筒的分散化优势（safe 40% 保底压住 momentum/sprint 高波动书）在 2022-23 熊折最显著
（年化 3.46% vs 0.81%）；但在 2024-26 强牛折**让渡收益**（12.24% vs 14.49%）——
40% 保底书的避险权重拖累牛市参与。**套筒不是"严格占优"层**：全窗口夏普/回撤优势
（R12）来自风险平滑，代价是强牛期收益让渡。证据门如实记录：生产默认保持单书
（用户运行口径），套筒为可选层；若用户偏好更低回撤可选用，若偏好牛市收益最大化则单书更合适。

## 4. 验证与复现

```powershell
python -m pytest Quant-4/tests -q -p no:cacheprovider          # 174 passed
python Quant-4/research/walkforward/walkforward_sleeve.py            # 套筒折线（切片，秒级）
python Quant-4/research/walkforward/walkforward_fullpool_pit.py      # 规范 PIT 折线重跑（~7 分钟，2 次全池）
# 结果：reports/_iter/walkforward_sleeve_fullpool_20260815.json、
#       reports/_iter/walkforward_fullpool_pit_20260815.json（gitignored）
```

## 5. 限制披露

- 无网络：真实新闻/论坛语料全池复验仍未执行（持续数据约束，非代码问题）。
- R9 折线（`walkforward_fullpool.json`）为 scratch-loader 口径，保留为历史记录；
  规范口径见 `walkforward_fullpool_pit_20260815.json`，二者判决一致。
- 套筒折线为切片分析（无新运行），与 R12 全窗口证据同源同口径。
