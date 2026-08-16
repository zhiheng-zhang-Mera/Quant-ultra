# 8-16 Round 19.8：代码落盘审计 + 代码位置整理 + 利润/资产负债表因子激活 + 全量回测

> 日期：2026-08-16 | 目标（用户指示，按顺序）：① 保证所有功能确实有代码实现且存在于项目；
> ② 按流程/阶段清理代码放置位置；③ 激活利润/资产负债表因子抓取；④ 激活后回测并检查
> 年化/夏普/卡玛/最大回撤/胜率。

## 1. 代码落盘审计（R19-R19.7 功能逐项核对）

- **生产/引擎/工具/测试**：全部已追踪（git ls-files 确认）——`Phase_3/alternative_data.py`
  （DeepSeek 回退）、`tools/fetch_forum_guba.py`（股吧）、`tools/fetch_history_notices_eastmoney.py`
  （东财公告）、`Main/weekly_rotation.py`（早期再入场参数）、`Main/fundamental_factors.py`（ocf_np +
  现金流缓存合并 + PIT 修复）、`Main/factor_library.py`（amount_share）、`run_sleeve_portfolio.py`
  （ocf_np 生产挂载）、5 个测试文件等；
- **研究/验证脚本**（R19-R19.7 的 LLM 评分/A-B/证据门 harness）：此前仅在 gitignored 的
  `reports/_iter`（本地存在但不在仓库）→ 已全部纳入版本控制（见 §2）。

## 2. 代码位置整理（按阶段/用途归位，commit f2cc3a3）

| 位置 | 内容 | 阶段 |
|---|---|---|
| `research/llm/` | deepseek_score_corpus / llm_vs_lexical / llm_real_news_ab / e2e_deepseek_pipeline | Phase-3 另类数据 LLM 研究 |
| `research/` | notice_ab_historical / event_signal_ab / crowding_ab / cashflow_ab_fullpool / sleeve_ocfnp_ab / sleeve_weight_sweep / reentry_sweep_fullpool / stop_sweep_fullpool / real_news_ab / alt_signal_fullpool_test / run_sweep | 因子/机制证据门回测研究 |
| `research/walkforward/` | walkforward_sleeve / walkforward_fullpool_pit / walkforward_warm / walkforward_fullpool | 暖启动折线切片 |
| `tools/` | fetch_cashflow_top600.py（新增）+ 既有抓取工具 | 数据获取 |
| `tests/` | 各轮测试（已追踪） | 验证 |

全部脚本根锚定（Q4 = PROJECT_ROOT = Quant-4 目录，硬编码 `Quant-4\...` 路径改为 Q4 相对），
任意 CWD 可运行；`test_honest_backtest.py` 的 sweep-harness 测试更新指向 `research/run_sweep.py`
（此前因脚本在 gitignored 目录而静默跳过）；7 个计划文档的脚本引用同步更新。**194/194 测试全绿。**

## 3. 利润/资产负债表因子激活（进行中）

- 工具：`tools/fetch_fundamentals.py`（baostock query_profit_data + query_growth_data）与
  `tools/fetch_balance.py`（query_balance_data + query_operation_data），年度 Q4、top-600 流动性
  名单、pubDate 公告日 PIT 对齐、断点续跑；baostock 冒烟测试通过（600519 2025Q4 真实返回）。
- 抓取：`fundamentals_annual.json`（gp_margin/np_margin/roe/epsTTM/yoyNI/yoyPNI）与
  `fundamentals_balance.json`（debt_ratio/current_ratio/asset_turn）——各 ~90 分钟（0.1/s 限速）。
- 激活后：生产默认 `run_sleeve_portfolio --fundamental-factors`（gp_margin:0.10,yoy_ni:0.05,
  np_margin:0.05,ocf_np:0.05）的利润/资产负债因子从"静默空转"转为真正生效（R19.7 发现的缓存缺失
  问题彻底消除）。

## 4. 激活后回测（完成，全池 PIT 5478 只，套筒 40/30/20/10）

利润表（598 只）+ 资产负债表（595 只）+ 现金流（506 只）缓存齐全后（合并 753 只、10 字段）：

| 配置 | 年化 | 夏普 | 卡玛 | 最大回撤 | 月度胜率 | 修复3y | OOS 年化/夏普 |
|---|---|---|---|---|---|---|---|
| ocf_np only（R19.7 采纳） | 8.70% | 1.366 | 1.316 | -6.61% | 68.8% | 185 | 11.42% / 1.750 |
| **全基本面激活（生产默认）** | **9.21%** | **1.470** | **1.609** | **-5.73%** | 67.2% | 185 | **11.69% / 1.827** |

**结论：利润/资产负债表因子激活带来进一步改善**——年化 8.70%→9.21%（+0.51pp）、夏普
1.366→1.470、卡玛 1.316→1.609、回撤 -6.61%→-5.73%、OOS 夏普 1.750→1.827；月度胜率 68.8%→
67.2%（高位微降）。生产默认 `run_sleeve_portfolio --fundamental-factors`
（gp_margin:0.10,yoy_ni:0.05,np_margin:0.05,ocf_np:0.05）从"静默空转"变为**全部真正生效**。

**工程修复（本轮）**：`load_all_fundamentals` 合并排序确定为流动性序（每缓存独立 top_n，
union 覆盖；原实现的无序 set 合并在上限截断时依赖 hash 序）——回归测试覆盖；
R19.7 的 ocf_np 数字（8.53%）是在 profit/balance 缓存缺失的窄横截面下测得，缓存齐全后
（含未覆盖→中位数的框架语义）实测 8.70%，方向一致且更优。

## 5. 验证与复现

```powershell
python Quant-4/tools/fetch_fundamentals.py --top-n 600      # 利润表（~90 分钟，可续跑）-> 598 只
python Quant-4/tools/fetch_balance.py --top-n 600           # 资产负债表（~90 分钟，可续跑）-> 595 只
python Quant-4/research/sleeve_fundamentals_ab.py           # 全因子激活 A/B（~36 分钟）
python -m pytest Quant-4/tests -q -p no:cacheprovider       # 195 passed
```

## 6. 用户要求的四项检查结果

| 要求 | 结果 |
|---|---|
| ① 所有功能有代码实现且存在于项目 | ✅ 生产/工具/测试已追踪；19 个研究脚本从 gitignored 的 reports/_iter 纳入版本控制（research/） |
| ② 按流程/阶段清理代码位置 | ✅ research/llm（Phase-3 LLM 研究）、research/（因子/机制证据门）、research/walkforward（折线）、tools/（抓取工具）、tests/（验证），根锚定路径 + 文档引用同步 |
| ③ 激活利润/资产负债表抓取 | ✅ baostock 利润表 598 只 + 资产负债表 595 只（pubDate PIT 对齐），合并 753 只 10 字段，生产默认从静默空转变真正生效 |
| ④ 激活后回测指标 | ✅ 年化 **9.21%**、夏普 **1.470**、卡玛 **1.609**、最大回撤 **-5.73%**、月度胜率 **67.2%**、OOS 11.69%/1.83（全池 PIT、套筒+全基本面） |
