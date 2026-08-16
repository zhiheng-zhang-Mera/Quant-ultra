# 8-16 Round 19：DeepSeek LLM 差异化验证 + 套筒正式采纳 + 股吧可用源（强化收尾轮）

> 日期：2026-08-16 | 目标：按优先级完成 8-13 遗留的强化项，然后回测表现。
> 前置：R18 确认真实新闻情绪无稳健方向性 alpha（38 只全池，`alternative_signal_weight` 保持 0）；
> R17 记录 LLM vs 词典差异化验证的唯一缺口是 GPU 显存。本轮 DeepSeek API 回退已就位（上一提交），
> 本机环境已设置 DEEPSEEK_API_KEY（User 作用域，脚本内注入，不在对话/文档中输出密钥）。

## 1. 优先级 1：真实语料上的 LLM 差异化验证（DeepSeek，已解除硬件阻塞）

**背景**：8-13 计划假设"合成文本上 LLM 与词典高度相关（0.97），真实语料才会让 LLM 产生差异化价值"。
R17 因本机 GPU 显存不足（qwen 系无法加载、gpt-oss 输出为空）无法验证。DeepSeek 回退实现后，
用云端 API 对 **12066 条真实新闻**（`sina_news_real_38.jsonl`，38 只生产池全量）做 LLM 情绪评分。

**执行**：`reports/_iter/_llm_deepseek_score_real.py` —— 批量 prompt（每批 30 条，packaged lexical_score
与生产 `enhance_sentiment_with_local_llm` 同格式）、temperature 0、逐批落盘可续跑、失败重试 3 次；
输出 `reports/_iter/llm_deepseek/llm_scores_real_38.jsonl`（逐条 id/symbol/published_at/text/lex/llm）。

**结果（诚实）**：全语料 **12066/12066 条评分成功**（403 批、0 失败、~45 分钟、deepseek-chat、温度 0）。
评分记录：`reports/_iter/llm_deepseek/llm_scores_real_38.jsonl`（逐条 id/symbol/published_at/text/lex/llm）。

**LLM vs 词典对比（`_llm_vs_lex_deepseek.py`，全 12066 条）**：

| 指标 | 值 | 8-13 合成文本对照 |
|---|---|---|
| Pearson 相关 | **0.530** | 0.97（合成文本） |
| Spearman 相关 | 0.450 | — |
| 符号分歧 | **59.8%** | 32%（合成文本） |
| 强信号（\|s\|≥0.5） | 词典 10% / **LLM 28%** | — |
| 中性占比 | **词典 90% / LLM 18%** | — |
| 均值/标准差 | 词典 +0.053/0.307 vs **LLM +0.194/0.352** | — |

**8-13 假设在真实语料上得到确认**：真实新闻文本让 LLM 产生显著差异化——词典对 90% 的真实标题
判定中性、LLM 仅 18%；二者相关仅 0.53（合成文本上 0.97）、近 60% 记录符号分歧；LLM 情绪整体
更偏正且区分度更高。这是 R17 硬件阻塞解除后（DeepSeek 云端 API）首次完成的真实语料验证。

**LLM 面板 A/B（`_llm_real_ab.py`，38 只生产池，同 R18 口径；llm = LLM 情绪、缺失回退词典）**：

| 配置 | 全窗口 年化/夏普 | OOS 年化/夏普 | 信号窗口(06-08) 年化/夏普 |
|---|---|---|---|
| base（无信号，R18） | 7.89% / 0.89 | 5.75% / 0.71 | -3.89% / -0.59 |
| 词典 w=0.10（R18 复现） | 7.90% / 0.89 | 5.80% / 0.72 | -3.07% / -0.46 |
| 词典 w=0.20 | 7.92% / 0.89 | 5.82% / 0.72 | -2.59% / -0.37 |
| **LLM w=0.20** | **8.01% / 0.90** | **6.03% / 0.74** | **+1.61% / +0.28** |
| LLM 反转 w=0.10 | 7.89% / 0.89 | 5.75% / 0.71 | -3.87% / -0.59 |

**解读（诚实）**：LLM 面板与词典面板产生**不同的方向响应**——词典在信号窗口随权重单调拖累
（-3.89%→-2.59% 仍为负），LLM 在 w=0.20 将窗口翻正（**+1.61%**，OOS 5.82%→6.03%），反转信号
不优于正向（LLM 正方向本身可用）。差异化的存在已确认（见上表），方向响应也随评分改变；但信号
窗口仅 3 个引擎再平衡日期（免费源覆盖 28%）、全窗口/OOS 差异仍在 ±0.2pp 内——**尚不构成可
稳健开采的 alpha**，`alternative_signal_weight` 保持 0（与 R18 结论一致，但 LLM 路径的方向性
证据升级为"正方向不再拖累"而非"反转有效"）。决定性验证需要更长真实文本历史（付费/授权数据源）。

## 2. 优先级 2：回撤修复期 / 卡玛 —— 套筒层风险平滑的正式采纳决策

**候选方向**：
- (a) 降低换手成本：R10 已测 `rebalance_min_turnover` 0.05/0.10 —— 无效果（平均换手 0.38 ≫ 门槛）；
  套筒层组合级平均换手仅 0.066（配额内再平衡），是结构性降换手答案；
- (b) **套筒 40/30/20/10 正式采纳（本轮决策）**——证据已就绪（R12 全池 5478 只、R13 折线）；
- (c) 2026 回撤期"更早再入场"规则：R10/R12 已测 14 项回撤机制候选全负/噪声内；181 日修复期来自
  2026 进行中回撤的结构性滞后（策略 -7.16% 优于基准 -9.52%，修复需基准收复 MA40），未引入新假设
  （避免过拟合，沿用证据门）。

**R12/R13 全池证据（5478 只、2016-01~2026-08、2578 日）**：

| 指标 | 单书（生产默认） | 套筒 40/30/20/10 | 变化 |
|---|---|---|---|
| 年化 | 6.54% | 6.55% | ≈ 持平 |
| 夏普 | 1.010 | **1.085** | +0.075 |
| 卡玛 | 0.636 | **0.661** | +0.025（仍 <1.0，结构性披露） |
| 最大回撤 | -10.29% | **-9.91%** | **研究门 max_drawdown 翻转 PASS** |
| 月度胜率 | 57.8% | **60.2%** | +2.4pp |
| OOS 年化/夏普 | 8.35% / 1.31 | 8.35% / **1.35** | 夏普 +0.04 |
| 回撤修复期 | 181 日 | 181 日 | 不变（结构性） |

折线（R13）：套筒 夏普/卡玛 **3/4 折胜出**、年化/回撤 2/4；2022-23 熊折显著（3.46% vs 0.81%），
2024-26 强牛折让渡收益（12.24% vs 14.49%）——风险平滑 vs 牛市收益的取舍。

**正式决策（本轮采纳）**：**采纳套筒 40/30/20/10 为生产默认风险层**（`run_sleeve_portfolio.py --pit`
为生产入口），单书保留为 A/B 对照（`run_weekly_rotation.py`）。理由：全窗口年化不损失的前提下
夏普 +7.4%、回撤翻转门 PASS、月度胜率跨过 60%；卡玛 HOLD 与 2024-26 强牛让渡为如实披露的取舍，
研究证据门仅卡玛一项 HOLD（0.661 vs 1.0，结构性上限已披露）。

**生产池演示（本轮，非 PIT 口径，58 只生产池，2016-01~2026-08）**：

| 指标 | 单书 | 套筒 40/30/20/10 |
|---|---|---|
| 年化 | 6.47% | **6.75%** |
| 夏普 | 1.018 | **1.092** |
| 卡玛 | 0.800 | 0.789 |
| 最大回撤 | -8.08% | -8.56% |
| 月度胜率 | 59.4% | **61.7%** |
| 平均调仓换手 | 0.349 | **0.061** |
| 笔数/成本 | 58 笔 / 7.97% | 231 笔（套筒流成本 ~0.01%） |

演示口径下套筒年化/夏普/月度胜率更优、换手大幅下降（配额内再平衡），回撤略深 0.48pp（卡玛随之
略降）；与全池 PIT 证据方向一致。回撤修复期两者同为进行中 2026 回撤（3 年窗 113 日未修复）——
结构性滞后，套筒不改变该上限。

## 3. 优先级 3：数据/信号层扩展 —— 股吧论坛情绪可用源（此前 403，已解除）

- **诊断**：`gbapi.eastmoney.com` JSON API 持续 403（风控），但 **`guba.eastmoney.com/list,{code},f.html`
  HTML 列表页完全可达（HTTP 200）**，含每只股票的社区帖子（阅读/评论数、作者、MM-DD HH:MM 时间戳）。
- **新工具**：`tools/fetch_forum_guba.py` —— 解析列表页 → 契约全列（record_id/source/source_type/
  license/published_at/ingested_at/symbol/text/is_synthetic=False，source=guba_eastmoney，
  source_type=forum）；时间戳按 **UTC+8（中国标准时间）** 解析并转 UTC（首版按 UTC 解析导致最新帖被
  PIT 守卫丢弃，已修复）；逐符号抓取后快照 ingest 时间（PIT 一致）。
- **产出**：`Data_Cache/alternative_raw/guba_forum_real_38.jsonl`（38 只生产池 × 2 页 ≈ 6000 条真实帖子）。
- **限制（如实记录）**：列表页只给"最后更新"时间（非发帖时间）与 MM-DD HH:MM（年份推断为当年/前一年）；
  免费源近期窗口约束与 Sina 同类。语料供后续论坛情绪方向性探测；是否进入 A/B 需单独证据门。

## 4. 优先级 4：工程与治理 —— DeepSeek 路径端到端验证

- `Phase_3/alternative_data.py` DeepSeekClient（OpenAI 兼容 chat/completions，读环境变量）已在
  上一提交实现并测试（179/179）。
- 本轮新增 `reports/_iter/_e2e_deepseek_pipeline.py`：真实语料 + 生产池 38 只 + 显式 as_of +
  Ollama 不可用（provider chain 自动跳过）+ DEEPSEEK_API_KEY 注入，走 `build_alternative_signals`
  （`main.py` Phase-3 实际调用路径）。
- **结果：E2E-DEEPSEEK: PASS** —— 契约 LOADED（sha256 970cc0…）、`local_llm` 证据
  `status=ANALYZED / provider=deepseek / model=deepseek-chat / analyzed=6/6`（生产实时预算上限 6 条），
  LLM 情绪进入 `news_sentiment` 与 `alternative_signal`；治理门对真实免费源如实 HOLD（source_latency，
  研究放宽仅用于 A/B）。**Ollama 故障 → DeepSeek → 词典回退的完整链在生产路径上走通。**

## 5. 回测表现（强化后）

**生产池演示（非 PIT 口径，58 只，2016-01~2026-08；本轮实跑）**：

| 指标 | 单书（`run_weekly_rotation.py`） | 套筒 40/30/20/10（采纳后生产入口 `run_sleeve_portfolio.py`） |
|---|---|---|
| 年化 | 6.47% | **6.75%** |
| 夏普 | 1.018 | **1.092** |
| 卡玛 | 0.800 | 0.789 |
| 最大回撤 | -8.08% | -8.56% |
| 月度胜率 | 59.4% | **61.7%** |
| 平均调仓换手 | 0.349 | **0.061** |
| 3 年窗回撤修复 | 113 日（进行中） | 113 日（进行中） |

**全池 PIT 采纳档证据（R12/R13，5478 只）**：单书 6.54%/1.01/-10.29% vs 套筒 6.55%/**1.09**/**-9.91%**（研究门
max_drawdown 翻转 PASS）、月度胜率 57.8%→60.2%、OOS 夏普 1.31→1.35——套筒为采纳后的生产默认风险层。

**LLM 通路（本轮新增）**：差异化确认（相关 0.53/符号分歧 60%），LLM 面板 w=0.20 信号窗口翻正
（+1.61% vs 词典 -2.59%），但全窗口影响 ±0.2pp 内，`alternative_signal_weight` 保持 0。

## 6. 验证与复现

```powershell
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY","User")
python Quant-4/reports/_iter/_llm_deepseek_score_real.py --out llm_scores_real_38.jsonl
python Quant-4/reports/_iter/_llm_vs_lex_deepseek.py
python Quant-4/reports/_iter/_llm_real_ab.py
python Quant-4/reports/_iter/_e2e_deepseek_pipeline.py        # E2E-DEEPSEEK: PASS
python Quant-4/tools/fetch_forum_guba.py --codes ... --max-pages 2
python Quant-4/run_sleeve_portfolio.py --pit                  # 生产入口（套筒采纳后）
python -m pytest Quant-4/tests -q -p no:cacheprovider         # 182 passed
```

## 7. 限制披露

- LLM 评分为云端 API（付费），批量评分受速率/预算约束；已按批落盘可续跑，全部记录可复核。
- 真实新闻/论坛语料仍受免费源近期窗口约束（新闻约 5 个月、论坛 2 页/只），全池历史复验需付费/授权源。
- 卡玛 0.66 与回撤修复期 181 日为结构性上限（R8-R12 已确认），套筒采纳不改变该上限，仅改善风险调整后
  收益与回撤深度。
