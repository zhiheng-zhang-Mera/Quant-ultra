"""Tiered report content for all pipeline phases.

Three detail levels are provided per phase so the same run can be read by a
financial newcomer, a self-taught investor, or a quantitative expert:

- layman:   plain language, one or two sentences, no jargon.
- beginner: key concepts explained, what the phase means for the portfolio.
- expert:   full technical description, inputs, algorithm, and caveats.
"""

PHASE_TIERS = {
    1: {
        "layman": (
            "zh: 这一步把全市场股票过滤了一遍，只留下交易量大、上市时间够长、数据完整的公司，并估算这套池子最多能容纳多少钱。",
            "en: This step filters the whole market down to stocks with enough trading volume, sufficient listing history, and complete data, then estimates how much money the pool can support.",
        ),
        "beginner": (
            "zh: 流动性筛选用 20 日平均成交额(ADV)衡量每只股票每天能买卖多少钱，剔除了成交过于清淡的标的；同时排除上市不足 20 天的次新股，避免价格异常。输出标的池、ADV 数据和理论容量上限(AUM limit)，供后续建模和仓位计算使用。",
            "en: Liquidity screening uses 20-day average daily value (ADV) to measure how much can be traded per day, removing illiquid names; it also excludes stocks listed less than 20 days to avoid price anomalies. Outputs the universe, ADV data, and a theoretical AUM limit used by later modeling and sizing.",
        ),
        "expert": (
            "zh: 输入为全市场日线 OHLCV。逐标的计算 ADV(20 日成交额均值)，以 min_adv_threshold=1e7 与 ipo_safety_days=20 为硬门槛；随后按 max_participation_rate、expected_turnover、max_single_stock_weight 反推单票容量并取最小值构造理论容量上限。同时构建 2010 年至今的存活矩阵 alive_mask，锁定每只标的的上市/退市时空边界，从源头消除生存者偏差。输出 assets、adv_data、alive_mask、theoretical_aum_limit。",
            "en: Inputs full-market daily OHLCV. Computes ADV (20-day mean turnover) per symbol and applies hard gates min_adv_threshold=1e7 and ipo_safety_days=20; capacity is back-solved per symbol from max_participation_rate, expected_turnover, max_single_stock_weight, and the minimum forms the AUM limit. Also builds the alive_mask survival matrix from 2010 onward to eliminate survivorship bias. Outputs assets, adv_data, alive_mask, theoretical_aum_limit.",
        ),
    },
    2: {
        "layman": (
            "zh: 把历史时间切成几段：模型只允许用前面一段学规律，用后面一段检查，绝不偷看未来。",
            "en: Time is split into segments so the model can only learn from the past and verify on later data, never peeking into the future.",
        ),
        "beginner": (
            "zh: 本阶段建立 Train-A/B1/B2、Validation、Test 五个时间窗口，并在训练与测试之间设置 embargo(禁运带)，防止相邻日期的信息泄漏。窗口边界与交易日历对齐，保证每段都是真实交易日序列。",
            "en: This phase creates Train-A/B1/B2, Validation, and Test windows plus an embargo gap between training and testing to prevent information leakage. Boundaries are aligned to real trading calendars.",
        ),
        "expert": (
            "zh: 输入 assets 与双市场交易日历。按固定日历边界切片(2010-01-04 起)，构造训练/验证/测试分区并附加 embargo_min 天禁运；输出 slices 字典与 embargo_window，供后续所有阶段按窗口取数。包含跨市场日期对齐表(sequence_token 双轨序号)以确保 A 股/美股样本时间轴可对齐。",
            "en: Uses assets and dual-market calendars. Slices fixed calendar boundaries from 2010-01-04 into train/validation/test partitions with an embargo_min day embargo; outputs the slices dict and embargo_window for all downstream phases, including a dual-market calendar alignment table.",
        ),
    },
    3: {
        "layman": (
            "zh: 为每只股票每天计算几十个基础指标(涨跌、波动、资金流向等)，并且只用当天已经知道的信息，杜绝“用未来数据作弊”。",
            "en: Computes dozens of basic indicators per stock per day (returns, volatility, money flow, etc.) using only information known at that day, preventing look-ahead cheating.",
        ),
        "beginner": (
            "zh: 生成两类特征：共享特征面板(所有标的统一计算)与私有特征面板(按市场分别计算)，并加入新闻/论坛情绪和资金池变化等另类信号。所有特征都遵守 Point-in-Time 原则：只使用发布当时可见的数据。",
            "en: Builds shared feature panels (computed uniformly for all symbols) and private panels (per market), plus alternative signals such as news/forum sentiment and capital-pool changes. All features obey Point-in-Time rules: only data visible at the time is used.",
        ),
        "expert": (
            "zh: 输入 PIT 行情与另类数据(news_input_path/forum_input_path，CSV/JSONL)。计算收益率、量价、波动率、资金池变化等白盒特征；可选本地 Ollama 大模型对最新有限文本做情绪增强(默认总数≤6 条、每标的≤2 条、单条≤300 字，超时 20s，失败自动保留字典结果)。输出 feature_panel_shared、feature_panel_private_a/us、online_regime_state、alternative_signals 及 alternative_data_evidence。",
            "en: Uses PIT quotes and alternative data (news/forum paths, CSV/JSONL). Computes returns, volume-price, volatility, and capital-pool white-box features; optionally enhances sentiment via a local Ollama model on a bounded set (total<=6, per-symbol<=2, <=300 chars, 20s timeout, dictionary fallback on failure). Outputs feature_panel_shared, feature_panel_private_a/us, online_regime_state, alternative_signals, and alternative_data_evidence.",
        ),
    },
    4: {
        "layman": (
            "zh: 给每只股票每天贴一个“未来涨还是跌”的标签，并给每个样本打分，让模型更关注真正重要的案例。",
            "en: Labels each stock-day as up or down in the future and weights each sample so the model focuses on genuinely important cases.",
        ),
        "beginner": (
            "zh: 构造两个标签：方向标签 y_clf(未来是否上涨)与收益标签 y_reg(未来收益率)，并基于事件去重和样本权重避免重复事件被过度加权。输出供模型训练使用。",
            "en: Creates two targets: direction label y_clf (will it rise?) and return label y_reg (how much?), with event deduplication and sample weights so repeated events are not overweighted. Outputs feed model training.",
        ),
        "expert": (
            "zh: 输入特征面板与切片。以未来 holding_period 天收益构造 y_reg，以符号构造 y_clf；通过事件去重(相同标的相邻窗口)与 sample_weights 平衡样本分布；检查借券可用性(borrow_manager)输出 borrowable_stocks。标签严格基于 PIT 数据，避免前视偏差。",
            "en: Uses features and slices. Builds y_reg from forward holding_period returns and y_clf from their sign; applies event deduplication and sample_weights to balance distributions; checks short-borrow availability via borrow_manager. Labels are strictly PIT, avoiding look-ahead.",
        ),
    },
    5: {
        "layman": (
            "zh: 用历史数据训练一个“预测器”，学习什么样的特征组合之后股价更容易上涨，并检查预测是否稳定可靠。",
            "en: Trains a predictor on historical data to learn which feature combinations lead to higher prices, and checks that predictions are stable and reliable.",
        ),
        "beginner": (
            "zh: 使用 LightGBM 梯度提升树训练方向分类器和分位数回归模型，做滚动交叉验证、特征筛选和校准(校正预测置信度)。输出模型、选中的特征、以及验证证据，供下一阶段生成仓位。",
            "en: Trains a LightGBM direction classifier and quantile regression models with rolling cross-validation, feature selection, and calibration. Outputs models, selected features, and validation evidence for position sizing.",
        ),
        "expert": (
            "zh: 输入特征面板、标签与日历对齐。执行 walk-forward CV(step_5_1)、分数阶特征生成与 VIF 过滤(step_5_2_3)、LightGBM 拟合(step_5_4，含 MMD 域适应权重)与级联校准(step_5_5)。输出 direction_classifier、quantile_models、gamma_star、q_error_threshold_dict、selected_features、fractional_features_cube、num_trials 与 trial_evidence。",
            "en: Uses features, labels, and calendar alignment. Runs walk-forward CV, fractional feature generation with VIF filtering, LightGBM fitting (with MMD domain-adaptation weights), and cascade calibration. Outputs direction_classifier, quantile_models, gamma_star, q_error_threshold_dict, selected_features, fractional_features_cube, num_trials, and trial_evidence.",
        ),
    },
    6: {
        "layman": (
            "zh: 根据模型的预测和风险控制规则，算出每天每只股票该买多少、留多少现金，保证不会把鸡蛋都放在一个篮子里。",
            "en: Turns model predictions and risk rules into daily target weights per stock and cash, making sure risk is not concentrated in one basket.",
        ),
        "beginner": (
            "zh: 在现金缓冲、单票上限、行业上限、换手率和成本约束下，用凸优化求解每日目标权重；融合 Black-Litterman 先验与风险厌恶参数，输出 daily_weights 供回测执行。",
            "en: Solves daily target weights with convex optimization under cash buffer, per-stock caps, sector caps, turnover, and cost constraints; blends Black-Litterman priors and risk aversion. Outputs daily_weights for the backtest.",
        ),
        "expert": (
            "zh: 输入方向/分位数预测与特征。构建协方差(稳健估计)、Black-Litterman 收益先验、条件方向掩码；以 CVXPY 求解二次规划，约束含 cash_buffer_weight、minimum_invested_weight、max_daily_turnover、sector_limit、max_single_stock_weight 及交易成本系数。输出 daily_weights、daily_intervals、daily_adv20。",
            "en: Uses direction/quantile predictions and features. Builds robust covariance, Black-Litterman return priors, and conditional direction masks; solves a CVXPY quadratic program under cash buffer, minimum invested, max daily turnover, sector, single-name, and cost constraints. Outputs daily_weights, daily_intervals, and daily_adv20.",
        ),
    },
    7: {
        "layman": (
            "zh: 模拟真实交易：按 100 股一手的规则、涨跌停、手续费和印花税，把每天的仓位变成一条真实的资金曲线。",
            "en: Simulates real trading with board lots, price limits, commissions, and stamp tax, turning daily weights into a realistic equity curve.",
        ),
        "beginner": (
            "zh: 用有限状态机(FSM)逐日执行：检查是否可交易、是否停牌、涨跌停能否成交，按整手成交并扣除佣金、印花税、滑点。输出每日净值 NAV、收益率和违规记录。",
            "en: Executes day by day with a finite state machine: checks tradability, halts, and limit conditions, fills in board lots, and deducts commissions, stamp tax, and slippage. Outputs daily NAV, returns, and violation records.",
        ),
        "expert": (
            "zh: 输入 daily_weights、日历对齐与交易状态映射。FSM 状态含空仓/建仓/持有/止盈止损，逐日模拟整手买入、涨跌停约束(主板10%/创业科创20%/ST5%)、手续费(commission_rate、minimum_commission、exchange_fee、stamp_tax)与滑点。输出 daily_nav、daily_returns、final_nav、nav_history、violations。",
            "en: Uses daily_weights, calendar alignment, and trading-status maps. The FSM states cover flat/open/hold/take-profit-stop-loss, simulating board-lot fills, price-limit constraints (10% main board / 20% ChiNext-STAR / 5% ST), fees (commission, minimum, exchange, stamp tax), and slippage. Outputs daily_nav, daily_returns, final_nav, nav_history, and violations.",
        ),
    },
    8: {
        "layman": (
            "zh: 给回测结果做全面体检：数据覆盖够不够、容量会不会踩踏、极端行情下会不会爆仓，全部检查通过才算合格。",
            "en: Gives the backtest a full health check: data coverage, capacity impact, and extreme-market survival. Only a fully passed audit is accepted.",
        ),
        "beginner": (
            "zh: 检查覆盖率和流动性冲击(大单会不会推高价格)、压力测试(股灾/暴涨暴跌场景)、以及统计稳健性(DSR 修正多重检验)。输出审计结论 audit_passed 与摘要，任何红线失败都会阻止后续执行。",
            "en: Checks coverage and liquidity impact (would large orders move prices), stress tests (crash and volatility scenarios), and statistical robustness (DSR-corrected multiple testing). Outputs audit_passed and audit_summary; any red-line failure blocks later execution.",
        ),
        "expert": (
            "zh: 输入 daily_weights、daily_adv20、daily_nav、violations。执行覆盖率审计、容量/参与率压力测试、DSR(Deflated Sharpe Ratio)检验与极端情景压力测试；汇总 audit_passed、audit_summary，红线失败即 HOLD_FOR_REVIEW。",
            "en: Uses daily_weights, daily_adv20, daily_nav, and violations. Runs coverage audit, capacity/participation stress tests, DSR (Deflated Sharpe Ratio), and extreme-scenario stress tests; summarizes audit_passed and audit_summary, with red-line failures triggering HOLD_FOR_REVIEW.",
        ),
    },
    9: {
        "layman": (
            "zh: 把“计划要买的仓位”和“实际成交的仓位”做对比，检查有没有偏差，并监控模型是否过时。",
            "en: Compares planned versus actually filled positions, checks for drift, and monitors whether the model is going stale.",
        ),
        "beginner": (
            "zh: 计算目标仓位与实际仓位的平均绝对误差(MAE)，监控特征分布漂移(PSI)和拥挤度；异常时激活交易门禁或停机开关。输出对账结论 recon_passed 与告警报告。",
            "en: Computes mean absolute error between target and executed positions, monitors feature distribution drift (PSI) and crowding; activates trading gates or kill switches on anomalies. Outputs recon_passed and alert reports.",
        ),
        "expert": (
            "zh: 输入 nav_history、audit_summary。执行影子对账(shadow_reconciliation，MAE 阈值 mae_threshold)、PSI 连续越限检测、拥挤度上限强制与分层更新(tiered_updater)。输出 reconciliation_mae、recon_passed、trading_halted、kill_switch_report、psi_consecutive_breaches、enforce_crowded_allocation_cap。",
            "en: Uses nav_history and audit_summary. Runs shadow reconciliation (MAE vs mae_threshold), consecutive PSI breach detection, crowding-cap enforcement, and tiered updates. Outputs reconciliation_mae, recon_passed, trading_halted, kill_switch_report, psi_consecutive_breaches, and enforce_crowded_allocation_cap.",
        ),
    },
    10: {
        "layman": (
            "zh: 把前面所有检查结果汇总给“投资委员会”，由它决定：继续、暂停观察、还是需要人工复核。",
            "en: Summarizes all previous checks for an investment committee, which decides: proceed, observe, or require manual review.",
        ),
        "beginner": (
            "zh: 汇总审计、对账、净值等证据，输出 CIO 决策：全部通过则为 ELIGIBLE_FOR_PHASE_11，任何缺失或失败则为 HOLD_FOR_REVIEW。同时登记需要人工审批的参数提案。",
            "en: Aggregates audit, reconciliation, and NAV evidence into a CIO decision: ELIGIBLE_FOR_PHASE_11 when all pass, otherwise HOLD_FOR_REVIEW. Parameter proposals are registered for human approval.",
        ),
        "expert": (
            "zh: 输入 audit_passed、recon_passed、audit_summary、final_nav 等。校验证据完整性(缺项即失败)与治理门禁(审计+对账必须通过)，输出 cio_decision、cio_evidence、cio_report_path、parameter_proposal_status。治理失败时 Phase 11 只能进入 OBSERVATION_ONLY。",
            "en: Uses audit_passed, recon_passed, audit_summary, final_nav, etc. Validates evidence completeness and governance gates (audit and reconciliation must pass), outputting cio_decision, cio_evidence, cio_report_path, and parameter_proposal_status. Failed governance limits Phase 11 to OBSERVATION_ONLY.",
        ),
    },
    11: {
        "layman": (
            "zh: 把最终结论整理成一份能看懂的投资观察清单：哪些股票值得关注、建议什么价位关注，并明确说明这只是分析、不是下单指令。",
            "en: Turns final conclusions into a readable observation list: which stocks to watch and at what price levels, with a clear statement that this is analysis, not an order.",
        ),
        "beginner": (
            "zh: 基于四阶段主导方法(选股/入场/持有/止盈止损)生成候选列表，包含建议仓位、入场区间、止盈止损；治理未通过时仅观察、禁止执行。输出 Markdown 与 CSV 报告。",
            "en: Generates candidates from four dominant methods (selection/entry/holding/take-profit-stop-loss) with suggested weight, entry range, and stops; when governance fails, results are observation-only. Outputs Markdown and CSV reports.",
        ),
        "expert": (
            "zh: 输入 phase10_ready、daily_weights、data_manager。经 investment_advisor 生成候选(advisory_mode=ANALYSIS_ONLY 或 OBSERVATION_ONLY)，附主导方法链与价格区间；非交互模式只写报告。输出 investment_candidates、phase11_report_path、phase11_csv_path、phase11_ready。",
            "en: Uses phase10_ready, daily_weights, and data_manager. Builds candidates via investment_advisor (advisory_mode=ANALYSIS_ONLY or OBSERVATION_ONLY) with the dominant-method chain and price bands; non-interactive mode writes reports only. Outputs investment_candidates, phase11_report_path, phase11_csv_path, and phase11_ready.",
        ),
    },
}


def tier_text(phase_number: int, tier: str) -> dict:
    """Return {'zh': ..., 'en': ...} for one phase/tier, with safe defaults."""
    entries = PHASE_TIERS.get(int(phase_number), {}).get(tier)
    if not entries:
        return {"zh": "暂无此级别的详细说明。", "en": "No detailed description available for this level."}
    zh, en = entries
    return {"zh": zh, "en": en}
