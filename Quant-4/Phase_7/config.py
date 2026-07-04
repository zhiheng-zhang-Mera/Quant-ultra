# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Backtest Engine Configurations & Fee Schedules
"""

# 基础多头现货交易规费（完全剥离任何融资融券利息或衍生品杠杆信用开销）
DEFAULT_HANDLING_FEE = 0.0000487   # 交易所经手费
DEFAULT_MANAGEMENT_FEE = 0.00002   # 登记过户费
DEFAULT_STAMP_TAX = 0.0005         # 印花税 (仅卖出时单边计提)
DEFAULT_SLIPPAGE_BPS = 0.0002      # 集合竞价基础默认滑点因数 (2bp)

# 停牌与退市极限惩罚处理机制参数
DEFAULT_HALT_DAYS_LIMIT = 20       # 触发持续停牌特殊资产减值的刚性交易日阈值
DEFAULT_IMPAIRMENT_RATE = 0.1      # 单次触发减值的惩罚因子系数 (减值 10%)
DEFAULT_RESIDUAL_RATE = 0.0        # 退市资产清算残值率 (根据契约刚性归零)

# 调仓冲击函数参数 (平方根冲击定律)
STATIC_KAPPA_IMPACT = 0.001        # 10bp 冲击基准线
STATIC_ALPHA_IMPACT = 0.5          # 弹性系数

# 追高防御与个体持仓限制
GAP_UP_THRESHOLD = 0.07            # 隔夜高开追高防御阈值 (7%)
MAX_SINGLE_TICKET_PROP = 0.10      # 个人单票绝对持仓权重上限 (10%)

# 资产池特异性整手单元映射规则
STAR_MARKET_LOT = 200              # 科创板 (688) 交易整手单元
MAIN_BOARD_LOT = 100               # 普通主板/创业板/科创板通用整手单元

# 回测账户初始资金（若上下文未提供）
INITIAL_CASH = 10_000_000.0