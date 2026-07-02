# -*- coding: utf-8 -*-
"""
step7/config.py
交易引擎成本、静态冲击与个体全现货风控刚性常数配置箱
"""

# 基础现货交易规费 (完全剔除任何融资融券利息、两融服务等信用开销)
DEFAULT_HANDLING_FEE = 0.0000487  # 交易所经手费
DEFAULT_MANAGEMENT_FEE = 0.00002  # 登记过户费
DEFAULT_STAMP_TAX = 0.0005        # 印花税 (仅卖出时单边计提)
DEFAULT_SLIPPAGE_BPS = 0.0002     # 集合竞价基础默认滑点因数 (2bp)

# 停牌与退市处理机制参数
DEFAULT_HALT_DAYS_LIMIT = 20      # 触发持续停牌特殊减值的刚性交易日阈值
DEFAULT_IMPAIRMENT_RATE = 0.1     # 单次触发减值的惩罚因子系数 (10%)
DEFAULT_RESIDUAL_RATE = 0.0       # 退市资产清算残值率 (根据 Flow-Pro 刚性归零)

# 刚性降级静态冲击常数 (Flow-Pro 7.2 剥离动态自适应扰动)
STATIC_KAPPA_IMPACT = 0.001       # 刚性指定 10bp 冲击基准线
STATIC_ALPHA_IMPACT = 0.5         # 平方根市场冲击定律弹性系数

# 追高防御与个体持仓限制
GAP_UP_THRESHOLD = 0.03           # 开盘跳空高开涨幅限制硬红线 (3%)
MAX_SINGLE_TICKET_PROP = 0.10     # 个人单票绝对持仓权重上限 (10%)

# 资产池特异性整手单元映射规则
STAR_MARKET_LOT = 200             # 科创板 (688) 交易整手单元
MAIN_BOARD_LOT = 100              # 普通主板/创业板/科创板通用整手单元