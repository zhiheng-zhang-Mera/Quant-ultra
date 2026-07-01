# -*- coding: utf-8 -*-
"""
step7/config.py
交易引擎成本与两融风控刚性常数配置箱
"""

# 基础交易费用
DEFAULT_HANDLING_FEE = 0.0000487  # 经手费
DEFAULT_MANAGEMENT_FEE = 0.00002  # 管理费/过户费
DEFAULT_STAMP_TAX = 0.0005        # 印花税 (卖出时计提)
DEFAULT_SLIPPAGE_BPS = 0.0002     # 集合竞价默认滑点基点因数

# 停牌与退市处理机制参数
DEFAULT_HALT_DAYS_LIMIT = 20      # 触发持续停牌减值的刚性交易日阈值
DEFAULT_IMPAIRMENT_RATE = 0.1     # 单次触发减值的惩罚因子系数 (10%)
DEFAULT_RESIDUAL_RATE = 0.0       # 退市资产清算残值率 (默认归零)

# 两融信用交易利息与强制平仓线
DEFAULT_MARGIN_INTEREST = 0.06    # 融资年化利率 (折算至每日 0.06/252)
DEFAULT_SHORT_INTEREST = 0.08     # 融券年化费率 (折算至每日 0.08/252)
MAINTENANCE_RATIO_LIMIT = 1.3     # 维持担保比例最低警戒线 (低于此值触发强制平仓)

# 资产池特异性整手单元映射规则
STAR_MARKET_LOT = 200             # 科创板 (688) 交易整手单元
MAIN_BOARD_LOT = 100              # 普通主板/创业板交易整手单元