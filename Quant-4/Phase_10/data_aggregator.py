# Phase_10/data_aggregator.py
import os
import json

def collect_pipeline_data(results_dir):
    """
    模拟从 Phase 1-9 的输出目录中抓取关键数据。
    实际应用中，您应该读取具体的 JSON/CSV 文件（如 step7的回测结果, step9的监控日志等）。
    """
    print("[DataAggregator] 正在读取 Phase 1~9 运行结果...")
    
    # 这里用结构化 Mock 数据代替读取文件的逻辑，展示高信息密度的降维文本
    pipeline_data = {
        "Phase_1_to_3_DataFoundation": {
            "Total_Assets": 3500,
            "Date_Range": "2015-01-01 to 2026-07-14",
            "Missing_Data_Imputation_Rate": "0.4%"
        },
        "Phase_4_to_5_Modeling": {
            "Model_Type": "LGBM + XGBoost Ensemble",
            "Cross_Validation_Sharpe": 2.15,
            "Top_3_Factors": ["Momentum_1M", "Volatility_60D", "Turnover_Ratio_20D"]
        },
        "Phase_6_to_8_Backtest_Risk": {
            "Annualized_Return": "18.4%",
            "Max_Drawdown": "-9.2%",
            "Information_Ratio": 1.82,
            "Win_Rate": "54.3%",
            "Stress_Test_2020_Crash": "Passed (Drawdown -12%)"
        },
        "Phase_9_Live_MLOps": {
            "Model_Drift_Status": "Normal",
            "Data_Staleness": "0 hours",
            "Live_Slippage_Avg": "2.1 bps"
        }
    }
    
    return pipeline_data