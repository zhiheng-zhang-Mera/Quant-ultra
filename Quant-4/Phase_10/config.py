# Phase_10/config.py
import os
from datetime import datetime

# ================= 基础配置 =================
# 当前设定日期 (由系统上下文提供)
CURRENT_DATE = datetime.today().strftime('%Y-%m-%d')

# ================= 模型配置 =================
# 推荐使用对数据敏感且推理能力较强的开源模型，如 qwen2.5-coder 或 llama3.1
TARGET_MODEL = "qwen2.5:7b" 
OLLAMA_HOST = "http://localhost:11434"

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results") # 假设前9个Phase的数据保存在这里
REPORT_OUTPUT_DIR = os.path.join(BASE_DIR, "reports")

# 确保报告目录存在
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)