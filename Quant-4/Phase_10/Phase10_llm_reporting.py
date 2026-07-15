# Phase_10/Phase10_llm_reporting.py
import os
import json
from datetime import datetime
from .config import CURRENT_DATE, TARGET_MODEL, REPORT_OUTPUT_DIR, RESULTS_DIR
from .env_setup import setup_all
from .data_aggregator import collect_pipeline_data

def generate_prompt(data_payload):
    """构建输入给大模型的 Prompt"""
    
    json_data = json.dumps(data_payload, indent=4, ensure_ascii=False)
    
    prompt = f"""你是一个顶级的量化投资分析师与风险合规官。
今天是 {CURRENT_DATE}。

我刚刚运行了我们量化系统（Quant-Ultra）的 Phase 1 到 Phase 9。
请你根据以下提供的 JSON 格式的流水线运行数据，撰写一份专业的量化模型诊断与趋势分析报告。

【流水线运行数据】：
{json_data}

【输出要求】：
1. 报告必须使用 Markdown 格式。
2. 包含以下核心章节：
   - 摘要 (Executive Summary)：一句话总结当前策略的健康度。
   - 因子与模型评估 (Factor & Model Evaluation)：基于特征和CV表现进行分析。
   - 风险与回测表现 (Risk & Backtest Performance)：解读收益回撤比及压力测试结果。
   - 实盘监控 (Live MLOps Monitor)：针对模型漂移和实盘滑点提供运维建议。
   - 趋势展望：基于当前日期({CURRENT_DATE})，给出接下来的参数调整或市场应对建议。
3. 语言风格：客观、严谨、尖锐，切勿产生虚假数据（不要幻觉）。数据基于上述提供的 JSON。
"""
    return prompt

def run_llm_analysis():
    # 1. 前置依赖自检与拉取
    setup_all()
    
    # 延迟导入 ollama (确保在 env_setup 成功后)
    import ollama
    
    # 2. 收集数据
    pipeline_data = collect_pipeline_data(RESULTS_DIR)
    
    # 3. 构造提示词
    prompt = generate_prompt(pipeline_data)
    print(f"\n[Phase-10] 正在调用本地模型 {TARGET_MODEL} 进行深度解读分析...")
    
    # 4. 调用本地 Ollama
    try:
        response = ollama.chat(
            model=TARGET_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        report_content = response['message']['content']
    except Exception as e:
        print(f"[错误] LLM 生成失败: {e}")
        return

    # 5. 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"LLM_Quant_Report_{CURRENT_DATE}_{timestamp}.md"
    report_path = os.path.join(REPORT_OUTPUT_DIR, report_filename)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\n[Phase-10 成功] 分析报告已生成，保存在:\n => {report_path}")

if __name__ == "__main__":
    run_llm_analysis()