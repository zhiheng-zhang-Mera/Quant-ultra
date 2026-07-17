'''
附加增强模块：Multi-Agent CIO 决策输出结构化硬校验 (Phase 10)
设计模式：容错校验钩子 + 双轨汇报 + 闭环参数自动修正
使用方法：提供增强的 prompt 生成、解析及参数应用函数，供主流程调用。
'''
import json
import logging
import re
import os
import ast
from typing import Dict, Any, Tuple

# -------------------- 原有装饰器（保留） --------------------
def llm_decision_parser_hook(llm_call_func):
    def wrapper(*args, **kwargs):
        logging.info("增强注入：校验多智能体联合 CIO 的 JSON 输出并设定底线阈值...")
        raw_response = llm_call_func(*args, **kwargs)
        try:
            decision = json.loads(raw_response)
            if 'action' not in decision or 'confidence' not in decision:
                raise ValueError("JSON响应体缺失操作动作或置信度核心字段。")
            if decision['confidence'] < 0.6:
                logging.warning("增强介入：CIO 最终决断置信度过低，降级为保守 HOLD 策略以控制下行风险。")
                decision['action'] = 'HOLD'
            return decision
        except Exception as e:
            logging.error(f"增强介入失败，遭遇幻觉或格式错误: {e}。执行默认安全回退...")
            return {"action": "HOLD", "confidence": 1.0, "reason": "Structural parser fallback"}
    return wrapper


# -------------------- 新增：双轨汇报支持 --------------------
def generate_analyst_prompt_dual_track(data_payload: Dict[str, Any], model_name: str) -> str:
    """
    生成分析师提示词，强制输出 JSON（结构数据） + 独立逻辑链文本（双轨）。
    返回的 prompt 要求模型输出一个 JSON 对象，其中包含：
      - Health_Status, Risk_Assessment, MLOps_Alerts, Trend_Verdict (结构化)
      - Logic_Chain: 不超过150字的浓缩因果逻辑链
    """
    json_data = json.dumps(data_payload, indent=2, ensure_ascii=False)
    return f"""你是一名严格的量化分析师（代号：{model_name}）。请基于以下系统运行数据，给出诊断。
你必须且只能输出一个合法的 JSON 字典，不能有任何 Markdown 格式或额外解释文本。

【数据输入】：
{json_data}

【输出格式要求（必须完全遵循此 JSON 结构）】：
{{
  "Health_Status": "Healthy/Warning/Critical",
  "Risk_Assessment": "描述回撤和压力测试的最大隐患",
  "MLOps_Alerts": "针对实盘漂移和滑点的应对动作",
  "Trend_Verdict": "看多/看空/震荡及调仓建议",
  "Logic_Chain": "在此提供浓缩逻辑链（不超过150字），说明核心因果推导，例如：由于20日均线斜率转负且波动率放大，系统性Beta存在转弱风险。"
}}
仅返回上述 JSON 数据！
"""


def parse_analyst_response(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """
    解析分析师原始输出，返回 (结构化字典, 逻辑链文本)。
    若解析失败，返回 ({'Error': '...'}, '')
    """
    try:
        # 尝试提取 JSON
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(raw_text)
        logic_chain = parsed.get("Logic_Chain", "")
        # 移除逻辑链字段，保留其余结构化数据
        structured = {k: v for k, v in parsed.items() if k != "Logic_Chain"}
        return structured, logic_chain
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"解析分析师响应失败: {e}")
        return {"Error": "解析失败", "Raw": raw_text[:200]}, ""


# -------------------- 新增：CIO 闭环参数修正支持 --------------------
def generate_judge_prompt_with_params(raw_data: Dict[str, Any], analysts_reports: Dict[str, Any]) -> str:
    """
    生成裁判模型提示词，要求输出包含文本报告和参数调整指令的 JSON。
    输出格式：
    {
      "report": "完整的 Markdown 格式报告文本...",
      "parameter_adjustments": {
         "HRP_shrinkage_alpha": 0.85,
         "max_sector_exposure": 0.15
      }
    }
    """
    reports_str = json.dumps(analysts_reports, indent=2, ensure_ascii=False)
    raw_str = json.dumps(raw_data, indent=2, ensure_ascii=False)
    return f"""你是量化机构的 CIO (首席投资官)。你今天需要给出最终评估并输出可执行的参数调整指令。

【原始客观数据】：
{raw_str}

【3位分析师的独立评估（JSON格式）】：
{reports_str}

【你的任务】：
1. 撰写一份专业的 Markdown 格式报告，内容包含执行摘要、多模型共识与分歧剖析、模型与因子终评、风控与滑点裁决、下一阶段优化指令。
2. 同时，基于分析师的警报和你的专业判断，生成具体的超参数调整指令（parameter_adjustments），用于修正底层量化策略。
   - 仅当存在明确的警示信号（如风险指标恶化、因子失效）时才给出调整，否则可输出空对象 {{}}。
   - 参数名应与底层 config.py 中的变量名完全一致。
3. 你必须且只能输出一个合法的 JSON 对象，不能包含任何其他文本。

【输出格式】：
{{
  "report": "完整的 Markdown 报告文本",
  "parameter_adjustments": {{
    "参数名1": 数值,
    "参数名2": 数值
  }}
}}
仅返回上述 JSON！
"""


def parse_judge_response(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    解析裁判模型输出，返回 (报告文本, 参数字典)。
    若解析失败，返回 ("", {{}}) 并记录错误。
    """
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(raw_text)
        report = parsed.get("report", "")
        params = parsed.get("parameter_adjustments", {})
        if not isinstance(params, dict):
            logging.warning("parameter_adjustments 字段不是字典，将忽略")
            params = {}
        return report, params
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"解析裁判响应失败: {e}")
        return "", {}


# -------------------- 新增：参数自动写入 config 的闭环函数 --------------------
def apply_parameter_adjustments(params: Dict[str, Any], config_file_path: str) -> bool:
    """
    将参数字典写入指定的 config.py 文件。
    支持两种模式：
      1. 如果文件存在且包含同名变量，则更新其值（使用 ast 解析并重写）。
      2. 否则在文件末尾追加赋值语句。
    返回是否成功。
    """
    if not params:
        logging.info("无参数调整指令，跳过写入。")
        return True

    if not os.path.exists(config_file_path):
        logging.warning(f"目标 config 文件不存在: {config_file_path}，无法写入参数。")
        return False

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用 ast 解析文件，提取所有赋值语句
        tree = ast.parse(content)
        assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node

        # 构建新内容：逐行处理，替换已有变量的赋值，否则追加
        lines = content.splitlines()
        new_lines = []
        written_vars = set()
        for line in lines:
            # 简单检测是否包含变量名赋值，此处采用保守策略：仅当行以变量名开头且含有 '=' 时认为匹配
            # 更精确可用 ast 匹配，但为简化，我们使用正则粗略判断
            matched = False
            for var_name in params.keys():
                # 匹配形如 "var_name =" 或 "var_name= " 的行，忽略注释
                pattern = rf'^\s*{var_name}\s*='
                if re.search(pattern, line):
                    # 替换该行
                    new_line = f"{var_name} = {repr(params[var_name])}"
                    # 保留注释（如果原行有注释，可保留在末尾，但这里简化处理）
                    # 检查原行是否有注释
                    comment = re.search(r'#.*$', line)
                    if comment:
                        new_line += f"  {comment.group(0)}"
                    new_lines.append(new_line)
                    written_vars.add(var_name)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)

        # 追加未匹配的变量
        for var_name, val in params.items():
            if var_name not in written_vars:
                new_lines.append(f"{var_name} = {repr(val)}")

        # 写回文件
        with open(config_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        logging.info(f"参数调整已写入 {config_file_path}: {params}")
        return True
    except Exception as e:
        logging.error(f"写入参数到 config 文件失败: {e}")
        return False


# -------------------- 统一增强入口（可选） --------------------
def enhance_cio_pipeline(original_report_func):
    """
    装饰器：包装原有的 run_multi_agent_analysis 函数，
    使其在生成报告后自动提取参数并写入 config。
    注意：这需要原有函数返回报告路径或内容，此装饰器仅为示例。
    """
    def wrapper(*args, **kwargs):
        # 调用原函数，假设它返回最终报告文本或路径
        result = original_report_func(*args, **kwargs)
        # 这里需要额外逻辑来获取裁判原始输出，但原函数未提供，故实际使用时需配合修改。
        logging.info("增强装饰器：建议在主流程中显式调用 parse_judge_response 和 apply_parameter_adjustments。")
        return result
    return wrapper