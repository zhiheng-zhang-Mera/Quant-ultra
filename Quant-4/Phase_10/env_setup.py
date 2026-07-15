# Phase_10/env_setup.py
import os
import sys
import shutil
import subprocess
import time
from .config import TARGET_MODEL

def ensure_python_package():
    """检测并安装 Python 端的 ollama 库"""
    try:
        import ollama
        print("[EnvSetup] Python包 'ollama' 已安装。")
    except ImportError:
        print("[EnvSetup] 未找到 'ollama' Python包，正在自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
        print("[EnvSetup] 'ollama' 安装完成。")

def ensure_ollama_framework():
    """检测系统中是否安装了 Ollama 客户端"""
    if shutil.which("ollama") is None:
        print("[EnvSetup] 未检测到系统级别的 Ollama 框架。")
        if sys.platform == "linux" or sys.platform == "darwin":
            print("[EnvSetup] 正在尝试通过官方脚本自动安装 Ollama (Linux/Mac)...")
            try:
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
                print("[EnvSetup] Ollama 框架安装成功！")
            except subprocess.CalledProcessError:
                raise RuntimeError("[EnvSetup 错误] 自动安装失败，可能需要 sudo 权限。请手动运行: curl -fsSL https://ollama.com/install.sh | sh")
        else:
            raise RuntimeError("[EnvSetup 错误] Windows 系统无法一键静默安装，请前往 https://ollama.com/ 手动下载并安装。")
    else:
        print("[EnvSetup] Ollama 框架已就绪。")

def ensure_model_pulled():
    """检测并拉取指定的本地大模型"""
    import ollama
    
    # 尝试连接服务，如果未启动则提示
    try:
        models_info = ollama.list()
    except Exception as e:
        raise RuntimeError(f"[EnvSetup 错误] 无法连接到 Ollama 服务，请确保服务已启动 (终端运行 `ollama serve`): {e}")

    # 解析已安装的模型名称
    available_models = [m.get('name') if isinstance(m, dict) else m.model for m in models_info.get('models', [])]
    # 兼容带标签或不带标签的情况
    target_match = any(TARGET_MODEL in m for m in available_models)

    if not target_match:
        print(f"[EnvSetup] 模型 '{TARGET_MODEL}' 未存在本地，正在拉取 (此过程根据网络可能需要数分钟)...")
        # 启用流式拉取以显示进度
        for progress in ollama.pull(TARGET_MODEL, stream=True):
            status = progress.get('status', '')
            print(f"\r拉取状态: {status}", end='', flush=True)
        print(f"\n[EnvSetup] 模型 '{TARGET_MODEL}' 拉取完成！")
    else:
        print(f"[EnvSetup] 模型 '{TARGET_MODEL}' 已存在。")

def setup_all():
    print("=== 开始 Phase-10 环境自检 ===")
    ensure_python_package()
    ensure_ollama_framework()
    ensure_model_pulled()
    print("=== 环境自检完成，一切就绪 ===\n")