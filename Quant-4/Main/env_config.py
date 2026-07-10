import subprocess
from pathlib import Path
import logging

logger = logging.getLogger("EnvConfig")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.info("[OP] Initialize Base Path Topologies | [SOURCE] OS Filesystem Subsystem | [RESULT] Root path anchored at: %s | [SIGNIFICANCE] Establishes deterministic storage boundaries for cache alignment", PROJECT_ROOT)
logger.info("[操作] 初始化基础路径拓扑 | [来源] 操作系统文件系统 | [结果] 根路径锚定于: %s | [意义] 确立缓存对齐的确定性存储边界")

def get_git_hash() -> str:
    try:
        res = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        logger.debug("[OP] Fetch Git Commit Hash | [SOURCE] Local .git Metadata Directory | [RESULT] Active Commit: %s | [SIGNIFICANCE] Guarantees runtime logic provenance tracking", res)
        logger.debug("[操作] 获取Git提交哈希 | [来源] 本地.git元数据目录 | [结果] 当前提交: %s | [意义] 保证运行时逻辑具备可追溯的血缘记录")
        return res
    except Exception:
        logger.warning("[OP] Fetch Git Commit Hash | [SOURCE] Local Shell Exec | [RESULT] Command Failed or Repository Absent | [SIGNIFICANCE] Sandbox environment detected; missing strict version control pinning", exc_info=True)
        logger.warning("[操作] 获取Git提交哈希 | [来源] 本地Shell执行 | [结果] 命令失败或代码库不存在 | [意义] 检测到沙箱环境；缺失严格的版本控制锁定")
        return "NO_GIT"

def get_git_status() -> str:
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        res = "CLEAN" if not status else "DIRTY"
        logger.debug("[OP] Verify Workspace Isolation | [SOURCE] Git Porcelain Diff | [RESULT] Workspace state is %s | [SIGNIFICANCE] Guards production against uncommitted experimental modifications", res)
        logger.debug("[操作] 验证工作区隔离状态 | [来源] Git瓷器差异比对 | [结果] 工作区状态为 %s | [意义] 保护生产环境免受未提交的实验性代码污染")
        return res
    except Exception:
        logger.warning("[OP] Verify Workspace Isolation | [SOURCE] Git Shell Exception | [RESULT] Status unknown | [SIGNIFICANCE] Version control bridge broken; assuming permissive validation pass")
        logger.warning("[操作] 验证工作区隔离状态 | [来源] Git异常捕获 | [结果] 状态未知 | [意义] 版本控制桥接断开；默认允许宽松验证通过")
        return "UNKNOWN"