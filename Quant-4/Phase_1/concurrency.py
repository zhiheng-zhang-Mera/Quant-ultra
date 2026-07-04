"""
Quant-Ultra Flow - System Adaptive Concurrency Controller
"""
import time
import logging
import threading
from Phase_1.config import CONFIG

logger = logging.getLogger("Orchestrator.Phase1.Concurrency")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class AdaptiveConcurrencyLimiter:
    def __init__(self):
        self.min = CONFIG["ADAPTIVE_MIN_WORKERS"]
        self.max = CONFIG["ADAPTIVE_MAX_WORKERS"]
        self.target_cpu = CONFIG["TARGET_CPU_UTIL"]
        self.check_interval = CONFIG["CHECK_INTERVAL"]
        self.error_threshold = CONFIG["ERROR_RATE_THRESHOLD"]

        self.current_limit = 4
        self.running = 0
        self.error_count = 0
        self.total_count = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._stop = False
        self._monitor_thread = None

        if HAS_PSUTIL:
            self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
            self._monitor_thread.start()
            logger.info("[OP] Start Adaptive Concurrency Controller | [SOURCE] OS Kernel Core Telemetry | [RESULT] Worker Limit Initialized to 4 | [SIGNIFICANCE] Protects distributed downpour tasks against remote cluster bans")
            logger.info("[操作] 启动自适应并发限制器 | [来源] 操作系统内核遥测 | [结果] 初始并发度设定为 4 | [意义] 保护分布式高密度抓取任务免遭远程数据源硬熔断封锁")
        else:
            logger.warning("[OP] Downgrade Concurrency Strategy | [SOURCE] Library Dependency Checker | [RESULT] psutil absent; using static fallback pool | [SIGNIFICANCE] Runs framework with constant threading thresholds")
            logger.warning("[操作] 降级并发控制策略 | [来源] 环境依赖检查器 | [结果] 缺少 psutil 库；切往静态固定池 | [意义] 采用守恒常量线程阈值维持流水线运转")

    def _monitor(self):
        while not self._stop:
            try:
                cpu = psutil.cpu_percent(interval=0.5) / 100.0
                with self.lock:
                    error_rate = self.error_count / max(1, self.total_count)
                
                if error_rate > self.error_threshold:
                    new_limit = max(self.min, self.current_limit - 2)
                elif cpu < self.target_cpu * 0.9:
                    new_limit = min(self.max, self.current_limit + 1)
                elif cpu > self.target_cpu * 1.1:
                    new_limit = max(self.min, self.current_limit - 1)
                else:
                    new_limit = self.current_limit

                if new_limit != self.current_limit:
                    with self.cond:
                        self.current_limit = new_limit
                        self.cond.notify_all()
                    logger.info("[OP] Modulate Dynamic Thread Bounds | [SOURCE] Hardware Utilization Resampling | [RESULT] New Active Limit: %s | [SIGNIFICANCE] Balances parsing loads and network safety margins dynamically", new_limit)
                    logger.info("[操作] 调节动态线程边界 | [来源] 硬件利用率重采样 | [结果] 新并发限制上限: %s | [意义] 动态平衡计算解析负载与网络安全边界")
            except Exception as e:
                logger.debug(f"Concurrency inspector loop anomaly: {e}")
            time.sleep(self.check_interval)

    def acquire(self) -> bool:
        with self.cond:
            while self.running >= self.current_limit and not self._stop:
                self.cond.wait()
            if self._stop: return False
            self.running += 1
            return True

    def release(self, success: bool = True):
        with self.cond:
            self.running -= 1
            self.total_count += 1
            if not success: self.error_count += 1
            self.cond.notify()

    def stop(self):
        with self.cond:
            self._stop = True
            self.cond.notify_all()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)