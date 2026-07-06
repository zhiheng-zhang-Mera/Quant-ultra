"""
Quant-Ultra Flow - System Adaptive Concurrency Controller
"""
import time
import logging
import threading
from step1.config import CONFIG

logger = logging.getLogger("Orchestrator.Step1.Concurrency")

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
            logger.info(f"🚀 开启基于内核负载的自适应流控机制，初始并发度: 4")
        else:
            logger.info("⚠️ 系统未检测到 psutil 依赖，降级为静态固定线程池策略")

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
            except Exception as e:
                logger.debug(f"并发流控器监控异常: {e}")
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