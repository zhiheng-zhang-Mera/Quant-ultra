"""Hardware-aware distributed task planning with safe single-node fallback."""
from __future__ import annotations
import os, platform, shutil, socket, subprocess
import ctypes
from dataclasses import dataclass, asdict
from pathlib import Path
try:
    import psutil
except ImportError:
    psutil = None

@dataclass
class HardwareProfile:
    logical_cpu: int
    physical_cpu: int
    available_memory_gb: float
    total_memory_gb: float
    workspace_free_gb: float
    gpu_devices: list
    platform: str

def _reachable(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except OSError: return False

def _gpu_devices():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=3, text=True)
        return [{"name": line.rsplit(",", 1)[0].strip(), "memory_mb": int(line.rsplit(",", 1)[1])} for line in output.splitlines() if "," in line]
    except (OSError, subprocess.SubprocessError, ValueError): return []

def inspect_hardware(workspace):
    logical = max(os.cpu_count() or 1, 1); physical = psutil.cpu_count(logical=False) if psutil else logical
    if psutil:
        memory = psutil.virtual_memory(); available, total = memory.available, memory.total
    elif os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong), ("extended_available", ctypes.c_ulonglong)]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)): raise OSError("GlobalMemoryStatusEx failed")
        available, total = status.available, status.total
    else: available = total = 4 * 1024 ** 3
    disk = shutil.disk_usage(Path(workspace).anchor or workspace)
    return HardwareProfile(logical, max(physical or logical, 1), round(available / 1024 ** 3, 2), round(total / 1024 ** 3, 2), round(disk.free / 1024 ** 3, 2), _gpu_devices(), platform.platform())

def build_resource_plan(profile, config, connectivity):
    reserve = 2 if profile.logical_cpu >= 8 else 1
    cpu_capacity = max(1, min(profile.physical_cpu, profile.logical_cpu - reserve))
    memory_capacity = max(1, int(profile.available_memory_gb // 1.5))
    cpu_workers = min(cpu_capacity, memory_capacity, int(config.get("distributed_cpu_worker_cap", 16)))
    online = connectivity.get("market_https", False)
    io_workers = min(max(2, cpu_workers * 2), int(config.get("distributed_io_worker_cap", 24)))
    # Market downloads are I/O bound. Size them from hardware/memory instead of
    # pinning every machine to four workers; provider circuit breakers remain the
    # remote-rate-limit safety layer.
    downloads = min(io_workers, max(4, profile.logical_cpu * 2), max(2, int(profile.available_memory_gb * 2))) if online else 1
    gpu_available = bool(profile.gpu_devices)
    gpu_enabled = gpu_available and bool(config.get("distributed_allow_gpu", True)) and bool(config.get("distributed_gpu_backend_ready", False))
    return {"distributed_enabled": cpu_workers > 1, "cpu_workers": cpu_workers, "io_workers": io_workers, "download_workers": int(config.get("download_workers") or downloads), "data_load_workers": int(config.get("data_load_workers") or io_workers), "optimization_workers": min(io_workers, 16), "model_threads": min(cpu_workers, 8), "gpu_available": gpu_available, "gpu_enabled": gpu_enabled, "safety_reserve_cpu": reserve, "memory_per_cpu_worker_gb": 1.5, "online_mode": online}

def initialize_distributed_compute(config, workspace):
    profile = inspect_hardware(workspace); timeout = float(config.get("connectivity_probe_timeout", 1.0))
    connectivity = {"market_https": _reachable("query1.finance.yahoo.com", 443, timeout), "dns": _reachable("1.1.1.1", 53, timeout), "local_ollama": _reachable("127.0.0.1", 11434, .25)}
    return {"hardware": asdict(profile), "connectivity": connectivity, "resource_plan": build_resource_plan(profile, config, connectivity)}

def apply_resource_plan(config, audit):
    merged = dict(config); plan = audit["resource_plan"]
    merged.update({key: plan[key] for key in ("download_workers", "data_load_workers", "optimization_workers")})
    lgb = dict(merged.get("lgb_params", {})); lgb["num_threads"] = plan["model_threads"]; merged["lgb_params"] = lgb
    merged["compute_resource_plan"] = plan
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[variable] = str(plan["model_threads"])
    return merged
