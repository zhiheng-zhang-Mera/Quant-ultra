"""High-throughput download primitives shared by market-data providers.

The module is deliberately provider-neutral: existing engine APIs stay intact while
REST providers can opt into one persistent connection pool and non-blocking batches.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import requests
from requests.adapters import HTTPAdapter


@dataclass(frozen=True)
class DownloadPlan:
    workers: int
    connection_pool: int
    batch_size: int


def build_download_plan(worker_override: int | None = None) -> DownloadPlan:
    """Size I/O concurrency from CPU and available memory with conservative caps."""
    logical = max(os.cpu_count() or 1, 1)
    available_gb = 4.0
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass
    memory_limit = max(2, int(available_gb * 2))
    automatic = min(32, max(4, logical * 2), memory_limit)
    workers = max(1, int(worker_override or automatic))
    pool = max(8, workers * 2)
    return DownloadPlan(workers=workers, connection_pool=pool, batch_size=max(16, workers * 4))


def create_persistent_session(plan: DownloadPlan, proxy_url: str | None = None) -> requests.Session:
    """Create one keep-alive pool instead of opening a TCP connection per request."""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=plan.connection_pool, pool_maxsize=plan.connection_pool, max_retries=2, pool_block=True)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Quant-Ultra/8-6", "Connection": "keep-alive"})
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session


class AsyncRestBatchClient:
    """Persistent aiohttp client for high-concurrency, non-blocking REST batches."""

    def __init__(self, plan: DownloadPlan, timeout_seconds: float = 30.0, proxy_url: str | None = None):
        self.plan = plan
        self.timeout_seconds = timeout_seconds
        self.proxy_url = proxy_url

    async def fetch_json(self, requests_: Iterable[Mapping[str, Any]]) -> list[Any]:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("aiohttp is required for asynchronous REST downloads") from exc

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=self.plan.connection_pool, limit_per_host=self.plan.workers, ttl_dns_cache=300)
        semaphore = asyncio.Semaphore(self.plan.workers)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": "Quant-Ultra/8-6"}) as session:
            async def one(spec: Mapping[str, Any]):
                async with semaphore:
                    async with session.request(
                        spec.get("method", "GET"), spec["url"], params=spec.get("params"),
                        json=spec.get("json"), headers=spec.get("headers"), proxy=self.proxy_url,
                    ) as response:
                        response.raise_for_status()
                        return await response.json(content_type=None)

            return await asyncio.gather(*(one(spec) for spec in requests_))

    def fetch_json_sync(self, requests_: Iterable[Mapping[str, Any]]) -> list[Any]:
        """Synchronous bridge for the existing engine; call async API inside event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.fetch_json(requests_))
        raise RuntimeError("fetch_json_sync cannot run inside an active event loop; await fetch_json instead")
