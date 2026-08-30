from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResearchBudget:
    max_requests: int = 100
    max_tokens: int = 1_000_000
    max_compute_seconds: float = 3600.0
    requests_used: int = 0
    tokens_used: int = 0
    compute_seconds_used: float = 0.0

    def consume(self, *, requests: int = 0, tokens: int = 0, compute_seconds: float = 0.0) -> None:
        next_values = (self.requests_used + requests, self.tokens_used + tokens,
                       self.compute_seconds_used + compute_seconds)
        if next_values[0] > self.max_requests or next_values[1] > self.max_tokens or next_values[2] > self.max_compute_seconds:
            raise RuntimeError("research budget exhausted")
        self.requests_used, self.tokens_used, self.compute_seconds_used = next_values
