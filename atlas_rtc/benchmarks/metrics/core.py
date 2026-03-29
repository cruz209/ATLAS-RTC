from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkResult:
    success: bool
    latency_ms: float
    interventions: int
    retries: int
    output_text: str
