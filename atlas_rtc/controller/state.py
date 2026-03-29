from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from atlas_rtc.contracts.base import ContractProgress


@dataclass(slots=True)
class RuntimeState:
    prompt: str
    generated_text: str = ""
    step_index: int = -1
    entropy: float = 0.0
    drift_score: float = 0.0
    failure_probability: float = 0.0
    contract_progress: ContractProgress | None = None
    raw_features: Dict[str, float] = field(default_factory=dict)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)
    correction_count: int = 0  # number of mid-step rollback+re-steer corrections applied
