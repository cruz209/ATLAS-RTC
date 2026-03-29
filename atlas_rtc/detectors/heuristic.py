from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class HeuristicWeights:
    invalid_mass: float = 0.55
    entropy: float = 0.15
    missing_structure: float = 0.30


class HeuristicDriftDetector:
    def __init__(self, weights: HeuristicWeights | None = None, entropy_threshold: float = 1.75):
        self.weights = weights or HeuristicWeights()
        self.entropy_threshold = entropy_threshold

    def score(self, features: Dict[str, float | str]) -> float:
        invalid_mass = float(features.get("invalid_mass", 0.0))
        entropy = float(features.get("entropy", 0.0))
        opened = float(features.get("opened", 0.0))
        step_index = float(features.get("step_index", 0.0))
        missing_structure = 1.0 if step_index <= 2 and opened == 0.0 else 0.0
        entropy_term = min(1.0, max(0.0, entropy / self.entropy_threshold))
        score = (
            self.weights.invalid_mass * invalid_mass
            + self.weights.entropy * entropy_term
            + self.weights.missing_structure * missing_structure
        )
        return max(0.0, min(1.0, score))
