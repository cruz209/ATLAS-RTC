from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class LightweightLogisticDetector:
    """A tiny dependency-free logistic detector baseline.

    This avoids forcing scikit-learn just to have a learned-looking detector.
    You can later swap in sklearn or a neural classifier with the same API.
    """

    feature_order: List[str] = field(default_factory=lambda: [
        "entropy",
        "top_probability",
        "invalid_mass",
        "opened",
        "closed",
        "step_index",
    ])
    weights: List[float] = field(default_factory=lambda: [0.2, -0.4, 2.2, -0.7, -0.2, 0.05])
    bias: float = -0.5

    def predict_proba(self, features: Dict[str, float | str]) -> float:
        z = self.bias
        for name, weight in zip(self.feature_order, self.weights):
            z += weight * float(features.get(name, 0.0))
        return 1.0 / (1.0 + (2.718281828459045 ** (-z)))

    def fit(self, rows: Iterable[Dict[str, float]], labels: Iterable[int], lr: float = 0.05, epochs: int = 100) -> None:
        rows = list(rows)
        labels = list(labels)
        for _ in range(epochs):
            grad_w = [0.0 for _ in self.weights]
            grad_b = 0.0
            for row, y in zip(rows, labels):
                p = self.predict_proba(row)
                err = p - y
                for i, name in enumerate(self.feature_order):
                    grad_w[i] += err * float(row.get(name, 0.0))
                grad_b += err
            n = max(1, len(rows))
            self.weights = [w - lr * g / n for w, g in zip(self.weights, grad_w)]
            self.bias -= lr * grad_b / n
