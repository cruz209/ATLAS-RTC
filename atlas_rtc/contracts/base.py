from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class ContractProgress:
    stage: str = "start"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    parsed: Any = None


class BaseContract(ABC):
    name: str = "base"

    @abstractmethod
    def initial_progress(self) -> ContractProgress:
        raise NotImplementedError

    @abstractmethod
    def update_progress(self, text: str, progress: ContractProgress) -> ContractProgress:
        raise NotImplementedError

    @abstractmethod
    def allowed_tokens(self, text: str, progress: ContractProgress) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def structural_bias(self, text: str, progress: ContractProgress) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def validate(self, text: str) -> ValidationResult:
        raise NotImplementedError
