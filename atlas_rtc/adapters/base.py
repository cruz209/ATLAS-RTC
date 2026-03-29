from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(slots=True)
class TokenCandidate:
    token: str
    logit: float
    probability: float


@dataclass(slots=True)
class DecodeStep:
    step_index: int
    generated_text: str
    top_k: List[TokenCandidate]
    entropy: float
    raw_logits: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class InterventionDirectives:
    token_bias: Dict[str, float] = field(default_factory=dict)
    masked_tokens: List[str] = field(default_factory=list)
    temperature: float | None = None
    restart: bool = False
    rollback_depth: int = 0  # tokens to rewind before re-stepping; 0 = no rollback
    notes: List[str] = field(default_factory=list)


class BaseAdapter(ABC):
    """Interface for stepping through model generation with optional control.

    Adapters may support true logit control or a degraded black-box mode.
    """

    name: str = "base"
    supports_logit_control: bool = False

    @abstractmethod
    def initialize(self, prompt: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, directives: InterventionDirectives | None = None) -> DecodeStep:
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_text(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def restart(self, prompt: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def rollback(self, n: int) -> None:
        """Rewind the last n generated tokens, restoring adapter state to t-n.

        Implementations must truncate generated_text and reset any internal
        step index accordingly. For real adapters this requires KV-cache
        truncation. Raises ValueError if n exceeds the number of generated tokens.
        """
        raise NotImplementedError
