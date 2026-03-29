from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from atlas_rtc.adapters.base import BaseAdapter, DecodeStep, InterventionDirectives, TokenCandidate


@dataclass(slots=True)
class MockScenario:
    prompt: str
    planned_tokens: List[str]
    candidates: List[Dict[str, float]]


class MockAdapter(BaseAdapter):
    name = "mock"
    supports_logit_control = True

    def __init__(self, scenario: MockScenario):
        self.scenario = scenario
        self._index = 0
        self._text = ""
        # Track individual token strings so rollback can truncate text correctly.
        self._token_history: List[str] = []

    def initialize(self, prompt: str) -> None:
        self._index = 0
        self._text = ""
        self._token_history = []
        self.scenario.prompt = prompt

    def restart(self, prompt: str) -> None:
        self.initialize(prompt)

    def rollback(self, n: int) -> None:
        if n <= 0:
            return
        if n > len(self._token_history):
            raise ValueError(
                f"Cannot rollback {n} tokens; only {len(self._token_history)} have been generated."
            )
        self._token_history = self._token_history[:-n]
        self._index -= n
        self._text = "".join(self._token_history)

    def is_finished(self) -> bool:
        return self._index >= len(self.scenario.planned_tokens)

    def get_text(self) -> str:
        return self._text

    def step(self, directives: InterventionDirectives | None = None) -> DecodeStep:
        if self.is_finished():
            raise RuntimeError("generation already finished")
        directives = directives or InterventionDirectives()
        distribution = dict(self.scenario.candidates[self._index])

        for token in directives.masked_tokens:
            if token in distribution:
                distribution[token] = float("-inf")
        for token, bias in directives.token_bias.items():
            if token in distribution and distribution[token] != float("-inf"):
                distribution[token] += bias

        chosen = max(distribution.items(), key=lambda kv: kv[1])[0]
        self._token_history.append(chosen)
        self._text += chosen
        top_k = self._to_candidates(distribution)
        entropy = self._entropy(top_k)
        out = DecodeStep(
            step_index=self._index,
            generated_text=self._text,
            top_k=top_k,
            entropy=entropy,
            raw_logits=distribution,
        )
        self._index += 1
        return out

    @staticmethod
    def _to_candidates(distribution: Dict[str, float]) -> List[TokenCandidate]:
        finite = {k: v for k, v in distribution.items() if v != float("-inf")}
        max_logit = max(finite.values()) if finite else 0.0
        exps = {k: math.exp(v - max_logit) for k, v in finite.items()}
        denom = sum(exps.values()) or 1.0
        probs = {k: v / denom for k, v in exps.items()}
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        out: List[TokenCandidate] = []
        for token, p in ordered:
            out.append(TokenCandidate(token=token, logit=distribution[token], probability=p))
        return out

    @staticmethod
    def _entropy(candidates: List[TokenCandidate]) -> float:
        total = 0.0
        for c in candidates:
            if c.probability > 0:
                total -= c.probability * math.log(c.probability)
        return total
