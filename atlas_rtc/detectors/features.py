from __future__ import annotations

from typing import Dict

from atlas_rtc.adapters.base import DecodeStep
from atlas_rtc.contracts.base import BaseContract, ContractProgress


class FeatureExtractor:
    def extract(self, step: DecodeStep, text: str, contract: BaseContract, progress: ContractProgress) -> Dict[str, float | str]:
        top_prob = step.top_k[0].probability if step.top_k else 0.0
        valid_tokens = set(contract.allowed_tokens(text, progress))
        invalid_mass = 0.0
        candidate_tokens = []
        for cand in step.top_k:
            candidate_tokens.append(cand.token)
            if cand.token not in valid_tokens:
                invalid_mass += cand.probability
        opened = 1.0 if "{" in text else 0.0
        closed = 1.0 if text.count("{") > 0 and text.count("{") == text.count("}") else 0.0
        return {
            "entropy": step.entropy,
            "top_probability": top_prob,
            "invalid_mass": invalid_mass,
            "opened": opened,
            "closed": closed,
            "step_index": float(step.step_index),
            "candidate_tokens": "|".join(candidate_tokens),
        }
