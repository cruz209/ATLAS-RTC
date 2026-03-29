from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from atlas_rtc.contracts.base import BaseContract, ContractProgress, ValidationResult


@dataclass(slots=True)
class ExtractionContract(BaseContract):
    fields: List[str]
    name: str = "extraction"

    def initial_progress(self) -> ContractProgress:
        return ContractProgress(stage="extract_start")

    def update_progress(self, text: str, progress: ContractProgress) -> ContractProgress:
        found = [f for f in self.fields if f'"{f}"' in text]
        stage = "closed" if text.count("{") > 0 and text.count("{") == text.count("}") else "extracting"
        return ContractProgress(stage=stage, metadata={"found_fields": found})

    def allowed_tokens(self, text: str, progress: ContractProgress) -> List[str]:
        return ["{", "}", ":", ",", '"', *self.fields]

    def structural_bias(self, text: str, progress: ContractProgress) -> dict[str, float]:
        found = set(progress.metadata.get("found_fields", []))
        return {field: 1.25 for field in self.fields if field not in found}

    def validate(self, text: str) -> ValidationResult:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return ValidationResult(False, [str(exc)])
        missing = [f for f in self.fields if f not in parsed]
        return ValidationResult(valid=not missing, errors=[f"missing fields: {missing}"] if missing else [], parsed=parsed)
