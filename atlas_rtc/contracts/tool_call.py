from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from atlas_rtc.contracts.base import BaseContract, ContractProgress, ValidationResult


@dataclass(slots=True)
class ToolCallContract(BaseContract):
    tool_name: str
    required_args: List[str]
    name: str = "tool_call"

    def initial_progress(self) -> ContractProgress:
        return ContractProgress(stage="expect_call")

    def update_progress(self, text: str, progress: ContractProgress) -> ContractProgress:
        stage = "expect_call"
        if self.tool_name in text:
            stage = "tool_seen"
        if text.count("{") > 0:
            stage = "args_started"
        if text.count("{") == text.count("}") and text.count("{") > 0:
            stage = "closed"
        return ContractProgress(stage=stage)

    def allowed_tokens(self, text: str, progress: ContractProgress) -> List[str]:
        return [self.tool_name, "{", "}", ":", ",", '"', "(", ")", *self.required_args]

    def structural_bias(self, text: str, progress: ContractProgress) -> Dict[str, float]:
        out = {self.tool_name: 2.5} if self.tool_name not in text else {}
        for arg in self.required_args:
            if arg not in text:
                out[arg] = 1.0
        return out

    def validate(self, text: str) -> ValidationResult:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return ValidationResult(False, [str(exc)])
        if parsed.get("tool") != self.tool_name:
            return ValidationResult(False, [f"expected tool {self.tool_name}"], parsed=parsed)
        missing = [a for a in self.required_args if a not in parsed.get("args", {})]
        if missing:
            return ValidationResult(False, [f"missing args: {missing}"], parsed=parsed)
        return ValidationResult(True, parsed=parsed)
