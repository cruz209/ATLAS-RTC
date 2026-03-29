from __future__ import annotations

from atlas_rtc.contracts.base import BaseContract, ValidationResult


class Verifier:
    def verify(self, text: str, contract: BaseContract) -> ValidationResult:
        return contract.validate(text)
