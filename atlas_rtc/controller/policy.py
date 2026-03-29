from __future__ import annotations

from dataclasses import dataclass

from atlas_rtc.controller.actions import ControlAction, adjust_temperature, apply_bias, apply_mask, correct_in_place, noop, restart
from atlas_rtc.controller.state import RuntimeState
from atlas_rtc.contracts.base import BaseContract

# Stages where the model is generating value content — policy should not interfere
_VALUE_STAGES = {"in_value", "expect_value", "closed", "expect_comma_or_close"}

# Stages where structural intervention is appropriate
_STRUCTURAL_STAGES = {"expect_open", "expect_key", "expect_colon", "expect_comma_or_close"}


@dataclass(slots=True)
class PolicyThresholds:
    low: float = 0.25
    medium: float = 0.5
    high: float = 0.8
    critical: float = 0.95


class LadderPolicy:
    """Escalating control policy with stage awareness.

    During value generation stages (in_value, expect_value) the policy
    backs off entirely — intervening mid-value corrupts output.
    Interventions only fire during structural stages where the model
    is deciding what token TYPE comes next.

    Low drift     -> noop
    Medium drift  -> structural bias (structural stages only)
    High drift    -> temperature clamp
    Critical drift -> token masking (structural stages only)
    Unrecoverable -> rollback + re-steer (structural stages only, step >= 15)
    Budget exhausted -> restart (last resort)
    """

    def __init__(
        self,
        thresholds: PolicyThresholds | None = None,
        rollback_depth: int = 2,
        max_corrections: int = 3,
    ):
        self.thresholds = thresholds or PolicyThresholds()
        self.rollback_depth = rollback_depth
        self.max_corrections = max_corrections

    def choose(self, state: RuntimeState, contract: BaseContract) -> ControlAction:
        score = max(state.drift_score, state.failure_probability)
        progress = state.contract_progress or contract.initial_progress()
        stage = progress.stage

        # ── Initial priming (before first token) ─────────────────────────────
        if state.step_index < 0:
            return apply_bias(
                contract.structural_bias(state.generated_text, progress),
                "initial structural priming",
            )

        # ── Value generation zone — back off completely ───────────────────────
        if stage in _VALUE_STAGES:
            return noop(f"value generation stage ({stage}) — no intervention")

        # ── Structural stages — apply ladder ─────────────────────────────────
        if score < self.thresholds.low:
            return noop("trajectory appears healthy")

        if score < self.thresholds.medium:
            bias = contract.structural_bias(state.generated_text, progress)
            if bias:
                return apply_bias(bias, "light structural shaping")
            return noop("no structural bias needed")

        if score < self.thresholds.high:
            return adjust_temperature(0.3, "reduce sampling variance under moderate drift")

        if score < self.thresholds.critical:
            mask = self._mask_from_contract(state, contract)
            if mask:
                return apply_mask(mask, "mask invalid structural tokens under high drift")
            return adjust_temperature(0.1, "high drift but no maskable tokens")

        # ── Critical drift — rollback + re-steer ─────────────────────────────
        min_step = max(self.rollback_depth, 15)
        if state.correction_count < self.max_corrections and state.step_index >= min_step:
            bias = contract.structural_bias(state.generated_text, progress)
            strong_bias = {tok: v * 4.0 for tok, v in bias.items()}
            mask = self._mask_from_contract(state, contract)
            return correct_in_place(
                rollback_depth=self.rollback_depth,
                token_bias=strong_bias,
                masked_tokens=mask,
                reason=f"critical drift {score:.2f}; rolling back {self.rollback_depth} tokens",
            )

        return restart("correction budget exhausted; falling back to restart")

    @staticmethod
    def _mask_from_contract(state: RuntimeState, contract: BaseContract) -> list[str]:
        progress = state.contract_progress or contract.initial_progress()
        allowed = set(contract.allowed_tokens(state.generated_text, progress))
        invalid = []
        for token in str(state.raw_features.get("candidate_tokens", "")).split("|"):
            if token and token not in allowed:
                invalid.append(token)
        return invalid
