from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from atlas_rtc.adapters.base import InterventionDirectives


@dataclass(slots=True)
class ControlAction:
    name: str
    directives: InterventionDirectives = field(default_factory=InterventionDirectives)
    reason: str = ""


def noop(reason: str = "") -> ControlAction:
    return ControlAction(name="noop", reason=reason)


def apply_mask(tokens: List[str], reason: str = "") -> ControlAction:
    return ControlAction(
        name="mask",
        directives=InterventionDirectives(masked_tokens=tokens),
        reason=reason,
    )


def apply_bias(token_bias: Dict[str, float], reason: str = "") -> ControlAction:
    return ControlAction(
        name="bias",
        directives=InterventionDirectives(token_bias=token_bias),
        reason=reason,
    )


def adjust_temperature(value: float, reason: str = "") -> ControlAction:
    return ControlAction(
        name="temperature",
        directives=InterventionDirectives(temperature=value),
        reason=reason,
    )


def restart(reason: str = "") -> ControlAction:
    return ControlAction(
        name="restart",
        directives=InterventionDirectives(restart=True),
        reason=reason,
    )


def correct_in_place(
    rollback_depth: int,
    token_bias: Dict[str, float],
    masked_tokens: List[str] | None = None,
    reason: str = "",
) -> ControlAction:
    """Roll back n tokens then re-step with strong corrective directives.

    rollback_depth: how many tokens to rewind before re-stepping.
    token_bias: large positive biases on the structurally required tokens.
    masked_tokens: tokens to suppress outright during the corrected step.
    """
    return ControlAction(
        name="correct",
        directives=InterventionDirectives(
            rollback_depth=rollback_depth,
            token_bias=token_bias,
            masked_tokens=masked_tokens or [],
        ),
        reason=reason,
    )
