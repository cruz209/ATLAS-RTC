from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from atlas_rtc.contracts.base import BaseContract, ContractProgress, ValidationResult


@dataclass(slots=True)
class JSONSchemaContract(BaseContract):
    required_keys: List[str]
    name: str = "json_schema"

    # ── Stage machine ────────────────────────────────────────────────────────
    # Stages reflect what the parser expects NEXT, not what just happened.
    # This lets the policy know exactly when to intervene vs. back off.
    #
    #  expect_open       → waiting for {
    #  expect_key        → inside object, waiting for a key string
    #  expect_colon      → key seen, waiting for :
    #  expect_value      → colon seen, waiting for value (string/number/array/object)
    #  in_value          → currently generating value tokens — NOOP ZONE
    #  expect_comma_or_close → value done, waiting for , or }
    #  closed            → top-level } seen
    # ─────────────────────────────────────────────────────────────────────────

    def initial_progress(self) -> ContractProgress:
        return ContractProgress(
            stage="expect_open",
            metadata={"found_keys": [], "depth": 0, "in_string": False},
        )

    def update_progress(self, text: str, progress: ContractProgress) -> ContractProgress:
        """Re-derive stage from full text each step — cheap and correct."""
        if not text:
            return self.initial_progress()

        # Parse structural state
        depth = 0
        in_str = False
        escaped = False
        last_char = ""
        found_keys = []
        for k in self.required_keys:
            if f'"{k}"' in text:
                found_keys.append(k)

        for ch in text:
            if escaped:
                escaped = False
                last_char = ch
                continue
            if ch == "\\" and in_str:
                escaped = True
                last_char = ch
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch in "{[":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
            last_char = ch

        # Top-level object not yet opened
        if "{" not in text:
            return ContractProgress(
                stage="expect_open",
                metadata={"found_keys": found_keys, "depth": 0, "in_string": False},
            )

        # Top-level object closed
        top_open  = text.count("{")
        top_close = text.count("}")
        if top_open > 0 and top_open == top_close:
            return ContractProgress(
                stage="closed",
                metadata={"found_keys": found_keys, "depth": 0, "in_string": False},
            )

        # Currently inside a string value — noop zone
        if in_str:
            return ContractProgress(
                stage="in_value",
                metadata={"found_keys": found_keys, "depth": depth, "in_string": True},
            )

        # Determine what comes next based on last non-whitespace char
        stripped = text.rstrip()
        if not stripped:
            return self.initial_progress()

        last_non_ws = stripped[-1]

        if last_non_ws == "{" or last_non_ws == ",":
            stage = "expect_key"
        elif last_non_ws == ":":
            stage = "expect_value"
        elif last_non_ws in ('"', "'"):
            # End of a key string or value string — could be after key or value
            # Heuristic: if depth == 1 and we just closed a string, it's after a key
            # if the char before the closing quote is alphanumeric it's a value end
            stage = "expect_comma_or_close"
        elif last_non_ws in ("}", "]"):
            stage = "expect_comma_or_close"
        elif last_non_ws.isdigit() or last_non_ws in ("e", "E", ".", "l", "e", "s", "e"):
            # Mid or end of numeric/bool/null value
            stage = "in_value"
        else:
            stage = "in_value"

        return ContractProgress(
            stage=stage,
            metadata={"found_keys": found_keys, "depth": depth, "in_string": False},
        )

    def allowed_tokens(self, text: str, progress: ContractProgress) -> List[str]:
        """Broad allowlist — only used for invalid_mass computation, not hard masking."""
        stage = progress.stage
        if stage == "expect_open":
            return ["{"]
        if stage in ("in_value", "expect_comma_or_close", "closed"):
            # Everything is valid — return huge set so invalid_mass ≈ 0
            return self._broad_allowlist()
        if stage == "expect_key":
            keys = [f'"{k}"' for k in self.required_keys]
            return ['"', "}"] + keys + self._common_json_tokens()
        if stage == "expect_colon":
            return [":", ": "]
        if stage == "expect_value":
            return self._broad_allowlist()
        return self._broad_allowlist()

    def structural_bias(self, text: str, progress: ContractProgress) -> Dict[str, float]:
        """Return logit biases appropriate for current stage."""
        stage = progress.stage
        found = set(progress.metadata.get("found_keys", []))
        missing = [k for k in self.required_keys if k not in found]

        if stage == "expect_open":
            return {"{": 5.0}

        if stage in ("in_value", "expect_value", "closed"):
            # Do NOT interfere with value generation
            return {}

        if stage == "expect_key":
            # Bias toward missing required keys
            bias = {}
            for k in missing:
                bias[f'"{k}"'] = 2.0
                bias[k] = 1.5
            if not missing:
                # All keys present — bias toward closing
                bias["}"] = 3.0
            return bias

        if stage == "expect_comma_or_close":
            if missing:
                return {",": 2.0, '",': 1.5}
            else:
                return {"}": 3.0, "}\n": 2.0}

        return {}

    def validate(self, text: str) -> ValidationResult:
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        # Strip anything after the first complete JSON object (string-aware)
        try:
            depth = 0
            in_str = False
            escaped = False
            end_idx = None
            for i, ch in enumerate(clean):
                if escaped:
                    escaped = False
                    continue
                if ch == "\\" and in_str:
                    escaped = True
                    continue
                if ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and i > 0:
                            end_idx = i
                            break
            if end_idx is not None:
                clean = clean[:end_idx + 1]
        except Exception:
            pass
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            return ValidationResult(valid=False, errors=[str(exc)])
        missing = [k for k in self.required_keys if k not in parsed]
        if missing:
            return ValidationResult(valid=False, errors=[f"missing keys: {missing}"], parsed=parsed)
        return ValidationResult(valid=True, parsed=parsed)

    @staticmethod
    def _broad_allowlist() -> List[str]:
        base = [
            "{", "}", ":", ",", '"', " ", "\n", "\r",
            '",', ',"', ': ', ' "', '": ', '",\n', '},', "},\n",
            '": "', ', "', '{"', '":', '" ', ' }', '\n}',
        ]
        base.extend([str(i) for i in range(10)])
        base.extend(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        base.extend([".", "-", "_", "@", "!", "?", "'", "(", ")", "[", "]", "/"])
        return base

    @staticmethod
    def _common_json_tokens() -> List[str]:
        return ["{", "}", ":", ",", " ", "\n", '": ', '",', ", "]
