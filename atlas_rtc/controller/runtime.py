from __future__ import annotations

from dataclasses import asdict

from atlas_rtc.adapters.base import BaseAdapter
from atlas_rtc.controller.actions import ControlAction
from atlas_rtc.controller.policy import LadderPolicy
from atlas_rtc.controller.state import RuntimeState
from atlas_rtc.contracts.base import BaseContract, ValidationResult
from atlas_rtc.detectors.features import FeatureExtractor
from atlas_rtc.detectors.heuristic import HeuristicDriftDetector
from atlas_rtc.detectors.learned import LightweightLogisticDetector
from atlas_rtc.telemetry.logger import EventLogger
from atlas_rtc.verification.validators import Verifier


class RuntimeController:
    def __init__(
        self,
        adapter: BaseAdapter,
        contract: BaseContract,
        heuristic_detector: HeuristicDriftDetector | None = None,
        learned_detector: LightweightLogisticDetector | None = None,
        policy: LadderPolicy | None = None,
        logger: EventLogger | None = None,
        verifier: Verifier | None = None,
        max_steps: int = 128,
        max_restarts: int = 1,
    ) -> None:
        self.adapter = adapter
        self.contract = contract
        self.heuristic_detector = heuristic_detector or HeuristicDriftDetector()
        self.learned_detector = learned_detector or LightweightLogisticDetector()
        self.policy = policy or LadderPolicy()
        self.logger = logger or EventLogger()
        self.verifier = verifier or Verifier()
        self.max_steps = max_steps
        self.max_restarts = max_restarts
        self.feature_extractor = FeatureExtractor()

    def run(self, prompt: str) -> tuple[str, ValidationResult, RuntimeState]:
        self.adapter.initialize(prompt)
        state = RuntimeState(prompt=prompt, contract_progress=self.contract.initial_progress())
        restarts = 0
        for _ in range(self.max_steps):
            action = self.policy.choose(state, self.contract)

            # --- restart: full sequence reset, last resort ---
            if action.name == "restart":
                if restarts < self.max_restarts:
                    restarts += 1
                    self._log_action(action, state)
                    self.adapter.restart(prompt)
                    state = RuntimeState(prompt=prompt, contract_progress=self.contract.initial_progress())
                else:
                    self.logger.log("restart_skipped", reason="max_restarts exceeded")
                continue

            # --- correct: rollback N tokens then re-step with strong directives ---
            if action.name == "correct":
                depth = action.directives.rollback_depth
                if depth > 0 and state.step_index >= depth:
                    self.adapter.rollback(depth)
                    state.generated_text = self.adapter.get_text()
                    state.step_index -= depth
                    state.correction_count += 1
                self._log_action(action, state)
                # Fall through to step() with corrective directives.
            else:
                self._log_action(action, state)

            step = self.adapter.step(action.directives)
            state.step_index = step.step_index
            state.generated_text = step.generated_text
            state.entropy = step.entropy
            state.contract_progress = self.contract.update_progress(
                step.generated_text,
                state.contract_progress or self.contract.initial_progress(),
            )
            features = self.feature_extractor.extract(step, step.generated_text, self.contract, state.contract_progress)
            state.raw_features = dict(features)
            heuristic = self.heuristic_detector.score(features)
            learned = self.learned_detector.predict_proba(features)
            state.drift_score = heuristic
            state.failure_probability = learned
            self.logger.log(
                "decode_step",
                step_index=step.step_index,
                text=step.generated_text,
                entropy=step.entropy,
                drift_score=heuristic,
                failure_probability=learned,
                correction_count=state.correction_count,
                features={k: v for k, v in features.items() if k != "candidate_tokens"},
            )
            if self.adapter.is_finished():
                break

        result = self.verifier.verify(self.adapter.get_text(), self.contract)
        self.logger.log("verification", valid=result.valid, errors=result.errors)
        return self.adapter.get_text(), result, state

    def _log_action(self, action: ControlAction, state: RuntimeState) -> None:
        state.intervention_history.append({"name": action.name, "reason": action.reason})
        self.logger.log(
            "action",
            action=action.name,
            reason=action.reason,
            drift_score=state.drift_score,
            failure_probability=state.failure_probability,
        )
