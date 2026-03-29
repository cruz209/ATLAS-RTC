"""
Run ATLAS-RTC against Qwen2.5-7B-Instruct via vLLM on vast.ai.

Usage (on vast.ai instance):
    pip install vllm
    pip install -e .
    python -m atlas_rtc.examples.run_vllm_demo
"""
from __future__ import annotations

from atlas_rtc.adapters.vllm_adapter import VLLMAdapter
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.policy import LadderPolicy, PolicyThresholds
from atlas_rtc.controller.runtime import RuntimeController

MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT = (
    "You are a helpful assistant that only outputs valid JSON.\n"
    "Return a JSON object with exactly these keys: name, age, city.\n"
    "Output only the JSON object, nothing else.\n"
)

def main() -> None:
    print(f"Loading {MODEL}...")
    adapter = VLLMAdapter(MODEL, max_new_tokens=128, top_k=20)

    contract = JSONSchemaContract(required_keys=["name", "age", "city"])

    # Use a slightly lower critical threshold so corrections fire more readily
    # during initial testing — tune upward once you've confirmed the path works.
    policy = LadderPolicy(
        thresholds=PolicyThresholds(low=0.2, medium=0.4, high=0.7, critical=0.85),
        rollback_depth=2,
        max_corrections=3,
    )

    controller = RuntimeController(
        adapter=adapter,
        contract=contract,
        policy=policy,
        max_steps=128,
        max_restarts=1,
    )

    print("Running controller...\n")
    text, result, state = controller.run(PROMPT)

    print(f"Output:         {text}")
    print(f"Valid:          {result.valid}")
    print(f"Errors:         {result.errors}")
    print(f"Interventions:  {len(state.intervention_history)}")
    print(f"Corrections:    {state.correction_count}")
    print()

    # Print the action trace so you can see exactly what fired at each step.
    print("Action trace:")
    for i, entry in enumerate(state.intervention_history):
        print(f"  [{i:02d}] {entry['name']:12s}  {entry['reason']}")

    # Dump full telemetry for offline analysis.
    controller.logger.dump_jsonl("outputs/vllm_run.jsonl")
    print("\nTelemetry written to outputs/vllm_run.jsonl")


if __name__ == "__main__":
    main()
