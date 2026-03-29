from __future__ import annotations
from atlas_rtc.adapters.hf_adapter import HuggingFaceAdapter
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.policy import LadderPolicy, PolicyThresholds
from atlas_rtc.controller.runtime import RuntimeController

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REQUIRED_KEYS = ["name", "age", "city"]
PROMPT = (
    "You are a helpful assistant that only outputs valid JSON.\n"
    "Return a JSON object with exactly these keys: name, age, city.\n"
    "Output only the JSON object, nothing else.\n"
)

def main() -> None:
    adapter = HuggingFaceAdapter(MODEL, max_new_tokens=128, top_k=20, required_keys=REQUIRED_KEYS)
    contract = JSONSchemaContract(required_keys=REQUIRED_KEYS)
    policy = LadderPolicy(
        thresholds=PolicyThresholds(low=0.2, medium=0.4, high=0.7, critical=0.85),
        rollback_depth=2, max_corrections=3,
    )
    controller = RuntimeController(adapter=adapter, contract=contract, policy=policy, max_steps=128, max_restarts=1)

    print("Running controller...\n")
    text, result, state = controller.run(PROMPT)
    print(f"Output:         {text}")
    print(f"Valid:          {result.valid}")
    print(f"Errors:         {result.errors}")
    print(f"Interventions:  {len(state.intervention_history)}")
    print(f"Corrections:    {state.correction_count}")
    print("\nAction trace:")
    for i, e in enumerate(state.intervention_history):
        print(f"  [{i:02d}] {e['name']:12s}  {e['reason']}")
    controller.logger.dump_jsonl("outputs/hf_run.jsonl")
    print("\nTelemetry written to outputs/hf_run.jsonl")

if __name__ == "__main__":
    main()
