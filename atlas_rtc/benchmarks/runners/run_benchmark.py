from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from atlas_rtc.adapters.mock_adapter import MockAdapter, MockScenario
from atlas_rtc.benchmarks.tasks.json_tasks import sample_tasks
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.runtime import RuntimeController


def build_mock_success_scenario(required_keys: list[str]) -> MockScenario:
    # intentionally makes the first step ambiguous so controller has something to do
    planned = ["{", f'"{required_keys[0]}"', ":", '"alice"', ",", f'"{required_keys[1]}"', ":", "30", "}"]
    candidates = [
        {"H": 1.8, "{": 1.2},
        {f'"{required_keys[0]}"': 2.0, "oops": 1.7},
        {":": 2.5, "-": 0.5},
        {'"alice"': 2.0, '"bob"': 1.5},
        {",": 2.0, "}": 1.0},
        {f'"{required_keys[1]}"': 2.0, '"junk"': 1.6},
        {":": 2.5},
        {"30": 2.3, '"unknown"': 1.0},
        {"}": 2.5, ",": 0.8},
    ]
    return MockScenario(prompt="", planned_tokens=planned, candidates=candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(config.get("out_dir", "outputs/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, task in enumerate(sample_tasks()):
        adapter = MockAdapter(build_mock_success_scenario(task.required_keys))
        contract = JSONSchemaContract(required_keys=task.required_keys)
        controller = RuntimeController(adapter=adapter, contract=contract)
        start = time.perf_counter()
        text, result, state = controller.run(task.prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        controller.logger.dump_jsonl(out_dir / f"run_{idx}.jsonl")
        rows.append({
            "task": idx,
            "valid": result.valid,
            "latency_ms": round(elapsed_ms, 2),
            "interventions": len(state.intervention_history),
            "text": text,
        })

    summary_path = out_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump({"results": rows}, sort_keys=False))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
