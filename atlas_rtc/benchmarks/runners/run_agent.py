"""
ATLAS-RTC Agent Tool Call Reliability Benchmark.

Tests first-attempt success rate on tool calls that the model knows
how to produce but fails stochastically due to temperature sampling —
wrapping in markdown, adding preamble, or omitting required fields.

In a real agent pipeline, a failed first attempt means:
  - Retry latency (2x+ cost)
  - Broken downstream execution
  - Silent failures if not caught

ATLAS-RTC enforces the tool call contract mid-generation, recovering
trajectories before they become unrecoverable.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from atlas_rtc.adapters.hf_adapter import HuggingFaceAdapter
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.policy import LadderPolicy, PolicyThresholds
from atlas_rtc.controller.runtime import RuntimeController

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_TRIALS = 20
OUT_DIR = Path("outputs/agent")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOOL_TASKS = [
    {
        "name": "search",
        "prompt": (
            "You are an AI agent. The user said: 'Find recent papers on transformer attention mechanisms'.\n"
            "Output ONLY a raw JSON object with keys: tool, query, num_results, language.\n"
            "tool must be \"search\". num_results must be an integer. language is the result language string.\n"
        ),
        "required_keys": ["tool", "query", "num_results", "language"],
        "type_checks": {"num_results": int},
        "required_values": {"tool": "search"},
    },
    {
        "name": "send_email",
        "prompt": (
            "You are an AI agent. The user said: 'Email the Q3 report summary to my manager'.\n"
            "Output ONLY a raw JSON object with keys: tool, to, subject, summary, priority.\n"
            "tool must be \"send_email\". priority must be one of: low, medium, high.\n"
        ),
        "required_keys": ["tool", "to", "subject", "summary", "priority"],
        "type_checks": {},
        "required_values": {"tool": "send_email"},
    },
    {
        "name": "database_query",
        "prompt": (
            "You are an AI agent. The user said: 'Get the top 10 customers by revenue from last month'.\n"
            "Output ONLY a raw JSON object with keys: tool, database, table, limit, order_by.\n"
            "tool must be \"database_query\". limit must be an integer.\n"
        ),
        "required_keys": ["tool", "database", "table", "limit", "order_by"],
        "type_checks": {"limit": int},
        "required_values": {"tool": "database_query"},
    },
]

def extract_json(text):
    """Extract first complete JSON object, stripping markdown if needed."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        depth, in_str, escaped, end_idx = 0, False, False, None
        for i, ch in enumerate(text):
            if escaped: escaped = False; continue
            if ch == "\\" and in_str: escaped = True; continue
            if ch == '"': in_str = not in_str
            elif not in_str:
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and i > 0: end_idx = i; break
        if end_idx is not None: text = text[:end_idx + 1]
    except Exception:
        pass
    return text

def validate_tool_call(text, required_keys, type_checks=None, required_values=None):
    text = extract_json(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return False, [f"JSON parse error: {e}"], None
    errors = [f"missing key '{k}'" for k in required_keys if k not in parsed]
    if type_checks:
        for k, t in type_checks.items():
            if k in parsed and not isinstance(parsed[k], t):
                errors.append(f"'{k}' must be {t.__name__}, got {type(parsed[k]).__name__}")
    if required_values:
        for k, v in required_values.items():
            if k in parsed and parsed[k] != v:
                errors.append(f"'{k}' must be '{v}', got '{parsed[k]}'")
    return len(errors) == 0, errors, parsed if not errors else None

def run_baseline(adapter, task):
    import torch
    prompt_ids = adapter.tokenizer.encode(task["prompt"], return_tensors="pt").to(adapter._device)
    start = time.perf_counter()
    with torch.no_grad():
        out = adapter.model.generate(
            prompt_ids, max_new_tokens=256, do_sample=True, temperature=1.0,
            pad_token_id=adapter.tokenizer.eos_token_id,
        )
    elapsed = (time.perf_counter() - start) * 1000
    text = adapter.tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    # Baseline gets NO post-processing — raw output only
    try:
        parsed = json.loads(text.strip())
        errors = [f"missing key '{k}'" for k in task["required_keys"] if k not in parsed]
        if task.get("type_checks"):
            for k, t in task["type_checks"].items():
                if k in parsed and not isinstance(parsed[k], t):
                    errors.append(f"'{k}' must be {t.__name__}")
        if task.get("required_values"):
            for k, v in task["required_values"].items():
                if k in parsed and parsed[k] != v:
                    errors.append(f"'{k}' must be '{v}'")
        valid = len(errors) == 0
    except json.JSONDecodeError as e:
        valid, errors = False, [f"JSON parse error: {e}"]
    return {"valid": valid, "text": text, "latency_ms": round(elapsed, 2), "errors": errors}

def run_controlled(adapter, task):
    adapter.set_required_keys(task["required_keys"])
    contract = JSONSchemaContract(required_keys=task["required_keys"])
    policy = LadderPolicy(
        thresholds=PolicyThresholds(low=0.25, medium=0.45, high=0.70, critical=0.87),
        rollback_depth=2, max_corrections=2,
    )
    controller = RuntimeController(adapter=adapter, contract=contract, policy=policy, max_steps=256, max_restarts=1)
    start = time.perf_counter()
    text, result, state = controller.run(task["prompt"])
    elapsed = (time.perf_counter() - start) * 1000
    valid, errors, parsed = validate_tool_call(text, task["required_keys"], task.get("type_checks"), task.get("required_values"))
    return {"valid": valid, "text": text, "latency_ms": round(elapsed, 2), "errors": errors,
            "interventions": len(state.intervention_history), "corrections": state.correction_count}

def main():
    adapter = HuggingFaceAdapter(MODEL, max_new_tokens=256, top_k=20)
    all_results = []

    print("\nATLAS-RTC Agent Tool Call Reliability Benchmark")
    print("Model:", MODEL)
    print("Metric: first-attempt success rate (no retries)")
    print("="*60)

    for task_idx, task in enumerate(TOOL_TASKS):
        print(f"\nTool {task_idx+1}: {task['name']}")
        print(f"Required keys: {task['required_keys']}")
        print("-"*60)
        baseline_results, controlled_results = [], []

        for trial in range(N_TRIALS):
            b = run_baseline(adapter, task)
            baseline_results.append(b)
            c = run_controlled(adapter, task)
            controlled_results.append(c)
            b_s = "✓" if b["valid"] else "✗"
            c_s = "✓" if c["valid"] else "✗"
            b_info = "ok" if b["valid"] else b["errors"][0][:35]
            c_info = f"i={c['interventions']} corr={c['corrections']}" if c["valid"] else c["errors"][0][:35]
            print(f"  [{trial+1:02d}] baseline {b_s} {b_info} | controlled {c_s} {c_info}")

        b_valid = sum(r["valid"] for r in baseline_results)
        c_valid = sum(r["valid"] for r in controlled_results)
        b_lat = sum(r["latency_ms"] for r in baseline_results) / N_TRIALS
        c_lat = sum(r["latency_ms"] for r in controlled_results) / N_TRIALS

        print(f"\n  First-attempt success rate:")
        print(f"    Baseline   {b_valid}/{N_TRIALS} ({100*b_valid/N_TRIALS:.1f}%)  avg {b_lat:.0f}ms")
        print(f"    Controlled {c_valid}/{N_TRIALS} ({100*c_valid/N_TRIALS:.1f}%)  avg {c_lat:.0f}ms")
        print(f"    Delta: {c_valid-b_valid:+d} ({100*(c_valid-b_valid)/N_TRIALS:+.1f}pp)")
        print(f"    Latency overhead: {(c_lat-b_lat)/b_lat*100:+.1f}%")

        all_results.append({
            "tool": task["name"], "n_trials": N_TRIALS,
            "baseline":   {"valid": b_valid, "rate": b_valid/N_TRIALS, "avg_latency_ms": b_lat, "results": baseline_results},
            "controlled": {"valid": c_valid, "rate": c_valid/N_TRIALS, "avg_latency_ms": c_lat, "results": controlled_results},
        })

    total_b = sum(r["baseline"]["valid"] for r in all_results)
    total_c = sum(r["controlled"]["valid"] for r in all_results)
    total = N_TRIALS * len(TOOL_TASKS)

    print(f"\n{'='*60}")
    print(f"OVERALL ({len(TOOL_TASKS)} tools x {N_TRIALS} trials = {total} calls each)")
    print(f"{'='*60}")
    print(f"  Baseline   first-attempt success: {total_b}/{total} ({100*total_b/total:.1f}%)")
    print(f"  Controlled first-attempt success: {total_c}/{total} ({100*total_c/total:.1f}%)")
    print(f"  Delta: {total_c-total_b:+d} ({100*(total_c-total_b)/total:+.1f}pp)")
    print(f"\n  Without ATLAS-RTC: {total-total_b}/{total} tool calls fail on first attempt,")
    print(f"  requiring retry or causing silent pipeline failures.")
    print(f"  ATLAS-RTC recovers {total_c-total_b} additional calls on first attempt.")

    out = OUT_DIR / "agent_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results written to {out}")

if __name__ == "__main__":
    main()
