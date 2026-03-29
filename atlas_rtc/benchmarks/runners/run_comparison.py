from __future__ import annotations
import json, time
from pathlib import Path
from atlas_rtc.adapters.hf_adapter import HuggingFaceAdapter
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.policy import LadderPolicy, PolicyThresholds
from atlas_rtc.controller.runtime import RuntimeController

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_TRIALS = 20
OUT_DIR = Path("outputs/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Weak prompts — no "no markdown" instruction, model fails ~50% with ```json wrapping
TASKS = [
    {
        "name": "Weak prompt (name/age/city)",
        "prompt": "Return a JSON object with keys \"name\", \"age\", and \"city\".\n",
        "required_keys": ["name", "age", "city"],
        "type_checks": {},
    },
    {
        "name": "Weak prompt (title/year/director)",
        "prompt": "Return a JSON object with keys \"title\", \"year\", and \"director\".\n",
        "required_keys": ["title", "year", "director"],
        "type_checks": {},
    },
    {
        "name": "Weak prompt (country/capital/population)",
        "prompt": "Return a JSON object with keys \"country\", \"capital\", and \"population\".\n",
        "required_keys": ["country", "capital", "population"],
        "type_checks": {},
    },
]

def validate_strict(text, required_keys, type_checks=None):
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
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return False, [f"JSON parse error: {e}"]
    errors = [f"missing key: {k}" for k in required_keys if k not in parsed]
    if type_checks:
        for k, t in type_checks.items():
            if k in parsed and not isinstance(parsed[k], t):
                errors.append(f"'{k}' expected {t.__name__}")
    return len(errors) == 0, errors

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
    valid, errors = validate_strict(text, task["required_keys"], task.get("type_checks"))
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
    valid, errors = validate_strict(text, task["required_keys"], task.get("type_checks"))
    return {"valid": valid, "text": text, "latency_ms": round(elapsed, 2), "errors": errors,
            "interventions": len(state.intervention_history), "corrections": state.correction_count}

def main():
    adapter = HuggingFaceAdapter(MODEL, max_new_tokens=256, top_k=20)
    all_results = []

    for task_idx, task in enumerate(TASKS):
        print(f"\n{'='*60}\nTask {task_idx+1}: {task['name']}\n{'='*60}")
        baseline_results, controlled_results = [], []

        for trial in range(N_TRIALS):
            b = run_baseline(adapter, task)
            baseline_results.append(b)
            c = run_controlled(adapter, task)
            controlled_results.append(c)
            b_s = "✓" if b["valid"] else "✗"
            c_s = "✓" if c["valid"] else "✗"
            print(f"  trial {trial+1:02d} | baseline {b_s} | controlled {c_s} (i={c['interventions']} corr={c['corrections']}) | {repr(c['text'][:60])}")

        b_valid = sum(r["valid"] for r in baseline_results)
        c_valid = sum(r["valid"] for r in controlled_results)
        b_lat = sum(r["latency_ms"] for r in baseline_results) / N_TRIALS
        c_lat = sum(r["latency_ms"] for r in controlled_results) / N_TRIALS
        print(f"\n  SUMMARY:\n    Baseline   {b_valid}/{N_TRIALS} ({100*b_valid/N_TRIALS:.1f}%)  {b_lat:.0f}ms")
        print(f"    Controlled {c_valid}/{N_TRIALS} ({100*c_valid/N_TRIALS:.1f}%)  {c_lat:.0f}ms")
        print(f"    Delta: {c_valid-b_valid:+d} ({100*(c_valid-b_valid)/N_TRIALS:+.1f}pp)")
        print(f"    Latency overhead: {(c_lat-b_lat)/b_lat*100:+.1f}%")
        all_results.append({"task": task_idx+1, "name": task["name"], "n_trials": N_TRIALS,
            "baseline": {"valid": b_valid, "rate": b_valid/N_TRIALS, "avg_latency_ms": b_lat, "results": baseline_results},
            "controlled": {"valid": c_valid, "rate": c_valid/N_TRIALS, "avg_latency_ms": c_lat, "results": controlled_results}})

    total_b = sum(r["baseline"]["valid"] for r in all_results)
    total_c = sum(r["controlled"]["valid"] for r in all_results)
    total = N_TRIALS * len(TASKS)
    print(f"\n{'='*60}\nOVERALL ({len(TASKS)} tasks x {N_TRIALS} = {total} runs each)\n{'='*60}")
    print(f"  Baseline   {total_b}/{total} ({100*total_b/total:.1f}%)")
    print(f"  Controlled {total_c}/{total} ({100*total_c/total:.1f}%)")
    print(f"  Delta: {total_c-total_b:+d} ({100*(total_c-total_b)/total:+.1f}pp)")
    out = OUT_DIR / "comparison_weak_prompt.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out}")

if __name__ == "__main__":
    main()
