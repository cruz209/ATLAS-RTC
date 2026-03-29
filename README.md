# ATLAS-RTC

Token-level runtime control for LLM generation. Treats decoding as a closed-loop control problem:

```
observe → detect drift → intervene → verify → log
```

Supports real logit-level control (masking, biasing, temperature, rollback+re-steer) via vLLM on GPU, and black-box mock testing locally without any model.

---

## Quickstart — local (no GPU, mock adapter)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Smoke test
python -m atlas_rtc.examples.run_json_demo

# Full benchmark
python -m atlas_rtc.benchmarks.runners.run_benchmark \
  --config atlas_rtc/experiments/configs/json_baseline.yaml
```

---

## Quickstart — vast.ai GPU (full logit control)

### 1. Spin up an instance

- GPU: **RTX 4090** (24 GB VRAM) — fits Qwen2.5-7B comfortably
- Template: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
- Disk: 40 GB minimum (model weights ~15 GB)

### 2. SSH in and install

```bash
git clone <your-repo> atlas_rtc && cd atlas_rtc

# GPU install — pulls vLLM, torch, transformers
pip install -e ".[gpu]"
```

### 3. Run against Qwen2.5-7B-Instruct

```bash
python -m atlas_rtc.examples.run_vllm_demo
```

On first run this downloads `Qwen/Qwen2.5-7B-Instruct` (~15 GB). Cached after that.

Expected output:
```
Loading Qwen/Qwen2.5-7B-Instruct...
Running controller...

Output:         {"name": "Alice", "age": 30, "city": "Paris"}
Valid:          True
Errors:         []
Interventions:  9
Corrections:    0

Action trace:
  [00] bias          initial structural priming
  [01] noop          trajectory appears healthy
  ...
```

### 4. Tuning the controller

To make corrections fire more aggressively during testing, lower the `critical` threshold:

```python
from atlas_rtc.controller.policy import LadderPolicy, PolicyThresholds

policy = LadderPolicy(
    thresholds=PolicyThresholds(low=0.2, medium=0.4, high=0.7, critical=0.5),
    rollback_depth=2,
    max_corrections=3,
)
```

### 5. Inspect telemetry

Every run writes a JSONL trace to `outputs/vllm_run.jsonl`. Each line is one event:

```bash
# See all actions that fired
jq 'select(.name == "action")' outputs/vllm_run.jsonl

# See drift score per step
jq 'select(.name == "decode_step") | {step: .payload.step_index, drift: .payload.drift_score}' outputs/vllm_run.jsonl

# Check if any corrections fired
jq 'select(.payload.action == "correct")' outputs/vllm_run.jsonl
```

---

## Architecture

```
adapters/       model backends — MockAdapter (tests), VLLMAdapter (GPU), HFAdapter (stub)
contracts/      output contracts — JSONSchema, ToolCall, Extraction
detectors/      FeatureExtractor, HeuristicDriftDetector, LightweightLogisticDetector
controller/     RuntimeState, ControlActions, LadderPolicy, RuntimeController
verification/   Verifier, SimpleRepair
telemetry/      EventLogger (JSONL), plots
benchmarks/     tasks, runners, metrics
experiments/    YAML config-driven runs
tests/          unit + integration
```

## Control ladder

| Drift score | Action |
|---|---|
| < 0.25 | noop |
| 0.25 – 0.4 | structural bias |
| 0.4 – 0.7 | temperature clamp |
| 0.7 – 0.85 | token masking |
| ≥ 0.85 | **rollback N tokens + re-steer** |
| budget exhausted | full restart (last resort) |

## Model recommendations

| Model | VRAM | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 18 GB | Default. Good at structured outputs. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 16 GB | Slightly more drift — interesting test subject. |
| `mistralai/Mistral-Nemo-Instruct-2407` | 26 GB | 12B, needs A100 or 2x 4090. |

---

## Research framing

ATLAS-RTC is a runtime control plane for LLM systems, analogous to feedback controllers in distributed systems infrastructure. The core claim is not "better JSON prompting" — it is that online, stateful, token-level control with mid-step correction is a distinct runtime abstraction for reliable LLM deployment.
