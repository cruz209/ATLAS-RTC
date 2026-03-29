from __future__ import annotations

import math
from typing import Dict, List

from atlas_rtc.adapters.base import BaseAdapter, DecodeStep, InterventionDirectives, TokenCandidate

# vLLM is only available on the vast.ai instance — guard the import.
try:
    from vllm import LLM, SamplingParams
    import torch
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class ATLASLogitsProcessor:
    """Injected into vLLM's sampling pipeline to apply ATLAS-RTC directives.

    vLLM calls this before sampling with the full logits tensor.
    We apply token_bias and masked_tokens in-place.
    """

    def __init__(self) -> None:
        self.token_bias: Dict[int, float] = {}
        self.masked_token_ids: List[int] = []

    def update(self, directives: InterventionDirectives, tokenizer) -> None:
        self.token_bias = {}
        self.masked_token_ids = []
        for token_str, bias in directives.token_bias.items():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            for tid in ids:
                self.token_bias[tid] = bias
        for token_str in directives.masked_tokens:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            self.masked_token_ids.extend(ids)

    def __call__(self, token_ids, logits):
        for tid in self.masked_token_ids:
            if tid < logits.shape[-1]:
                logits[tid] = float("-inf")
        for tid, bias in self.token_bias.items():
            if tid < logits.shape[-1]:
                logits[tid] += bias
        return logits


class VLLMAdapter(BaseAdapter):
    """In-process vLLM adapter with full logit control.

    Runs vLLM as a library (not a server) so we can intercept logits
    at each decode step via a custom LogitsProcessor.

    Usage on vast.ai:
        adapter = VLLMAdapter("Qwen/Qwen2.5-7B-Instruct")
        adapter.initialize("Return a JSON object with keys name and age.")
        while not adapter.is_finished():
            step = adapter.step(directives)
    """

    name = "vllm"
    supports_logit_control = True

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        top_k: int = 10,
        dtype: str = "bfloat16",
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Run: pip install vllm  (on your vast.ai GPU instance)"
            )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._top_k = top_k
        self._prompt = ""
        self._token_history: List[str] = []
        self._step_index = 0
        self._finished = False
        self._current_text = ""

        self._logits_processor = ATLASLogitsProcessor()
        self._llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=4096,
            enable_prefix_caching=True,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def initialize(self, prompt: str) -> None:
        self._prompt = prompt
        self._current_text = ""
        self._token_history = []
        self._step_index = 0
        self._finished = False

    def restart(self, prompt: str) -> None:
        self.initialize(prompt)

    def rollback(self, n: int) -> None:
        if n <= 0:
            return
        if n > len(self._token_history):
            raise ValueError(
                f"Cannot rollback {n} tokens; only {len(self._token_history)} generated."
            )
        self._token_history = self._token_history[:-n]
        self._step_index -= n
        self._current_text = "".join(self._token_history)
        self._finished = False

    def is_finished(self) -> bool:
        return self._finished

    def get_text(self) -> str:
        return self._current_text

    def step(self, directives: InterventionDirectives | None = None) -> DecodeStep:
        directives = directives or InterventionDirectives()

        self._logits_processor.update(directives, self._tokenizer)
        temperature = directives.temperature if directives.temperature is not None else 1.0

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=max(temperature, 1e-6),
            top_k=self._top_k,
            logprobs=self._top_k,
            logits_processors=[self._logits_processor],
        )

        full_prompt = self._prompt + self._current_text
        outputs = self._llm.generate([full_prompt], sampling_params, use_tqdm=False)
        output = outputs[0].outputs[0]

        chosen_token = output.text
        self._token_history.append(chosen_token)
        self._current_text += chosen_token

        if output.finish_reason == "stop" or self._step_index >= self.max_new_tokens - 1:
            self._finished = True

        top_k_candidates = self._build_candidates(output)
        entropy = self._entropy(top_k_candidates)

        step = DecodeStep(
            step_index=self._step_index,
            generated_text=self._current_text,
            top_k=top_k_candidates,
            entropy=entropy,
        )
        self._step_index += 1
        return step

    def _build_candidates(self, output) -> List[TokenCandidate]:
        candidates = []
        if output.logprobs:
            logprob_dict = output.logprobs[0]
            items = [(self._tokenizer.decode([tid]), lp.logprob) for tid, lp in logprob_dict.items()]
            max_lp = max(lp for _, lp in items)
            exps = [(tok, math.exp(lp - max_lp)) for tok, lp in items]
            denom = sum(e for _, e in exps) or 1.0
            for tok, e in sorted(exps, key=lambda x: x[1], reverse=True):
                candidates.append(TokenCandidate(
                    token=tok,
                    logit=dict(items)[tok],
                    probability=e / denom,
                ))
        return candidates

    @staticmethod
    def _entropy(candidates: List[TokenCandidate]) -> float:
        total = 0.0
        for c in candidates:
            if c.probability > 0:
                total -= c.probability * math.log(c.probability)
        return total
