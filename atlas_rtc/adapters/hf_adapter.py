from __future__ import annotations

import math
from typing import Dict, List, Optional

from atlas_rtc.adapters.base import BaseAdapter, DecodeStep, InterventionDirectives, TokenCandidate

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _count_open_structures(text: str):
    """Return (unclosed_braces, unclosed_brackets, in_string)."""
    braces = brackets = 0
    in_str = escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_str:
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":    braces += 1
            elif ch == "}":  braces -= 1
            elif ch == "[":  brackets += 1
            elif ch == "]":  brackets -= 1
    return max(0, braces), max(0, brackets), in_str


class ATLASLogitsProcessor(LogitsProcessor):
    """Stateful logits processor for HuggingFace generation.

    Applies:
    - Markdown suppression (always)
    - Directive token biases and masks
    - Structure-aware closure biasing (} ] " when structure is open)
    - EOS forcing ONLY when structure is fully complete + all keys present
    - Endgame closure pressure in final 30% of token budget
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_bias: Dict[int, float] = {}
        self.masked_ids: List[int] = []
        self.open_braces: int = 0
        self.open_brackets: int = 0
        self.in_string: bool = False
        self.endgame_mode: bool = False
        self.structure_complete: bool = False
        self._eos_id = tokenizer.eos_token_id

        # Pre-compute token ID sets
        self._close_brace_ids: List[int] = []
        self._close_bracket_ids: List[int] = []
        self._quote_ids: List[int] = []
        self._open_ids: List[int] = []
        self._suppress_ids: List[int] = []

        for tok in ["}", '"}', "}\n", "},", '}"']:
            self._close_brace_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        for tok in ["]", "],", "]\n"]:
            self._close_bracket_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        for tok in ['"']:
            self._quote_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        for tok in ["{", "[", ","]:
            self._open_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        for tok in ["`", "```", "```json", " ```", "\n```"]:
            self._suppress_ids.extend(tokenizer.encode(tok, add_special_tokens=False))

        self._close_brace_ids   = list(set(self._close_brace_ids))
        self._close_bracket_ids = list(set(self._close_bracket_ids))
        self._quote_ids         = list(set(self._quote_ids))
        self._open_ids          = list(set(self._open_ids))
        self._suppress_ids      = list(set(self._suppress_ids))

    def update(
        self,
        directives: InterventionDirectives,
        current_text: str,
        step_index: int,
        max_new_tokens: int,
        required_keys: List[str] = None,
    ) -> None:
        self.token_bias = {}
        self.masked_ids = []

        for tok_str, bias in directives.token_bias.items():
            ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            for tid in ids:
                self.token_bias[tid] = float(bias)

        for tok_str in directives.masked_tokens:
            ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            self.masked_ids.extend(ids)

        self.open_braces, self.open_brackets, self.in_string = _count_open_structures(current_text)

        # Structure complete = braces balanced + has opened + not in string
        braces_balanced = (self.open_braces == 0 and "{" in current_text and not self.in_string)

        # All required keys present
        keys_present = True
        if required_keys and braces_balanced:
            keys_present = all(f'"{k}"' in current_text for k in required_keys)

        open_b, open_br, in_s = _count_open_structures(current_text)
        self.structure_complete = braces_balanced and keys_present and step_index > 3 and open_b == 0 and not in_s

        # Endgame: last 30% of budget
        self.endgame_mode = step_index > int(0.70 * max_new_tokens)

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
        vocab_size = scores.shape[-1]

        # Always suppress markdown
        for tid in self._suppress_ids:
            if tid < vocab_size:
                scores[:, tid] = -float("inf")

        # Apply directive masks
        for tid in self.masked_ids:
            if tid < vocab_size:
                scores[:, tid] = -float("inf")

        # Apply directive biases
        for tid, bias in self.token_bias.items():
            if tid < vocab_size:
                scores[:, tid] += bias

        # Structure-aware closure — always active when structure is incomplete
        if self.open_braces > 0:
            strength = 4.0 * min(self.open_braces, 3)
            for tid in self._close_brace_ids:
                if tid < vocab_size:
                    scores[:, tid] += strength
        if self.open_brackets > 0:
            strength = 4.0 * min(self.open_brackets, 3)
            for tid in self._close_bracket_ids:
                if tid < vocab_size:
                    scores[:, tid] += strength
        if self.in_string:
            for tid in self._quote_ids:
                if tid < vocab_size:
                    scores[:, tid] += 6.0

        # Endgame pressure — bias closing tokens, penalise opening tokens
        if self.endgame_mode and not self.structure_complete:
            for tid in self._close_brace_ids + self._close_bracket_ids:
                if tid < vocab_size:
                    scores[:, tid] += 6.0
            for tid in self._open_ids:
                if tid < vocab_size:
                    scores[:, tid] -= 4.0

        # EOS forcing — ONLY when structure is truly complete with all keys
        if self.structure_complete and self._eos_id is not None and self._eos_id < vocab_size:
            scores[:, self._eos_id] += 20.0

        return scores


class HuggingFaceAdapter(BaseAdapter):
    """Full-control HuggingFace adapter for ATLAS-RTC.

    True logit tensor access via transformers LogitsProcessor.
    Supports token biasing, masking, KV cache rollback, and
    structure-aware EOS forcing.
    """

    name = "hf"
    supports_logit_control = True

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        top_k: int = 20,
        device: str = "auto",
        required_keys: List[str] = None,
    ):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed.")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._top_k = top_k
        self._required_keys = required_keys or []

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        self._logits_processor = ATLASLogitsProcessor(self.tokenizer)
        self._processor_list = LogitsProcessorList([self._logits_processor])

        self._prompt = ""
        self._prompt_ids: Optional["torch.LongTensor"] = None
        self._generated_ids: List[int] = []
        self._past_key_values = None
        self._step_index = 0
        self._finished = False
        self._current_text = ""
        self._device = next(self.model.parameters()).device

    def set_required_keys(self, keys: List[str]) -> None:
        """Update required keys for EOS gating — call before each run."""
        self._required_keys = keys

    def initialize(self, prompt: str) -> None:
        self._prompt = prompt
        self._prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        self._generated_ids = []
        self._past_key_values = None
        self._step_index = 0
        self._finished = False
        self._current_text = ""

    def restart(self, prompt: str) -> None:
        self.initialize(prompt)

    def rollback(self, n: int) -> None:
        if n <= 0:
            return
        if n > len(self._generated_ids):
            raise ValueError(f"Cannot rollback {n}; only {len(self._generated_ids)} generated.")
        self._generated_ids = self._generated_ids[:-n]
        self._step_index -= n
        # Decode keeping special tokens so we can detect EOS
        raw = self.tokenizer.decode(self._generated_ids, skip_special_tokens=False)
        # Strip EOS/pad from display text but keep for detection
        self._current_text = self.tokenizer.decode(self._generated_ids, skip_special_tokens=True)
        self._finished = False
        # Truncate KV cache
        if self._past_key_values is not None:
            try:
                self._past_key_values = tuple(
                    tuple(t[..., :-n, :] for t in layer)
                    for layer in self._past_key_values
                )
            except Exception:
                self._past_key_values = None

    def is_finished(self) -> bool:
        return self._finished

    def get_text(self) -> str:
        return self._current_text

    def step(self, directives: InterventionDirectives | None = None) -> DecodeStep:
        directives = directives or InterventionDirectives()

        self._logits_processor.update(
            directives=directives,
            current_text=self._current_text,
            step_index=self._step_index,
            max_new_tokens=self.max_new_tokens,
            required_keys=self._required_keys,
        )

        temperature = max(directives.temperature or 1.0, 1e-6)

        with torch.no_grad():
            if self._past_key_values is None:
                full_ids = torch.cat([
                    self._prompt_ids,
                    torch.tensor([self._generated_ids], device=self._device)
                ], dim=1) if self._generated_ids else self._prompt_ids
            else:
                full_ids = torch.tensor([[self._generated_ids[-1]]], device=self._device)

            outputs = self.model(
                input_ids=full_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
                return_dict=True,
            )

        self._past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        input_ids_so_far = torch.cat([
            self._prompt_ids,
            torch.tensor([self._generated_ids], device=self._device) if self._generated_ids
            else torch.zeros((1, 0), dtype=torch.long, device=self._device),
        ], dim=1)
        logits = self._processor_list(input_ids_so_far, logits)
        logits = logits / temperature

        top_k_logits, top_k_ids = torch.topk(logits[0], min(self._top_k, logits.shape[-1]))
        top_k_probs = torch.softmax(top_k_logits, dim=-1)

        probs = torch.softmax(logits[0], dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        self._generated_ids.append(next_token_id)
        # Decode keeping special tokens so we can detect EOS
        raw = self.tokenizer.decode(self._generated_ids, skip_special_tokens=False)
        # Strip EOS/pad from display text but keep for detection
        self._current_text = self.tokenizer.decode(self._generated_ids, skip_special_tokens=True)

        eos_id = self.tokenizer.eos_token_id
        if next_token_id == eos_id or self._step_index >= self.max_new_tokens - 1 or (eos_id in self._generated_ids):
            self._finished = True

        top_k_candidates = [
            TokenCandidate(
                token=self.tokenizer.decode([tid], skip_special_tokens=False),
                logit=logit_val,
                probability=prob_val,
            )
            for logit_val, tid, prob_val in zip(
                top_k_logits.tolist(), top_k_ids.tolist(), top_k_probs.tolist()
            )
        ]

        entropy = sum(
            -c.probability * math.log(c.probability)
            for c in top_k_candidates if c.probability > 0
        )

        step = DecodeStep(
            step_index=self._step_index,
            generated_text=self._current_text,
            top_k=top_k_candidates,
            entropy=entropy,
        )
        self._step_index += 1
        return step
