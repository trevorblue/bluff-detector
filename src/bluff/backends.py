"""Generators: the language model under test.

HFBackend  - a real open-weights model via Hugging Face Transformers. Hidden states need
             open weights, which is why hosted APIs cannot be probed.
FakeBackend - a deterministic test double so the whole pipeline, the API and CI run on a
             laptop without a GPU or downloads. Its numbers mean nothing scientifically.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bluff.utils import stable_hash

POSITIONS = ("last_answer", "mean_answer")


@dataclass
class GenerationOutput:
    answer: str
    token_probs: list[float]
    token_entropies: list[float]
    hidden: np.ndarray  # [n_layers, n_positions, hidden_dim]
    latency_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str
    uses_reference: bool  # only the fake backend may see the gold answer

    def generate(
        self, prompt: str, *, question: str, condition: str, reference: str | None = None
    ) -> GenerationOutput: ...


class FakeBackend:
    """Simulates a model that 'knows' some questions and guesses on others.

    Knowledge is a deterministic function of the question text. Kenya-flavoured questions
    are known less often, mirroring the hypothesis. When it does not know, its token
    probabilities are flatter and its hidden states are shifted along a fixed direction
    (stronger in deeper layers), so probes have something learnable. In the offered
    condition it takes the "I don't know" option with a probability scaled by
    `abstain_scale`, which simulates fine-tuning suppressing abstention.
    """

    uses_reference = True
    _KENYA = re.compile(r"\b(kenya|kenyan|county|counties|nairobi|swahili)\b", re.IGNORECASE)
    _DISTRACTORS = ("Mombasa", "1964", "Kisumu", "the Senate", "Nakuru", "52", "Treasury")

    def __init__(
        self,
        n_layers: int,
        n_positions: int = len(POSITIONS),
        hidden_dim: int = 64,
        seed: int = 42,
        abstain_scale: float = 1.0,
        general_know_rate: float = 0.65,
        kenya_know_rate: float = 0.35,
    ) -> None:
        self.name = "fake-llm"
        self.shape = (n_layers, n_positions, hidden_dim)
        self.abstain_scale = abstain_scale
        self.general_know_rate = general_know_rate
        self.kenya_know_rate = kenya_know_rate
        rng = np.random.default_rng(seed)
        d = rng.normal(size=hidden_dim)
        self.direction = d / np.linalg.norm(d)

    def knows(self, question: str) -> bool:
        rate = self.kenya_know_rate if self._KENYA.search(question) else self.general_know_rate
        return stable_hash(question, 10_000) / 10_000 < rate

    def generate(
        self, prompt: str, *, question: str, condition: str, reference: str | None = None
    ) -> GenerationOutput:
        start = time.perf_counter()
        rng = np.random.default_rng(stable_hash(prompt + condition, 2**32))
        knows = self.knows(question)

        n_tok = int(rng.integers(2, 7))
        # Distributions overlap on purpose: output confidence is an imperfect signal.
        if knows:
            probs = rng.uniform(0.45, 0.99, n_tok)
            ents = rng.uniform(0.05, 1.8, n_tok)
        else:
            probs = rng.uniform(0.25, 0.97, n_tok)
            ents = rng.uniform(0.3, 2.4, n_tok)

        if condition == "offered":
            p_abstain = (0.05 if knows else 0.55) * self.abstain_scale
            if rng.random() < p_abstain:
                answer = "I don't know."
            else:
                answer = self._answer(knows, reference, rng)
        else:
            answer = self._answer(knows, reference, rng)

        L, P, D = self.shape
        hidden = rng.normal(size=(L, P, D))
        sign = 1.0 if not knows else -1.0
        for layer in range(L):
            hidden[layer] += sign * 1.2 * ((layer + 1) / L) * self.direction
        latency = (time.perf_counter() - start) * 1000
        return GenerationOutput(
            answer=answer,
            token_probs=probs.tolist(),
            token_entropies=ents.tolist(),
            hidden=hidden.astype(np.float32),
            latency_ms=latency,
            meta={"knows": knows},
        )

    def _answer(self, knows: bool, reference: str | None, rng: np.random.Generator) -> str:
        if knows and reference:
            return reference  # the pipeline passes the correct response as `reference`
        if knows:
            return "[fake-llm answer]"  # serving: no reference available
        return str(rng.choice(self._DISTRACTORS))


class HFBackend:
    """A real causal LM. Greedy decoding, then one forward pass to read hidden states.

    Not exercised in CI (needs `pip install .[llm]` and a model download).
    """

    uses_reference = False

    def __init__(
        self,
        name: str,
        layers: list[int],
        positions: list[str],
        dtype: str = "bfloat16",
        device: str = "auto",
        max_new_tokens: int = 32,
        adapter: str | None = None,
        seed: int = 42,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.name = name if not adapter else f"{name}+{adapter}"
        self.layers = list(layers)
        self.positions = list(positions)
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, dtype), device_map=device
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(
        self, prompt: str, *, question: str, condition: str, reference: str | None = None
    ) -> GenerationOutput:
        torch = self.torch
        start = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        n_prompt = enc.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
            gen_ids = out.sequences[0, n_prompt:]
            scores = torch.stack(out.scores, dim=0)[:, 0, :].float()  # [T, V]
            # Drop the end-of-sequence token and anything after it.
            keep = len(gen_ids)
            eos = (gen_ids == self.tok.eos_token_id).nonzero()
            if len(eos):
                keep = int(eos[0])
            gen_ids, scores = gen_ids[:keep], scores[:keep]
            logp = torch.log_softmax(scores, dim=-1)
            if keep:
                chosen = logp[torch.arange(keep), gen_ids].exp()
                entropy = -(logp.exp() * logp).sum(dim=-1)
            else:
                chosen = entropy = torch.zeros(0)

            full = out.sequences[:, : n_prompt + max(keep, 1)]
            hs = self.model(full, output_hidden_states=True).hidden_states  # (L+1) x [1,S,D]
            span = slice(n_prompt, n_prompt + max(keep, 1))
            rows = []
            for layer in self.layers:
                h = hs[layer][0, span].float()
                per_pos = []
                for pos in self.positions:
                    per_pos.append(h[-1] if pos == "last_answer" else h.mean(dim=0))
                rows.append(torch.stack(per_pos))
            hidden = torch.stack(rows).cpu().numpy()

        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return GenerationOutput(
            answer=answer,
            token_probs=chosen.cpu().tolist(),
            token_entropies=entropy.cpu().tolist(),
            hidden=hidden.astype(np.float32),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_backend(cfg: dict[str, Any]) -> Backend:
    m = cfg["model"]
    if m["backend"] == "fake":
        return FakeBackend(
            n_layers=len(m["layers"]),
            n_positions=len(m["positions"]),
            hidden_dim=m.get("fake_hidden_dim", 64),
            seed=cfg["seed"],
            abstain_scale=m.get("fake_abstain_scale", 1.0),
        )
    if m["backend"] == "hf":
        return HFBackend(
            name=m["name"],
            layers=m["layers"],
            positions=m["positions"],
            dtype=m.get("dtype", "bfloat16"),
            device=m.get("device", "auto"),
            max_new_tokens=m.get("max_new_tokens", 32),
            adapter=m.get("adapter"),
            seed=cfg["seed"],
        )
    raise ValueError(f"unknown backend {m['backend']!r}")
