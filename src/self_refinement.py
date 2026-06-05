from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RefinementConfig:
    model_name: Optional[str] = None
    max_new_tokens: int = 900
    temperature: float = 0.0


class HFRefiner:
    """
    Optional HuggingFace self-refinement module.

    This module is intentionally optional because PBS does not need generation.
    LogicLM reproduction can enable it with --refiner-model.
    """

    def __init__(self, config: RefinementConfig):
        if not config.model_name:
            raise ValueError("HFRefiner requires a model_name.")

        self.config = config
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self._model.eval()

    def refine(
        self,
        dataset: str,
        context: str,
        conclusion: str,
        current_formalisation: str,
        solver_result: Dict[str, Any],
    ) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        prompt = build_refinement_prompt(
            dataset=dataset,
            context=context,
            conclusion=conclusion,
            current_formalisation=current_formalisation,
            solver_result=solver_result,
        )

        messages = [
            {"role": "system", "content": "You correct logical formalizations. Output only the corrected formalization."},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature if self.config.temperature > 0 else None,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        outputs = self._model.generate(**inputs, **generate_kwargs)
        decoded = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return extract_formalization_text(decoded)


def build_refinement_prompt(
    dataset: str,
    context: str,
    conclusion: str,
    current_formalisation: str,
    solver_result: Dict[str, Any],
) -> str:
    error_text = str(solver_result.get("stderr") or solver_result.get("warnings") or "")[:1500]
    prediction = solver_result.get("prediction", "Unknown")

    return f"""
Dataset: {dataset}

Natural language context:
{context}

Target conclusion:
{conclusion}

Current formalization:
{current_formalisation}

Solver prediction: {prediction}
Solver error/warnings:
{error_text}

Task:
Correct the formalization while preserving the same format:
PREDICATES:
...
PREMISES:
...
CONCLUSION:
...

Rules:
- Do not add the target conclusion as a fact.
- Keep predicates lowercase when possible.
- Use universal rules with forall x (... -> ...).
- Use NOT predicate(x) for explicit negation.
- Output only the corrected formalization.
""".strip()


def extract_formalization_text(raw_output: str) -> str:
    text = str(raw_output).strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            candidate = candidate.replace("prolog", "", 1).replace("text", "", 1).strip()
            if candidate:
                return candidate
    return text


def should_refine(result: Dict[str, Any], dataset: str) -> bool:
    if not result.get("success", True):
        return True
    if result.get("prediction") == "Unknown":
        return True
    if result.get("stderr"):
        stderr = str(result.get("stderr"))
        if "syntax error" in stderr.lower() or "missing_query" in stderr.lower() or "empty_program" in stderr.lower():
            return True
    return False
