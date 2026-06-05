from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def log_scores_to_possibilities(log_scores: Sequence[float]) -> List[float]:
    """
    T1/Klir-style transformation used in the experiments.

    Given beam-level log scores s_i, we compute:
        pi_i = exp(s_i - max_j s_j)

    Therefore, the best beam candidate receives possibility 1.0 and the
    remaining candidates receive values in [0, 1].
    """
    if not log_scores:
        return []

    scores = [float(s) for s in log_scores]
    max_score = max(scores)
    return [math.exp(s - max_score) for s in scores]


def aggregate_possibility_by_label(
    labels: Sequence[str],
    possibilities: Sequence[float],
    label_space: Iterable[str],
) -> Dict[str, float]:
    pi: Dict[str, float] = {}
    for label in label_space:
        values = [
            float(possibilities[i])
            for i, current_label in enumerate(labels)
            if current_label == label and i < len(possibilities)
        ]
        pi[label] = max(values) if values else 0.0
    return pi


def aggregate_weight_by_label(
    labels: Sequence[str],
    possibilities: Sequence[float],
    label_space: Iterable[str],
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for label in label_space:
        weights[label] = sum(
            float(possibilities[i])
            for i, current_label in enumerate(labels)
            if current_label == label and i < len(possibilities)
        )
    return weights


def binary_necessity(pi: Dict[str, float]) -> Dict[str, float]:
    return {
        "True": max(0.0, 1.0 - float(pi.get("False", 0.0))),
        "False": max(0.0, 1.0 - float(pi.get("True", 0.0))),
    }


def choose_binary_label(pi: Dict[str, float]) -> str:
    necessity = binary_necessity(pi)
    true_key = (float(pi.get("True", 0.0)), float(necessity.get("True", 0.0)))
    false_key = (float(pi.get("False", 0.0)), float(necessity.get("False", 0.0)))
    return "True" if true_key > false_key else "False"


def choose_ternary_label(
    labels: Sequence[str],
    possibilities: Sequence[float],
    dataset: str,
) -> str:
    """
    Ternary decision used for FOLIO and ProofWriter.

    - If no True/False candidate is executable, return Unknown.
    - For ProofWriter, choose between True and False using total support W,
      then max possibility pi.
    - For FOLIO, use the stricter rule used in the Prover9 notebook:
      accept a True/False decision if it has at least two supporting candidates,
      or one support among F1-F3; otherwise return Unknown.
    """
    label_space = ("True", "False", "Unknown")
    weights = aggregate_weight_by_label(labels, possibilities, label_space)
    pi = aggregate_possibility_by_label(labels, possibilities, label_space)

    if weights["True"] <= 0.0 and weights["False"] <= 0.0:
        return "Unknown"

    if dataset == "folio":
        support_indices = {
            "True": [i for i, label in enumerate(labels) if label == "True"],
            "False": [i for i, label in enumerate(labels) if label == "False"],
        }
        best_label = max(
            ("True", "False"),
            key=lambda x: (len(support_indices[x]), weights[x], pi[x]),
        )
        supports = support_indices[best_label]
        if len(supports) >= 2:
            return best_label
        if len(supports) == 1 and supports[0] <= 2:
            return best_label
        return "Unknown"

    return max(("True", "False"), key=lambda x: (weights[x], pi[x]))
