from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_dataset_config, normalize_gold_label
from io_utils import get_formalisations, get_log_scores, item_id
from logiclm_pipeline import solve_candidate
from possibility import (
    aggregate_possibility_by_label,
    aggregate_weight_by_label,
    binary_necessity,
    choose_binary_label,
    choose_ternary_label,
    log_scores_to_possibilities,
)


def evaluate_pbs_item(
    item: Dict[str, Any],
    index: int,
    dataset: str,
    debug_dir: Path,
    timeout: int = 10,
) -> Dict[str, Any]:
    cfg = get_dataset_config(dataset)
    example_id = item_id(item, index)
    gold = normalize_gold_label(item.get("label"), dataset)
    conclusion = str(item.get("conclusion", ""))
    formalisations = get_formalisations(item)

    if not formalisations:
        pred = "False" if cfg.binary else "Unknown"
        return {
            "id": example_id,
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "solver": [],
            "poss": [],
            "pi": {label: 0.0 for label in cfg.labels},
            "W": {label: 0.0 for label in cfg.labels},
            "error": "missing_formalisations",
        }

    k = len(formalisations)
    log_scores = get_log_scores(item, k)
    possibilities = log_scores_to_possibilities(log_scores)

    candidate_results: List[Dict[str, Any]] = []
    candidate_predictions: List[str] = []

    for candidate_index, formalisation in enumerate(formalisations):
        result = solve_candidate(
            dataset=dataset,
            formalisation=formalisation,
            conclusion=conclusion,
            debug_dir=debug_dir,
            run_id=f"{example_id}_F{candidate_index + 1}",
            timeout=timeout,
        )

        pred_i = str(result.get("prediction", "Unknown"))
        if cfg.binary and pred_i == "Unknown":
            pred_i = "False"

        candidate_predictions.append(pred_i)
        candidate_results.append({
            "candidate": f"F{candidate_index + 1}",
            "log_score": log_scores[candidate_index] if candidate_index < len(log_scores) else None,
            "possibility": possibilities[candidate_index] if candidate_index < len(possibilities) else 0.0,
            "prediction": pred_i,
            "success": result.get("success"),
            "query": result.get("query"),
            "conclusion": result.get("conclusion"),
            "warnings": result.get("warnings", []),
            "stderr": result.get("stderr", ""),
            "time_s": result.get("time_s", 0.0),
        })

    label_space = cfg.labels
    pi = aggregate_possibility_by_label(candidate_predictions, possibilities, label_space)
    W = aggregate_weight_by_label(candidate_predictions, possibilities, label_space)

    if cfg.binary:
        N = binary_necessity(pi)
        pred = choose_binary_label(pi)
    else:
        N = {}
        pred = choose_ternary_label(candidate_predictions, possibilities, dataset=dataset)

    return {
        "id": example_id,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "solver": candidate_predictions,
        "pi": pi,
        "N": N,
        "W": W,
        "poss": possibilities,
        "candidate_results": candidate_results,
        "error": None,
    }


def run_pbs(
    data: List[Dict[str, Any]],
    dataset: str,
    debug_dir: Path,
    timeout: int = 10,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    examples = data[:limit] if limit else data
    results: List[Dict[str, Any]] = []

    for index, item in enumerate(examples):
        results.append(
            evaluate_pbs_item(
                item=item,
                index=index,
                dataset=dataset,
                debug_dir=debug_dir,
                timeout=timeout,
            )
        )

    return results
