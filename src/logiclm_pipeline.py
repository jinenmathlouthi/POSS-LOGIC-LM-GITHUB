from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_dataset_config, normalize_gold_label
from io_utils import get_formalisations, item_id
from self_refinement import HFRefiner, should_refine
from solvers.prolog_solver import solve_formalisation_with_prolog
from solvers.prover9_solver import solve_formalisation_with_prover9


def solve_candidate(
    dataset: str,
    formalisation: str,
    conclusion: str,
    debug_dir: Path,
    run_id: str,
    timeout: int,
) -> Dict[str, Any]:
    cfg = get_dataset_config(dataset)

    if cfg.solver == "prolog":
        return solve_formalisation_with_prolog(
            formalisation=formalisation,
            conclusion=conclusion,
            dataset=dataset,
            debug_dir=debug_dir,
            run_id=run_id,
            timeout=timeout,
        )

    if cfg.solver == "prover9":
        return solve_formalisation_with_prover9(
            formalisation=formalisation,
            debug_dir=debug_dir,
            run_id=run_id,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported solver: {cfg.solver}")


def evaluate_logiclm_item(
    item: Dict[str, Any],
    index: int,
    dataset: str,
    debug_dir: Path,
    max_refinements: int = 3,
    refiner: Optional[HFRefiner] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    example_id = item_id(item, index)
    gold = normalize_gold_label(item.get("label"), dataset)
    conclusion = str(item.get("conclusion", ""))
    context = str(item.get("context", ""))
    formalisations = get_formalisations(item)

    if not formalisations:
        return {
            "id": example_id,
            "gold": gold,
            "pred": "Unknown",
            "correct": gold == "Unknown",
            "steps": [],
            "error": "missing_formalisations",
        }

    current = formalisations[0]
    steps: List[Dict[str, Any]] = []

    for step in range(max_refinements + 1):
        result = solve_candidate(
            dataset=dataset,
            formalisation=current,
            conclusion=conclusion,
            debug_dir=debug_dir,
            run_id=f"{example_id}_logiclm_step_{step}",
            timeout=timeout,
        )

        steps.append({
            "step": step,
            "prediction": result.get("prediction", "Unknown"),
            "success": result.get("success"),
            "query": result.get("query"),
            "conclusion": result.get("conclusion"),
            "warnings": result.get("warnings", []),
            "stderr": result.get("stderr", ""),
            "time_s": result.get("time_s", 0.0),
        })

        if step >= max_refinements:
            break

        if not refiner or not should_refine(result, dataset):
            break

        current = refiner.refine(
            dataset=dataset,
            context=context,
            conclusion=conclusion,
            current_formalisation=current,
            solver_result=result,
        )

    pred = steps[-1]["prediction"] if steps else "Unknown"

    if get_dataset_config(dataset).binary and pred == "Unknown":
        pred = "False"

    return {
        "id": example_id,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "steps_used": len(steps) - 1,
        "steps": steps,
    }


def run_logiclm(
    data: List[Dict[str, Any]],
    dataset: str,
    debug_dir: Path,
    max_refinements: int = 3,
    refiner: Optional[HFRefiner] = None,
    timeout: int = 10,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    examples = data[:limit] if limit else data
    results: List[Dict[str, Any]] = []

    for index, item in enumerate(examples):
        results.append(
            evaluate_logiclm_item(
                item=item,
                index=index,
                dataset=dataset,
                debug_dir=debug_dir,
                max_refinements=max_refinements,
                refiner=refiner,
                timeout=timeout,
            )
        )

    return results
