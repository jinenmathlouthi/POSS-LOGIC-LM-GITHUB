from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def compute_stats(results: List[Dict[str, Any]], pred_key: str = "pred") -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if bool(r.get("correct")))
    executable = sum(1 for r in results if r.get(pred_key) != "Unknown")

    return {
        "total": total,
        "correct": correct,
        "accuracy": round((correct / total * 100.0) if total else 0.0, 2),
        "executable": executable,
        "exe_rate": round((executable / total * 100.0) if total else 0.0, 2),
        "gold_distribution": dict(Counter(str(r.get("gold")) for r in results)),
        "prediction_distribution": dict(Counter(str(r.get(pred_key)) for r in results)),
    }


def summary_row(
    model: str,
    dataset: str,
    method: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "method": method,
        "total": stats.get("total", 0),
        "correct": stats.get("correct", 0),
        "accuracy": stats.get("accuracy", 0.0),
        "exe_rate": stats.get("exe_rate", 0.0),
    }
