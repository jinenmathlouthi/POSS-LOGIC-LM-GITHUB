from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("data", "results", "examples", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported JSON structure in {path}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_formalisations(item: Dict[str, Any]) -> List[str]:
    values = item.get("formalisations") or item.get("formalizations") or []
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def get_log_scores(item: Dict[str, Any], k: int) -> List[float]:
    scores = item.get("log_scores") or item.get("scores") or []
    if not isinstance(scores, list):
        scores = []

    clean_scores: List[float] = []
    for value in scores[:k]:
        try:
            clean_scores.append(float(value))
        except (TypeError, ValueError):
            clean_scores.append(0.0)

    if len(clean_scores) < k:
        clean_scores.extend([0.0] * (k - len(clean_scores)))

    return clean_scores


def item_id(item: Dict[str, Any], index: int) -> str:
    return str(item.get("id", f"example_{index:04d}"))
