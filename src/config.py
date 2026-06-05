from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    labels: Tuple[str, ...]
    solver: str
    binary: bool
    default_database_name: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    database_dir: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "folio": DatasetConfig(
        name="folio",
        labels=("True", "False", "Unknown"),
        solver="prover9",
        binary=False,
        default_database_name="folio_beam_database.json",
    ),
    "prontoqa": DatasetConfig(
        name="prontoqa",
        labels=("True", "False"),
        solver="prolog",
        binary=True,
        default_database_name="prontoqa_beam_database.json",
    ),
    "proofwriter": DatasetConfig(
        name="proofwriter",
        labels=("True", "False", "Unknown"),
        solver="prolog",
        binary=False,
        default_database_name="proofwriter_beam_database.json",
    ),
}


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen": ModelConfig(name="qwen", database_dir="qwen"),
    "llama3": ModelConfig(name="llama3", database_dir="llama3"),
    "gemma": ModelConfig(name="gemma", database_dir="gemma"),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    key = dataset.lower().strip()
    if key not in DATASET_CONFIGS:
        valid = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Valid values: {valid}")
    return DATASET_CONFIGS[key]


def get_model_config(model: str) -> ModelConfig:
    key = model.lower().strip()
    if key not in MODEL_CONFIGS:
        valid = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model '{model}'. Valid values: {valid}")
    return MODEL_CONFIGS[key]


def default_database_path(repo_root: Path, model: str, dataset: str) -> Path:
    model_cfg = get_model_config(model)
    dataset_cfg = get_dataset_config(dataset)
    return repo_root / "databases" / model_cfg.database_dir / dataset_cfg.default_database_name


def normalize_gold_label(label: object, dataset: str) -> str:
    text = str(label).strip().lower()

    if text in {"true", "proved", "yes"}:
        return "True"
    if text in {"false", "disproved", "no"}:
        return "False"
    if text in {"unknown", "uncertain", "unk", "neutral"}:
        return "Unknown"

    dataset_cfg = get_dataset_config(dataset)
    if dataset_cfg.binary:
        raise ValueError(f"Invalid binary label for {dataset}: {label}")

    return "Unknown"

