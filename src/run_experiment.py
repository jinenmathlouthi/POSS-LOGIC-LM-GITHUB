from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import default_database_path, get_dataset_config
from io_utils import load_json_dataset, save_json, write_summary_csv
from logiclm_pipeline import run_logiclm
from metrics import compute_stats, summary_row
from pbs_pipeline import run_pbs
from self_refinement import HFRefiner, RefinementConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run POSS-LOGIC-LM experiments.")

    parser.add_argument("--model", required=True, choices=["qwen", "llama3", "gemma"])
    parser.add_argument("--dataset", required=True, choices=["folio", "prontoqa", "proofwriter"])
    parser.add_argument("--method", required=True, choices=["logiclm", "pbs"])

    parser.add_argument("--repo-root", default=".", help="Repository root containing databases/ and results/.")
    parser.add_argument("--data-path", default=None, help="Optional explicit database JSON path.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N examples.")
    parser.add_argument("--timeout", type=int, default=10, help="Solver timeout in seconds.")

    parser.add_argument("--max-refinements", type=int, default=3)
    parser.add_argument("--refiner-model", default=None, help="HF model name for LogicLM self-refinement.")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    dataset_cfg = get_dataset_config(args.dataset)

    data_path = Path(args.data_path).resolve() if args.data_path else default_database_path(repo_root, args.model, args.dataset)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "results" / args.model
    debug_dir = output_dir / "debug" / args.dataset / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.dataset}")
    print(f"Method     : {args.method}")
    print(f"Solver     : {dataset_cfg.solver}")
    print(f"Data path  : {data_path}")
    print(f"Output dir : {output_dir}")
    print("=" * 80)

    data = load_json_dataset(data_path)
    if args.limit:
        print(f"Running on first {args.limit} examples only.")

    refiner: Optional[HFRefiner] = None
    if args.method == "logiclm" and args.refiner_model:
        refiner = HFRefiner(
            RefinementConfig(
                model_name=args.refiner_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    if args.method == "logiclm":
        results = run_logiclm(
            data=data,
            dataset=args.dataset,
            debug_dir=debug_dir,
            max_refinements=args.max_refinements,
            refiner=refiner,
            timeout=args.timeout,
            limit=args.limit,
        )
        method_name = "LogicLM"
    else:
        results = run_pbs(
            data=data,
            dataset=args.dataset,
            debug_dir=debug_dir,
            timeout=args.timeout,
            limit=args.limit,
        )
        method_name = "PBS"

    stats = compute_stats(results, pred_key="pred")
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "method": method_name,
        "solver": dataset_cfg.solver,
        "max_refinements": args.max_refinements if args.method == "logiclm" else None,
        "refiner_model": args.refiner_model if args.method == "logiclm" else None,
        "transform": "T1_Klir" if args.method == "pbs" else None,
        "stats": stats,
        "results": results,
    }

    json_path = output_dir / f"{args.method}_{args.dataset}_results.json"
    summary_path = output_dir / f"{args.method}_{args.dataset}_summary.csv"

    save_json(json_path, payload)
    write_summary_csv(summary_path, [summary_row(args.model, args.dataset, method_name, stats)])

    print("\nRESULTS")
    print("-" * 80)
    print(f"Total    : {stats['total']}")
    print(f"Correct  : {stats['correct']}")
    print(f"Accuracy : {stats['accuracy']}%")
    print(f"Exe rate : {stats['exe_rate']}%")
    print("-" * 80)
    print(f"Saved JSON   : {json_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

