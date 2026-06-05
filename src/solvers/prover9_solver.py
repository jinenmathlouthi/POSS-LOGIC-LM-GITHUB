from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROVER9_TIMEOUT = 10


def find_prover9_binary() -> str:
    """
    Find Prover9.

    Priority:
    1. PROVER9_BIN environment variable
    2. executable named prover9 in PATH
    3. pyprover9 package executable used on Kaggle/Colab
    """
    env_path = os.environ.get("PROVER9_BIN")
    if env_path and Path(env_path).exists():
        return env_path

    path = shutil.which("prover9")
    if path:
        return path

    try:
        import pyprover9  # type: ignore

        package_dir = Path(pyprover9.__file__).resolve().parent
        candidates = list(package_dir.glob("executables/prover9*"))
        for candidate in candidates:
            if candidate.is_file():
                candidate.chmod(0o755)
                return str(candidate)
    except Exception:
        pass

    raise FileNotFoundError(
        "Prover9 was not found. Install pyprover9 or set PROVER9_BIN. "
        "On Kaggle/Colab: !pip install pyprover9"
    )


def normalize_prover9(line: str) -> str:
    line = str(line).strip().rstrip(".")
    line = re.sub(r"\bforall\b", "all", line, flags=re.I)
    line = re.sub(r"\bexists\b", "exists", line, flags=re.I)
    line = line.replace("\\/", "|")
    line = line.replace("/\\", "&")
    line = line.replace(" V ", " | ")
    line = line.replace(" v ", " | ")
    line = line.replace(" OR ", " | ")
    line = line.replace(" AND ", " & ")
    line = line.replace("NOT ", "-")
    line = line.replace("not ", "-")
    line = line.replace("¬", "-")
    line = line.replace("→", "->")
    line = line.replace("⇒", "->")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_valid_formula(line: str) -> bool:
    if not line or line.strip() == ".":
        return False
    if line.count("(") != line.count(")"):
        return False
    if re.search(r"[a-z_]+\s+[A-Za-z_]+\(", line):
        return False
    if ":::" in line:
        return False
    return True


def clean_formula_for_compare(s: str) -> str:
    s = str(s).lower().strip().rstrip(".")
    s = s.replace(" ", "")
    s = s.replace("-", "neg_")
    return s


def extract_conclusion(formalization: str) -> Optional[str]:
    m = re.search(r"CONCLUSION:\s*(.*)", str(formalization), re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    lines = [x.strip() for x in m.group(1).splitlines() if x.strip()]
    if not lines:
        return None

    conclusion = normalize_prover9(lines[0])
    if not is_valid_formula(conclusion):
        return None
    return conclusion


def conclusion_atom_for_filter(conclusion: Optional[str]) -> Optional[str]:
    if not conclusion:
        return None

    c = normalize_prover9(conclusion)
    if any(op in c for op in ("->", "|", "&", "<->")):
        return None

    return clean_formula_for_compare(c)


def extract_premises(formalization: str) -> Tuple[List[str], bool, List[str]]:
    lines: List[str] = []
    warnings: List[str] = []
    in_block = False
    conclusion = extract_conclusion(formalization)
    forbidden = conclusion_atom_for_filter(conclusion)
    injected = False

    for raw_line in str(formalization).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith(("PREMISES", "FACTS", "NEGATIONS", "RULES")):
            in_block = True
            continue

        if upper.startswith("CONCLUSION"):
            break

        if not in_block:
            continue

        if ":::" in line:
            continue

        normalized = normalize_prover9(line)
        if not is_valid_formula(normalized):
            warnings.append(f"ignored_formula: {line[:120]}")
            continue

        if forbidden:
            is_simple_fact = not any(op in normalized for op in ("->", "|", "&", "<->"))
            if is_simple_fact and clean_formula_for_compare(normalized) == forbidden:
                injected = True
                continue

        lines.append(normalized)

    if not lines:
        warnings.append("empty_premises")

    return lines, injected, warnings


def prover9_run(
    premises: List[str],
    goal: str,
    debug_dir: Path,
    run_id: str,
    timeout: int = PROVER9_TIMEOUT,
) -> Dict[str, Any]:
    prover9_bin = find_prover9_binary()
    debug_dir.mkdir(parents=True, exist_ok=True)

    problem = "formulas(assumptions).\n"
    for premise in premises:
        premise = premise.strip().rstrip(".")
        if premise:
            problem += premise + ".\n"
    problem += "end_of_list.\n\n"
    problem += "formulas(goals).\n"
    problem += goal.strip().rstrip(".") + ".\n"
    problem += "end_of_list.\n"

    input_file = debug_dir / f"{_safe_filename(run_id)}.in"
    input_file.write_text(problem, encoding="utf-8")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [prover9_bin, "-f", str(input_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 3)
        stdout_lower = proc.stdout.lower()
        proved = (
            "* proved *" in stdout_lower
            or "*proved*" in stdout_lower
            or "theorem proved" in stdout_lower
            or "max_proofs" in stdout_lower
        )
        return {
            "proved": proved,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "input_file": str(input_file),
            "time_s": elapsed,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "proved": False,
            "stdout": exc.stdout or "",
            "stderr": "PYTHON_TIMEOUT",
            "returncode": -1,
            "input_file": str(input_file),
            "time_s": round(time.time() - t0, 3),
        }
    except Exception as exc:
        return {
            "proved": False,
            "stdout": "",
            "stderr": repr(exc),
            "returncode": -1,
            "input_file": str(input_file),
            "time_s": round(time.time() - t0, 3),
        }


def prover9_decision(
    premises: List[str],
    conclusion: str,
    debug_dir: Path,
    run_id: str,
    timeout: int = PROVER9_TIMEOUT,
) -> Dict[str, Any]:
    proved_run = prover9_run(premises, conclusion, debug_dir, f"{run_id}_prove", timeout=timeout)
    disproved_run = prover9_run(premises, f"-({conclusion})", debug_dir, f"{run_id}_disprove", timeout=timeout)

    proved = bool(proved_run["proved"])
    disproved = bool(disproved_run["proved"])

    if proved and disproved:
        prediction = "Unknown"
    elif proved:
        prediction = "True"
    elif disproved:
        prediction = "False"
    else:
        prediction = "Unknown"

    return {
        "success": True,
        "prediction": prediction,
        "proved": proved,
        "disproved": disproved,
        "prove_run": proved_run,
        "disprove_run": disproved_run,
        "time_s": round(float(proved_run.get("time_s", 0.0)) + float(disproved_run.get("time_s", 0.0)), 3),
    }


def solve_formalisation_with_prover9(
    formalisation: str,
    debug_dir: Path,
    run_id: str,
    timeout: int = PROVER9_TIMEOUT,
) -> Dict[str, Any]:
    conclusion = extract_conclusion(formalisation)
    premises, injected, warnings = extract_premises(formalisation)

    if injected:
        return {
            "success": False,
            "prediction": "Unknown",
            "premises": premises,
            "conclusion": conclusion,
            "warnings": warnings + ["injected_conclusion_removed"],
            "injected": True,
            "time_s": 0.0,
        }

    if not premises or not conclusion:
        return {
            "success": False,
            "prediction": "Unknown",
            "premises": premises,
            "conclusion": conclusion,
            "warnings": warnings + ["missing_premises_or_conclusion"],
            "injected": injected,
            "time_s": 0.0,
        }

    run = prover9_decision(premises, conclusion, debug_dir, run_id, timeout=timeout)
    run.update({
        "premises": premises,
        "conclusion": conclusion,
        "warnings": warnings,
        "injected": injected,
    })
    return run


def _safe_filename(text: Any) -> str:
    s = str(text)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180] or "example"
