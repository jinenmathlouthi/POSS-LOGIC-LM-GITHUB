from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


PROLOG_TIMEOUT = 10


def normalize_symbol(text: Any) -> str:
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("'", "")
    s = s.replace('"', "")
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    if s[0].isdigit():
        s = "n_" + s
    return s


def normalize_arg(text: Any) -> str:
    raw = str(text).strip()
    if raw in {"X", "Y", "Z"}:
        return raw
    if raw.lower() in {"x", "y", "z"}:
        return raw.upper()
    return normalize_symbol(raw)


def make_atom(pred: Any, args: Sequence[Any], negated: bool = False) -> str:
    clean_pred = normalize_symbol(pred)
    clean_args = [normalize_arg(a) for a in args if str(a).strip()]
    if not clean_args:
        clean_args = ["unknown"]
    atom = f"{clean_pred}({','.join(clean_args)})"
    return f"neg_{atom}" if negated else atom


def safe_filename(text: Any) -> str:
    s = str(text)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180] or "example"


def find_swipl() -> str:
    path = shutil.which("swipl")
    if not path:
        raise FileNotFoundError(
            "SWI-Prolog executable 'swipl' was not found. "
            "Install it first, e.g. on Kaggle: !apt-get install -y swi-prolog"
        )
    return path


# ---------------------------------------------------------------------------
# Dataset query builders
# ---------------------------------------------------------------------------

def parse_prontoqa_conclusion(conclusion: str) -> Optional[str]:
    text = str(conclusion).strip()
    text = re.sub(r"^is the following statement true or false\?\s*", "", text, flags=re.I)
    text = text.replace(".", "").replace("?", "").strip()

    m = re.match(
        r"^([A-Za-z][A-Za-z0-9_-]*)\s+is\s+not\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z0-9_-]*)$",
        text,
        flags=re.I,
    )
    if m:
        return make_atom(m.group(2), [m.group(1)], negated=True)

    m = re.match(
        r"^([A-Za-z][A-Za-z0-9_-]*)\s+is\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z0-9_-]*)$",
        text,
        flags=re.I,
    )
    if m:
        return make_atom(m.group(2), [m.group(1)], negated=False)

    return None


def proofwriter_variants(conclusion: str) -> Set[str]:
    s = str(conclusion).lower().strip().rstrip(".")
    s = s.replace("does not", "not")
    s = s.replace("do not", "not")
    s = re.sub(r"\s+", " ", s)

    variants: Set[str] = set()

    m = re.match(r"^(?:the\s+)?(.+?)\s+is\s+not\s+(.+)$", s)
    if m:
        subj = normalize_symbol(m.group(1).replace("the ", ""))
        pred = normalize_symbol(m.group(2))
        variants.add(f"neg_{pred}({subj})")
        return variants

    m = re.match(r"^(?:the\s+)?(.+?)\s+is\s+(.+)$", s)
    if m:
        subj = normalize_symbol(m.group(1).replace("the ", ""))
        pred = normalize_symbol(m.group(2))
        variants.add(f"{pred}({subj})")
        return variants

    m = re.match(r"^(?:the\s+)?(.+?)\s+(not\s+)?(\w+)\s+(?:the\s+)?(.+)$", s)
    if m:
        subj = normalize_symbol(m.group(1).replace("the ", ""))
        neg = m.group(2)
        verb_raw = normalize_symbol(m.group(3))
        obj = normalize_symbol(m.group(4).replace("the ", ""))
        verb_base = verb_raw.rstrip("s")
        verb_forms = {verb_raw, verb_base, verb_base + "s"}
        for verb in verb_forms:
            if neg:
                variants.add(f"neg_{verb}({subj},{obj})")
                variants.add(f"neg_{verb}_{obj}({subj})")
            else:
                variants.add(f"{verb}({subj},{obj})")
                variants.add(f"{verb}_{obj}({subj})")

    return variants


def parse_proofwriter_conclusion(conclusion: str, program: str = "") -> str:
    variants = proofwriter_variants(conclusion)
    compact_program = program.replace(" ", "")

    for variant in variants:
        if variant.replace(" ", "") in compact_program:
            return variant

    for variant in variants:
        if "(" in variant and "," in variant:
            pred = variant.split("(")[0]
            args = variant.split("(", 1)[1].rstrip(")").split(",")
            if len(args) == 2:
                subj, obj = args
                alt = f"{pred}_{obj}({subj})"
                if alt in compact_program:
                    return alt

    return sorted(variants)[0] if variants else "unknown(unknown)"


# ---------------------------------------------------------------------------
# Formalization conversion
# ---------------------------------------------------------------------------

def remove_tool_artifacts(text: str) -> str:
    return str(text).replace("<tool_call>", "").replace("</tool_call>", "")


def extract_premises_block(formalisation: str) -> str:
    text = remove_tool_artifacts(formalisation)
    upper = text.upper()

    start_positions = []
    for marker in ("PREMISES:", "FACTS:", "RULES:", "GROUND FACTS:"):
        pos = upper.find(marker)
        if pos >= 0:
            start_positions.append((pos, marker))

    if start_positions:
        pos, marker = min(start_positions, key=lambda x: x[0])
        text = text[pos + len(marker):]

    upper = text.upper()
    end = upper.find("CONCLUSION:")
    if end >= 0:
        text = text[:end]

    return text.strip()


def clean_fol_line(line: str) -> str:
    line = str(line).strip()
    if not line:
        return ""

    line = line.replace("∀", "forall")
    line = line.replace("¬", "NOT")
    line = line.replace("→", "->")
    line = line.replace("⇒", "->")
    line = line.replace("↔", "<->")
    line = line.replace("`", "")
    line = line.replace(" V ", " | ")
    line = line.replace(" v ", " | ")
    line = line.replace(" AND ", " & ")
    line = line.replace(" OR ", " | ")
    line = re.sub(r"\s+", " ", line).strip()

    bad_prefixes = (
        "PREDICATES:", "PREMISES:", "CONCLUSION:", "GROUND FACTS:",
        "GROUND_FACTS:", "FACTS:", "RULES:", "STRICT RULES:",
    )
    if any(line.upper().startswith(prefix) for prefix in bad_prefixes):
        return ""
    if ":::" in line:
        return ""

    return line


def strip_forall_wrapper(line: str) -> str:
    s = line.strip()
    while True:
        m = re.match(r"^forall\s+[A-Za-z_][A-Za-z0-9_]*\s*\((.*)\)\s*\.?$", s, flags=re.I)
        if m:
            s = m.group(1).strip()
            continue
        m = re.match(r"^forall\s+[A-Za-z_][A-Za-z0-9_]*\s+(.*)$", s, flags=re.I)
        if m:
            s = m.group(1).strip()
            continue
        return s


def parse_atom_expr(expr: str, prefer: str = "first") -> Optional[str]:
    expr = str(expr).strip().rstrip(".")
    expr = expr.replace("()", "")

    # Handle not(pred(x)) generated by some models.
    m_not_fun = re.findall(r"\bnot\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*\)", expr, flags=re.I)
    if m_not_fun:
        pred, args_text = m_not_fun[-1] if prefer == "last" else m_not_fun[0]
        return make_atom(pred, [a.strip() for a in args_text.split(",")], negated=True)

    neg_matches = re.findall(
        r"\b(?:NOT|not)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)",
        expr,
        flags=re.I,
    )
    if neg_matches:
        pred, args_text = neg_matches[-1] if prefer == "last" else neg_matches[0]
        return make_atom(pred, [a.strip() for a in args_text.split(",")], negated=True)

    matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)", expr)
    if matches:
        pred, args_text = matches[-1] if prefer == "last" else matches[0]
        return make_atom(pred, [a.strip() for a in args_text.split(",")], negated=False)

    slash_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*/\s*([A-Za-z_][A-Za-z0-9_]*)\b", expr)
    if slash_matches:
        pred, const = slash_matches[-1] if prefer == "last" else slash_matches[0]
        return make_atom(pred, [const], negated=False)

    return None


def parse_rule_line(line: str) -> List[str]:
    line = clean_fol_line(line)
    if not line:
        return []

    line = strip_forall_wrapper(line)
    rules: List[str] = []

    if "<->" in line:
        left, right = line.split("<->", 1)
        left_atom = parse_atom_expr(left, prefer="last")
        right_atom = parse_atom_expr(right, prefer="first")
        if left_atom and right_atom:
            rules.append(f"{right_atom} :- {left_atom}.")
            rules.append(f"{left_atom} :- {right_atom}.")
        return rules

    if "->" in line:
        parts = [p.strip() for p in line.split("->")]
        head = parse_atom_expr(parts[-1], prefer="first")
        body_atoms = [parse_atom_expr(part, prefer="last") for part in parts[:-1]]
        body_atoms = [a for a in body_atoms if a]
        if head and body_atoms:
            rules.append(f"{head} :- {', '.join(body_atoms)}.")
        return rules

    return []


def parse_fact_line(line: str, forbidden_query: Optional[str] = None) -> List[str]:
    line = clean_fol_line(line)
    if not line:
        return []
    if "forall" in line.lower() or "->" in line or "<->" in line:
        return []

    atom = parse_atom_expr(line, prefer="first")
    if not atom:
        return []
    if forbidden_query and atom.replace(" ", "") == forbidden_query.replace(" ", ""):
        return []
    return [f"{atom}."]


def formalisation_to_prolog(
    formalisation: str,
    dataset: str,
    dataset_query: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    warnings: List[str] = []
    statements: List[str] = []
    injected = False
    forbidden = dataset_query.replace(" ", "") if dataset_query else None

    for raw_line in extract_premises_block(formalisation).splitlines():
        line = clean_fol_line(raw_line)
        if not line:
            continue

        rule_statements = parse_rule_line(line)
        if rule_statements:
            statements.extend(rule_statements)
            continue

        fact_statements = parse_fact_line(line, forbidden_query=dataset_query)
        if fact_statements:
            for fact in fact_statements:
                if forbidden and fact.rstrip(".").replace(" ", "") == forbidden:
                    injected = True
                    continue
                statements.append(fact)
            continue

        warnings.append(f"ignored_line: {line[:120]}")

    seen = set()
    unique: List[str] = []
    for statement in statements:
        if statement not in seen:
            unique.append(statement)
            seen.add(statement)

    if not unique:
        warnings.append("empty_program")

    return "\n".join(unique), warnings, injected


def extract_predicate_arities(program: str, query: Optional[str]) -> List[Tuple[str, int]]:
    text = program + "\n" + (query or "")
    pairs = set()
    for pred, args_text in re.findall(r"\b([a-z][a-z0-9_]*)\s*\(([^()]*)\)", text):
        arity = 0 if not args_text.strip() else len([a for a in args_text.split(",") if a.strip()])
        if arity > 0:
            pairs.add((pred, arity))
    return sorted(pairs)


def build_kb_file_content(program: str, query: Optional[str] = None) -> str:
    declarations = []
    for pred, arity in extract_predicate_arities(program, query):
        declarations.append(f":- dynamic {pred}/{arity}.")
        declarations.append(f":- discontiguous {pred}/{arity}.")
    return "\n".join(declarations + ["", program]).strip() + "\n"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_binary_query(
    program: str,
    query: Optional[str],
    debug_dir: Path,
    run_id: str,
    timeout: int = PROLOG_TIMEOUT,
) -> Dict[str, Any]:
    swipl = find_swipl()
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not query:
        return _prolog_failure("missing_query")
    if not program.strip():
        return _prolog_failure("empty_program")

    file_path = debug_dir / f"{safe_filename(run_id)}.pl"
    file_path.write_text(build_kb_file_content(program, query), encoding="utf-8")

    goal = f"(catch(once(({query})), _, fail) -> writeln('RESULT=true') ; writeln('RESULT=false')), halt(0)"
    t0 = time.time()

    try:
        proc = subprocess.run(
            [swipl, "-q", "-s", str(file_path), "-g", goal],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 3)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        q_true = "RESULT=true" in stdout
        q_false = "RESULT=false" in stdout
        success = proc.returncode == 0 and (q_true or q_false)
        return {
            "success": success,
            "q_true": q_true,
            "prediction": "True" if q_true else "False",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "prolog_file": str(file_path),
            "time_s": elapsed,
        }
    except subprocess.TimeoutExpired as e:
        return _prolog_failure("PYTHON_TIMEOUT", file_path=str(file_path), stdout=e.stdout or "")
    except Exception as exc:
        return _prolog_failure(repr(exc), file_path=str(file_path))


def run_ternary_query(
    program: str,
    query: Optional[str],
    debug_dir: Path,
    run_id: str,
    timeout: int = PROLOG_TIMEOUT,
) -> Dict[str, Any]:
    swipl = find_swipl()
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not query:
        return _prolog_failure("missing_query", prediction="Unknown")
    if not program.strip():
        return _prolog_failure("empty_program", prediction="Unknown")

    file_path = debug_dir / f"{safe_filename(run_id)}.pl"
    file_path.write_text(build_kb_file_content(program, query), encoding="utf-8")

    if query.startswith("neg_"):
        positive = query[4:]
        negative = query
    else:
        positive = query
        negative = f"neg_{query}"

    goal = (
        f"((catch(once(({positive})), _, fail), catch(once(({negative})), _, fail)) -> writeln('RESULT=conflict') ; "
        f"(catch(once(({positive})), _, fail) -> writeln('RESULT=true') ; "
        f"(catch(once(({negative})), _, fail) -> writeln('RESULT=false') ; writeln('RESULT=unknown')))), halt(0)"
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            [swipl, "-q", "-s", str(file_path), "-g", goal],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 3)
        stdout = proc.stdout.strip().lower()
        if "result=true" in stdout:
            prediction = "True"
        elif "result=false" in stdout:
            prediction = "False"
        else:
            prediction = "Unknown"
        return {
            "success": proc.returncode == 0,
            "prediction": prediction,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
            "prolog_file": str(file_path),
            "time_s": elapsed,
        }
    except subprocess.TimeoutExpired as e:
        return _prolog_failure("PYTHON_TIMEOUT", prediction="Unknown", file_path=str(file_path), stdout=e.stdout or "")
    except Exception as exc:
        return _prolog_failure(repr(exc), prediction="Unknown", file_path=str(file_path))


def _prolog_failure(
    stderr: str,
    prediction: str = "False",
    file_path: str = "",
    stdout: str = "",
) -> Dict[str, Any]:
    return {
        "success": False,
        "q_true": False,
        "prediction": prediction,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": -1,
        "prolog_file": file_path,
        "time_s": 0.0,
    }


def solve_formalisation_with_prolog(
    formalisation: str,
    conclusion: str,
    dataset: str,
    debug_dir: Path,
    run_id: str,
    timeout: int = PROLOG_TIMEOUT,
) -> Dict[str, Any]:
    dataset = dataset.lower().strip()

    if dataset == "prontoqa":
        query = parse_prontoqa_conclusion(conclusion)
        program, warnings, injected = formalisation_to_prolog(formalisation, dataset, dataset_query=query)
        run = run_binary_query(program, query, debug_dir, run_id, timeout=timeout)
        run.update({"query": query, "program": program, "warnings": warnings, "injected": injected})
        return run

    if dataset == "proofwriter":
        # First build a program without a query, then infer the best query variant from that program.
        program, warnings, injected = formalisation_to_prolog(formalisation, dataset, dataset_query=None)
        query = parse_proofwriter_conclusion(conclusion, program=program)
        # Rebuild while forbidding direct injection of the target query.
        program, warnings, injected = formalisation_to_prolog(formalisation, dataset, dataset_query=query)
        run = run_ternary_query(program, query, debug_dir, run_id, timeout=timeout)
        run.update({"query": query, "program": program, "warnings": warnings, "injected": injected})
        return run

    raise ValueError(f"Prolog solver does not support dataset: {dataset}")
