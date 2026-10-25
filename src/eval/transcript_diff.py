"""Judge-free recognition metric: compare a model transcript with the expert-rectified transcript.

Replaces the LLM judge for new transcribers (Qwen2.5-7B was not a usable judge: kappa 0.03 vs the paper's Gemini judge).
Only maths ($...$ etc.) is compared, as tokens: numbers, operators/relations, variable names (prose and diagram
descriptions are ignored: they differ in wording between transcribers).

Per sample:
  token_error_rate  = edit distance between the two token sequences / gold length
  number_mismatches = numbers in gold missing from the target + numbers in the target not in gold (multiset)
  has_number_error  = number_mismatches > 0   (sample-level, the analogue of the paper's SER)

  python transcript_diff.py --target-root <folder with Homework_collected_database_trial_*> --target-model-dir gemini-2.5-pro --name gemini25
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_recognition import DEFAULT_EDU_REPO, PROJECT_ROOT, RECTIFIED, load_env, load_markdown  # noqa: E402

_LATEX = [(r"\\left|\\right|\\,|\\;|\\!|\\quad|\\qquad|\\text\{[^}]*\}|\\mathrm|\\displaystyle", " "),
          (r"\\cdot|\\times", "*"), (r"\\div", "/"), (r"\\frac", " frac "), (r"\\Omega|Ω", " ohm "),
          (r"(?<=\d),(?=\d{3}\b)", "")]  # 1,000 -> 1000
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?|[=+\-*/^<>]")
_NUM = re.compile(r"\d+(?:\.\d+)?")


_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)


def math_only(text: str) -> str:
    """Keep only maths ($..$, $$..$$, \\[..\\], \\(..\\)): drops chat preambles and diagram descriptions,
    which differ in wording between transcribers without changing what the student wrote."""
    return "\n".join(next(g for g in m.groups() if g is not None) for m in _MATH.finditer(text or ""))


def tokens(text: str) -> list[str]:
    t = math_only(text)
    for pat, rep in _LATEX:
        t = re.sub(pat, rep, t)
    return [re.sub(r"[{}]", "", x) for x in _TOKEN.findall(t)]


def numbers(toks: list[str]) -> Counter:
    return Counter(float(x) for x in toks if _NUM.fullmatch(x))


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compare(gold: str, target: str) -> dict:
    g, t = tokens(gold), tokens(target)
    ng, nt = numbers(g), numbers(t)
    miss = sum((ng - nt).values()) + sum((nt - ng).values())
    return {"token_error_rate": edit_distance(g, t) / max(1, len(g)), "number_mismatches": miss,
            "has_number_error": miss > 0}


def gold_items(data: Path) -> pd.DataFrame:
    items = pd.read_csv(data / "Screenshot_output_anon" / "set_splitting" / "obsetf_involved_data.csv", dtype=str)
    has = [load_markdown(data / RECTIFIED, r["Homework ID"], r["Student ID"], r["Question ID"], "gemini-2.5-pro")
           is not None for _, r in items.iterrows()]
    return items[has]


def bootstrap(df: pd.DataFrame, col: str, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[col].astype(float).values for _, g in df.groupby("Student ID")]
    vals = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main():
    load_env(DEFAULT_EDU_REPO)
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("EDU_DATA"))
    p.add_argument("--target-root", required=True)
    p.add_argument("--target-model-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "transcript_diff"))
    a = p.parse_args()
    data = Path(a.data_root)
    rows, missing = [], 0
    for _, r in gold_items(data).iterrows():
        hw, st, qid = r["Homework ID"], r["Student ID"], r["Question ID"]
        gold = load_markdown(data / RECTIFIED, hw, st, qid, "gemini-2.5-pro")
        tgt = load_markdown(Path(a.target_root), hw, st, qid, a.target_model_dir)
        if tgt is None:
            missing += 1
            continue
        rows.append({"Homework ID": hw, "Student ID": st, "Question ID": qid, **compare(gold, tgt)})
    df = pd.DataFrame(rows)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_sample.csv", index=False)
    m = {"name": a.name, "n_samples": len(df), "missing_target": missing,
         "number_error_sample_rate_percent": round(100 * df["has_number_error"].mean(), 2),
         "number_error_95CI": [round(100 * x, 2) for x in bootstrap(df, "has_number_error")],
         "mean_number_mismatches": round(df["number_mismatches"].mean(), 3),
         "mean_token_error_rate": round(df["token_error_rate"].mean(), 4),
         "token_error_rate_95CI": bootstrap(df, "token_error_rate")}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
