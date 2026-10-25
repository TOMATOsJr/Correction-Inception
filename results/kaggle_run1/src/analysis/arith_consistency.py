"""No-API evidence for 'correction inception' in real university data.

For each sample with an expert-verified transcript (gold) and the paper's Gemini-2.5-Pro transcript (target):
  1. extract purely numeric equalities written by the student, e.g.  "12/6 = 2", "2 + 3 = 6", "\\frac{10}{4}=2.5";
  2. evaluate each side (no symbols, no units) -> the equality is arithmetically CONSISTENT or not;
  3. align gold and target equalities (difflib on normalized strings) and classify aligned pairs:
       silent correction : gold INconsistent (student slip) but target consistent with different numbers
       misread-induced   : gold consistent but target inconsistent
       student slip kept : both inconsistent (error transcribed faithfully)
Also reports a simple detector: "flag every target equality that is arithmetically inconsistent" and how often
that flag lands on a real transcription difference (precision) / a student slip.

This uses arithmetic correctness only to FIND student slips in the gold transcript (a measurement tool),
not as a correction rule; our system must never 'fix' toward correctness.
"""
from __future__ import annotations

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_recognition import DEFAULT_EDU_REPO, RECTIFIED, load_env, load_markdown  # noqa: E402

_SAFE = re.compile(r"^[0-9.+\-*/() ]+$")


_MULT = {"k": "*1e3", "M": "*1e6", "m": "*1e-3", "μ": "*1e-6", "u": "*1e-6", "n": "*1e-9", "p": "*1e-12"}
_UNIT = re.compile(r"(\d(?:\.\d+)?)\s*([kMmμunp])?\s*(?:Ω|V|A|W|F|H|J|Hz|s|ohms?|rad/s)(?![A-Za-z])")


def latex_to_expr(s: str) -> str | None:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r" \2 ", s)
    s = s.replace("\\Omega", "Ω").replace("\\mu", "μ").replace("\\%", "")
    s = _UNIT.sub(lambda m: m.group(1) + _MULT.get(m.group(2) or "", ""), s)
    s = re.sub(r"\\(left|right|,|;|!|quad|qquad)", " ", s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\(times|cdot|ast)", "*", s)
    s = re.sub(r"\\div", "/", s)
    s = s.replace("{", "(").replace("}", ")").replace("^", "**").replace("−", "-").replace("×", "*")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)  # thousands separators
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)      # 2(3) -> 2*(3)
    s = re.sub(r"\)\s*(\d)", r")*\1", s)
    s = s.strip()
    if not s or not re.search(r"\d", s) or not _SAFE.match(s.replace("**", "*")):
        return None
    if not re.search(r"[+\-*/]", s.lstrip("-")):  # plain number
        return s
    return s


def evaluate(expr: str):
    try:
        v = eval(expr, {"__builtins__": {}}, {})  # expr is whitelisted to digits/operators/parentheses
        return float(v)
    except Exception:
        return None


def equalities(md: str) -> list[dict]:
    """Numeric equalities 'A = B' where A has an operator and both sides evaluate."""
    out = []
    for line in md.splitlines():
        for seg in re.findall(r"\$\$?(.+?)\$\$?", line) or [line]:
            parts = [p for p in re.split(r"(?<![<>!])=(?!=)", seg)]
            for a, b in zip(parts, parts[1:]):
                ea, eb = latex_to_expr(a), latex_to_expr(b)
                if not ea or not eb or not re.search(r"[+\-*/]", ea.lstrip("-")):
                    continue
                va, vb = evaluate(ea), evaluate(eb)
                if va is None or vb is None:
                    continue
                tol = max(0.02 * abs(vb), 1e-6)  # students round; 2% tolerance
                out.append({"text": f"{ea} = {eb}", "lhs": ea, "rhs": eb, "consistent": abs(va - vb) <= tol})
    return out


def main():
    load_env(DEFAULT_EDU_REPO)
    data = Path(os.environ.get("EDU_DATA", "D:/lma/data/EDU-CIRCUIT-HW_v1"))
    items = pd.read_csv(data / "Screenshot_output_anon" / "set_splitting" / "obsetf_involved_data.csv", dtype=str)
    tgt_root = data / "Observationset_Final" / "v6_Gemini_2p5"
    stats = {k: 0 for k in ["samples", "gold_eq", "gold_slips", "target_eq", "target_inconsistent",
                            "aligned", "silent_correction", "misread_induced", "slip_kept", "flag_on_diff"]}
    examples = {"silent_correction": [], "misread_induced": []}
    for _, r in items.iterrows():
        hw, st, qid = r["Homework ID"], r["Student ID"], r["Question ID"]
        gold = load_markdown(data / RECTIFIED, hw, st, qid, "gemini-2.5-pro")
        tgt = load_markdown(tgt_root, hw, st, qid, "gemini-2.5-pro")
        if gold is None or tgt is None:
            continue
        stats["samples"] += 1
        ge, te = equalities(gold), equalities(tgt)
        stats["gold_eq"] += len(ge)
        stats["gold_slips"] += sum(not e["consistent"] for e in ge)
        stats["target_eq"] += len(te)
        stats["target_inconsistent"] += sum(not e["consistent"] for e in te)
        sm = SequenceMatcher(None, [e["lhs"] for e in ge], [e["lhs"] for e in te], autojunk=False)
        pairs = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
                pairs += list(zip(ge[i1:i2], te[j1:j2]))
        for g, t in pairs:
            # only pairs that plausibly refer to the same written equality
            if SequenceMatcher(None, g["text"], t["text"]).ratio() < 0.5:
                continue
            stats["aligned"] += 1
            differs = g["text"].replace(" ", "") != t["text"].replace(" ", "")
            if not t["consistent"] and differs:
                stats["flag_on_diff"] += 1
            ex = {"id": f"{hw}/{st}/{qid}", "gold": g["text"], "gemini": t["text"]}
            if not g["consistent"] and t["consistent"] and differs:
                stats["silent_correction"] += 1
                examples["silent_correction"].append(ex)
            elif g["consistent"] and not t["consistent"]:
                stats["misread_induced"] += 1
                examples["misread_induced"].append(ex)
            elif not g["consistent"] and not t["consistent"]:
                stats["slip_kept"] += 1
    res = {"stats": stats,
           "silent_correction_rate_among_aligned_student_slips_percent":
               round(100 * stats["silent_correction"] / max(1, stats["silent_correction"] + stats["slip_kept"]), 1),
           "detector_precision_inconsistent_flag_on_real_difference_percent":
               round(100 * stats["flag_on_diff"] / max(1, stats["target_inconsistent"]), 1),
           "examples": {k: v[:15] for k, v in examples.items()}}
    out = ROOT / "results" / "baseline_reference" / "arith_consistency_gemini25.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    for k, v in examples.items():
        print(f"\n{k}: {len(v)} examples (first 6)")
        for e in v[:6]:
            print(f"  {e['id']}: gold [{e['gold']}]  gemini [{e['gemini']}]")


if __name__ == "__main__":
    main()
