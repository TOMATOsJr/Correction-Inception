"""Consistency with the student's own work: does the candidate (or the written value) appear elsewhere on the page?

The proposal's second kind of evidence, never used in the checker pipeline: a misread "59" for a written "54" is
likely when 54 appears again on the page (the student reuses their values: "R = 54", later "I = V / 54"), and unlikely
when 59 does. This is consistency with what the student wrote, never with what is mathematically correct.

Per verified number (verify*.csv of a correction run), on the transcript page:
  best_elsewhere / written_elsewhere   other numbers on the page with the same value as the challenger / the written
  best_later / written_later            the same, only after the number (values carried forward)

Decision rules, each with the change threshold chosen per held-out student fold (<= --max-false-change of the
correct numbers changed on the other students), so every count is on students the rule was not tuned on:
  simple        correct.py's rule (checker probability, both orders, blank-image + answer-key guards)
  required      MAIN RULE, fixed in advance: simple, and the challenger appears elsewhere on the page and the written
                value does not
  veto          simple, unless the written value appears elsewhere and the challenger does not
  consistency_only   the checker only has to agree in both orders (p > 0.5), the page consistency decides

Outputs (in --dir/consistency): consistency_metrics.json, and the required rule's transcripts Outputs/{none,corrected}.

  python consistency.py --dir results/checker/correct_checker_fe_k10_beams --target-root R --model-dir qwen3b_ft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct import CHANGE, KEEP, TAU_GRID, actions, summarise, write_variants  # noqa: E402
from evaluate_detection import _num  # noqa: E402
from labels import math_tokens, page_files, page_text  # noqa: E402
from selector import student_folds  # noqa: E402
from verify_choice import qid_of  # noqa: E402

RULES = ["simple", "required", "veto", "consistency_only"]


def page_numbers(answer: str) -> list[tuple[int, float]]:
    """(maths-token index, value) of every number on the page, same indices as labels.csv."""
    out = []
    for t in math_tokens(answer):
        if t["type"] == "num":
            v = _num(t["raw"])
            if v is not None:
                out.append((t["i"], v))
    return out


def consistency_features(df: pd.DataFrame, pages: dict[str, str]) -> pd.DataFrame:
    """best/written _elsewhere and _later for each verify row (0 when the page is missing)."""
    cache: dict[str, list] = {}
    feats = []
    for _, r in df.iterrows():
        nums = cache.setdefault(r["page"], page_numbers(pages.get(r["page"], "")))
        i, bv, xv = int(r["i"]), _num(str(r["best"]) if pd.notna(r["best"]) else ""), _num(str(r["text"]))
        others = [(j, v) for j, v in nums if j != i]
        feats.append({"best_elsewhere": sum(v == bv for _, v in others) if bv is not None else 0,
                      "written_elsewhere": sum(v == xv for _, v in others) if xv is not None else 0,
                      "best_later": sum(v == bv for j, v in others if j > i) if bv is not None else 0,
                      "written_later": sum(v == xv for j, v in others if j > i) if xv is not None else 0})
    return pd.DataFrame(feats, index=df.index)


def allowed_mask(rule: str, f: pd.DataFrame) -> np.ndarray:
    be, we = f["best_elsewhere"].values > 0, f["written_elsewhere"].values > 0
    if rule in ("required", "consistency_only"):
        return be & ~we
    if rule == "veto":
        return ~(we & ~be)
    return np.ones(len(f), dtype=bool)


def pick_tau_allowed(df: pd.DataFrame, allowed: np.ndarray, tau_low: float, margin: float, max_fc: float) -> float:
    """correct.pick_tau with the rule's filter: the lowest threshold whose changes (only where the rule allows one)
    hit at most max_fc of ALL the correct numbers checked on these students."""
    ok = df["misread"].values == 0
    if not ok.any():
        return 1.01
    for t in TAU_GRID:
        ch = (actions(df, t, tau_low, margin) == CHANGE) & allowed
        if ch[ok].mean() <= max_fc:
            return t
    return 1.01


def decide(df: pd.DataFrame, f: pd.DataFrame, folds: np.ndarray, rule: str, max_fc: float,
           tau_low: float = 0.35, margin: float = 0.15) -> tuple[np.ndarray, list]:
    allowed = allowed_mask(rule, f)
    act = np.full(len(df), KEEP, dtype=object)
    taus = []
    for g in np.unique(folds):
        te = folds == g
        if rule == "consistency_only":
            t = 0.5
        else:
            t = pick_tau_allowed(df[~te], allowed[~te], tau_low, margin, max_fc)
        taus.append(t)
        act[te] = actions(df[te], t, tau_low, margin)
    act[(act == CHANGE) & ~allowed] = KEEP
    act[act != CHANGE] = KEEP  # no TA in this system
    return act, sorted(set(taus))


def load_pages(df: pd.DataFrame, target_root: Path, model_dir: str) -> dict[str, str]:
    pages = {}
    for pid in df["page"].unique():
        hw, st, stem = pid.split("|")
        src = page_files(target_root, hw, st, qid_of(stem), model_dir).get(stem)
        if src is not None:
            pages[pid] = page_text(src)
    return pages


def diagnostics(df: pd.DataFrame, f: pd.DataFrame) -> dict:
    """How often each consistency pattern holds for changes that would fix vs break (all verified rows)."""
    from correct import fixed_mask
    has = df["best"].fillna("").astype(str).str.len().values > 0
    fix = fixed_mask(df) & has
    bad = has & ~fix
    out = {}
    for name, m in (("challenger elsewhere", f["best_elsewhere"].values > 0),
                    ("written elsewhere", f["written_elsewhere"].values > 0),
                    ("challenger elsewhere and written not", allowed_mask("required", f))):
        out[name] = {"share_of_fixing_changes": round(float(m[fix].mean()), 4) if fix.any() else None,
                     "share_of_bad_changes": round(float(m[bad].mean()), 4) if bad.any() else None}
    return {"fixing_changes": int(fix.sum()), "bad_changes": int(bad.sum()), "patterns": out}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", default="qwen3b_ft")
    p.add_argument("--max-false-change", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    d = Path(a.dir)
    df = pd.concat([pd.read_csv(x, dtype={"gold": str, "best": str, "text": str}) for x in sorted(d.glob("verify*.csv"))],
                   ignore_index=True)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str, "gold_raw": str})
    pages = load_pages(df, Path(a.target_root), a.model_dir)
    f = consistency_features(df, pages)
    folds = student_folds(df["student"], min(a.folds, df["student"].nunique()))
    nums = labels[(labels["type"] == "num") & (labels["op"] != "ins") & labels["page"].isin(df["page"])]
    before, n_pages = int(nums["misread"].sum()), df["page"].nunique()
    out = d / "consistency"
    out.mkdir(exist_ok=True)
    res = {"pages_found": len(pages), "pages": n_pages, "main_rule": "required", "diagnostics": diagnostics(df, f)}
    acts = {}
    for rule in RULES:
        act, taus = decide(df, f, folds, rule, a.max_false_change)
        acts[rule] = act
        s = summarise(df, act, before, n_pages)
        s["net_fixes"] = s["auto_fixed"] - s["auto_wrong_fix"] - s["auto_broken_correct"]
        res[rule] = {"tau_high_per_fold": taus, **{k: v for k, v in s.items() if not k.startswith("ta_")
                                                   and "perfect_ta" not in k}}
    (out / "consistency_metrics.json").write_text(json.dumps(res, indent=2))
    df[["page", "student", "i"]].join(f).assign(fold=folds, **{f"action_{r}": acts[r] for r in RULES}).to_csv(
        out / "decisions.csv", index=False)
    write_variants(out, df, acts["required"], labels, Path(a.target_root), a.model_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
