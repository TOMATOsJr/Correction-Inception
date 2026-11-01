"""Decide and apply corrections for verified numbers: change / keep / TA, then write corrected transcripts (Phase 4)

Input: verify*.csv (verify_choice.py: best challenger c for each flagged number x, its forced-choice probability
p_change, order consistency, blank-image probability, answer-key flags). Per number:

  change   consistent and p_change >= tau_high, and no guard fired
  keep     p_change <= tau_low (the reader prefers x)
  TA       everything else: middling evidence, order flips, or a guard fired
  guards   answer key: c is in the rubric's answer key and x is not (would move towards the expected answer)
           blank image: p_blank >= p_change - blank_margin (the reader prefers c about as much without the page)

Default to keep: a correct number that is changed is a new error. tau_high is chosen per held-out student fold, on
the OTHER students, as the lowest threshold whose false-change rate (correct flagged numbers changed) is at most
--max-false-change. Every reported number is therefore on students the threshold was not tuned on.

Outputs (in --dir):
  decisions.csv           one row per verified number: action, fold, tau_high used
  correct_metrics.json    fixes, wrong fixes, broken numbers, TA load, and misread counts before / after
  Outputs/<variant>/...   transcripts in the usual Compare layout, for transcript_diff.py and grade_impact.py:
      none          the transcripts as they were (same pages, for a fair comparison)
      corrected     automatic changes only
      corrected_ta  automatic changes + TA queue resolved by the expert transcript (a perfect-TA upper bound)

  python correct.py --dir results/detect/qwen3b_ft_ftreader --target-root R --model-dir qwen3b_ft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detection import _num  # noqa: E402
from labels import page_files  # noqa: E402
from selector import student_folds  # noqa: E402
from verify_choice import qid_of  # noqa: E402

TAU_GRID = [round(t, 2) for t in np.arange(0.5, 1.0, 0.025)]
CHANGE, KEEP, TA = "change", "keep", "ta"


def actions(df: pd.DataFrame, tau_high: float, tau_low: float, blank_margin: float) -> np.ndarray:
    p = df["p_change"].astype(float).values
    cons = df["consistent"].astype(bool).values
    has = df["best"].fillna("").astype(str).str.len().values > 0
    key = df["cand_is_key"].astype(bool).values & ~df["orig_is_key"].astype(bool).values
    pb = df["p_blank"].astype(float).values
    blank = ~np.isnan(pb) & (pb >= p - blank_margin)
    want = has & cons & (p >= tau_high)
    out = np.full(len(df), TA, dtype=object)
    out[want & ~key & ~blank] = CHANGE
    out[~has | (p <= tau_low)] = KEEP
    return out


def fixed_mask(df: pd.DataFrame) -> np.ndarray:
    """The best challenger equals the expert's value (only meaningful for misreads)."""
    return np.array([m == 1 and _num(b) is not None and _num(b) == float(g)
                     for m, b, g in zip(df["misread"], df["best"].fillna("").astype(str), df["gold"])])


def pick_tau(df: pd.DataFrame, tau_low: float, blank_margin: float, max_false_change: float) -> float:
    ok = df["misread"].values == 0
    if not ok.any():  # nothing to tune on (e.g. a one-student smoke run): never change automatically
        return 1.01
    for t in TAU_GRID:
        if (actions(df, t, tau_low, blank_margin)[ok] == CHANGE).mean() <= max_false_change:
            return t
    return 1.01


def summarise(df: pd.DataFrame, act: np.ndarray, all_misreads: int, pages: int) -> dict:
    mis = df["misread"].values == 1
    fixed = fixed_mask(df)
    ch = act == CHANGE
    n_fix, n_wrong, n_broken = int((ch & fixed).sum()), int((ch & mis & ~fixed).sum()), int((ch & ~mis).sum())
    ta_mis, ta_ok = int(((act == TA) & mis).sum()), int(((act == TA) & ~mis).sum())
    return {
        "verified": len(df), "verified_misreads": int(mis.sum()),
        "auto_fixed": n_fix, "auto_wrong_fix": n_wrong, "auto_broken_correct": n_broken,
        "ta_queue": ta_mis + ta_ok, "ta_queue_misreads": ta_mis, "ta_per_page": round((ta_mis + ta_ok) / max(1, pages), 2),
        "kept_misreads": int(((act == KEEP) & mis).sum()),
        "false_change_rate": round(n_broken / max(1, int((~mis).sum())), 4),
        "misreads_before": all_misreads,
        "misreads_after_auto": all_misreads - n_fix + n_broken,
        "misreads_after_auto_and_perfect_ta": all_misreads - n_fix - ta_mis + n_broken,
    }


def fmt_gold(r) -> str:
    raw = r.get("gold_raw")
    if isinstance(raw, str) and raw:
        return raw
    v = float(r["gold"])
    return str(int(v)) if v.is_integer() else repr(v)


def write_variants(d: Path, df: pd.DataFrame, act: np.ndarray, labels: pd.DataFrame, target_root: Path, model_dir: str):
    gold_raw = labels.set_index(["page", "i"])["gold_raw"] if "gold_raw" in labels else None
    edits = {"none": {}, "corrected": {}, "corrected_ta": {}}
    for (_, r), a in zip(df.iterrows(), act):
        s, e = int(r["start"]), int(r["end"])
        if a == CHANGE:
            for v in ("corrected", "corrected_ta"):
                edits[v].setdefault(r["page"], []).append((s, e, r["text"], str(r["best"])))
        elif a == TA and r["misread"] == 1:
            g = gold_raw.get((r["page"], r["i"]), "") if gold_raw is not None else ""
            edits["corrected_ta"].setdefault(r["page"], []).append(
                (s, e, r["text"], g if isinstance(g, str) and g else fmt_gold(r)))
    skipped = 0
    for pid in labels["page"].unique():
        hw, st, stem = pid.split("|")
        src = page_files(target_root, hw, st, qid_of(stem), model_dir).get(stem)
        if src is None:
            skipped += 1
            continue
        head, _, body = src.read_text(encoding="utf-8").partition("\n")
        for v, ed in edits.items():
            text = body
            for s, e, old, new in sorted(ed.get(pid, []), reverse=True):
                if text[s:e] == old:  # offsets index page_text (= body); never patch a moved span
                    text = text[:s] + new + text[e:]
            dst = d / "Outputs" / v / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
            dst.mkdir(parents=True, exist_ok=True)
            (dst / src.name).write_text(head + "\n" + text, encoding="utf-8")
    return skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv and verify*.csv")
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--max-false-change", type=float, default=0.01)
    p.add_argument("--tau-low", type=float, default=0.35)
    p.add_argument("--blank-margin", type=float, default=0.15)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    d = Path(a.dir)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str, "gold_raw": str})
    df = pd.concat([pd.read_csv(f, dtype={"gold": str, "best": str, "text": str}) for f in sorted(d.glob("verify*.csv"))],
                   ignore_index=True)
    folds = student_folds(df["student"], min(a.folds, df["student"].nunique()))
    act = np.empty(len(df), dtype=object)
    taus = np.zeros(len(df))
    for f in np.unique(folds):
        te = folds == f
        t = pick_tau(df[~te], a.tau_low, a.blank_margin, a.max_false_change)
        act[te], taus[te] = actions(df[te], t, a.tau_low, a.blank_margin), t
    # before / after counts over the pages that were verified (a partial or sharded run must not mix scopes)
    nums = labels[(labels["type"] == "num") & (labels["op"] != "ins") & labels["page"].isin(df["page"])]
    pages = df["page"].nunique()
    res = {"settings": {"max_false_change": a.max_false_change, "tau_low": a.tau_low, "blank_margin": a.blank_margin,
                        "tau_high_per_fold": sorted({float(t) for t in taus})},
           "heldout": summarise(df, act, int(nums["misread"].sum()), pages)}
    out = df[["page", "student", "i"]].assign(fold=folds, tau_high=taus, action=act)
    out.to_csv(d / "decisions.csv", index=False)
    skipped = write_variants(d, df, act, labels, Path(a.target_root), a.model_dir)
    res["pages_without_source"] = skipped
    (d / "correct_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
