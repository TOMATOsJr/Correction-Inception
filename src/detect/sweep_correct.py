"""What if the decision rule were different? Re-decide verify*.csv under many settings, no GPU needed (Phase 4b)

correct.py applies one rule. This tries the variants we may want later, each with tau_high chosen on the other
students (same folds as correct.py), so every row is on students its threshold was not tuned on:

  max_false_change   how many correct numbers the corrector may change
  guards             all | no_key | no_blank | none   (which silent-correction guards are on)
  filter             all | digit_doubt   (digit_doubt: keep unless the reader's preferred token at the number is a
                     digit swap, so doubt about spaces or LaTeX never leads to a change)

net_fixes = auto_fixed - auto_broken_correct: a rule is only worth using if this is positive.

Output (in --dir): sweep_metrics.csv (counts only, no transcript text).

  python sweep_correct.py --dir results/detect/qwen3b_ft_ftreader
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct import CHANGE, KEEP, actions, pick_tau, summarise  # noqa: E402
from evaluate_detection import _num  # noqa: E402
from selector import student_folds  # noqa: E402

FALSE_CHANGE = [0.0025, 0.005, 0.01, 0.02, 0.05]
GUARDS = ["all", "no_key", "no_blank", "none"]
FILTERS = ["all", "digit_doubt"]
OFF = -9.0  # a blank margin no probability can reach


def with_guards(df: pd.DataFrame, guards: str, margin: float) -> tuple[pd.DataFrame, float]:
    if guards in ("no_key", "none"):
        df = df.assign(cand_is_key=False)
    return df, (OFF if guards in ("no_blank", "none") else margin)


def digit_doubt(df: pd.DataFrame) -> np.ndarray:
    """The reader's preferred reading of the number is itself a number."""
    return np.array([_num(a) is not None for a in df["alt"].fillna("").astype(str)])


def run_rule(df, folds, max_fc, guards, flt, tau_low, margin) -> np.ndarray:
    d, m = with_guards(df, guards, margin)
    act = np.empty(len(d), dtype=object)
    allowed = digit_doubt(d) if flt == "digit_doubt" else np.ones(len(d), dtype=bool)
    for f in np.unique(folds):
        te = folds == f
        tr = d[~te]
        t = pick_tau(tr[allowed[~te]], tau_low, m, max_fc)
        act[te] = actions(d[te], t, tau_low, m)
    act[~allowed & (act == CHANGE)] = KEEP
    return act


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv, scores*.csv and verify*.csv")
    p.add_argument("--tau-low", type=float, default=0.35)
    p.add_argument("--blank-margin", type=float, default=0.15)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    d = Path(a.dir)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str, "gold_raw": str})
    df = pd.concat([pd.read_csv(f, dtype={"gold": str, "best": str, "text": str}) for f in sorted(d.glob("verify*.csv"))],
                   ignore_index=True)
    scores = pd.concat([pd.read_csv(f, dtype={"alt": str}) for f in sorted(d.glob("scores*.csv"))])
    df = df.merge(scores[["page", "i", "alt"]], on=["page", "i"], how="left")
    folds = student_folds(df["student"], min(a.folds, df["student"].nunique()))
    nums = labels[(labels["type"] == "num") & (labels["op"] != "ins") & labels["page"].isin(df["page"])]
    rows = []
    for fc in FALSE_CHANGE:
        for g in GUARDS:
            for flt in FILTERS:
                act = run_rule(df, folds, fc, g, flt, a.tau_low, a.blank_margin)
                s = summarise(df, act, int(nums["misread"].sum()), df["page"].nunique())
                rows.append({"max_false_change": fc, "guards": g, "filter": flt, **s,
                             "net_fixes": s["auto_fixed"] - s["auto_broken_correct"]})
    out = pd.DataFrame(rows).sort_values("net_fixes", ascending=False)
    out.to_csv(d / "sweep_metrics.csv", index=False)
    cols = ["max_false_change", "guards", "filter", "auto_fixed", "auto_wrong_fix", "auto_broken_correct",
            "net_fixes", "ta_per_page"]
    print(out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
