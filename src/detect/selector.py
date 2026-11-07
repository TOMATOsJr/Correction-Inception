"""Learned misread selector: which numbers should the corrector look at? (Phase 2b)

One confidence cut-off treats every page alike, but 0.95 is suspicious on a clean page and normal on a messy one.
Instead, each number is described by ~28 clues computed from score_tokens.py's output (no new GPU work) and a
logistic regression learns from the expert labels how to weigh them:

  own doubt     nll, neg_gap, ent, dent, n_pieces
  page context  page mean of nll / dent, the number's rank of nll / dent within its page, page length, position
  neighbours    nll and dent of the previous and next maths token
  shape         number of digits, decimal point, which digits occur, reader's alternative differs in one digit

Evaluation never sees a student it was trained on: students are split into folds; for each fold the model is
trained on the other students, the cut-off for the target recall is chosen on those training students, and both
are applied to the held-out fold. Every number therefore gets an out-of-fold misread probability and flag.

Outputs (in --dir):
  selector.csv           page, i, student, fold, p_misread, flagged  (no transcript text)
  selector_metrics.json  recall / precision / share flagged on held-out students, vs the nll cut-off at equal recall

  python selector.py --dir results/detect/qwen3b_ft_ftreader --target-recall 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLS = ["nll", "neg_gap", "ent", "dent", "n_pieces"]
DIGITS = "0123456789"


def clues(scores: pd.DataFrame) -> pd.DataFrame:
    """Per maths token of score_tokens.py's output: the selector's features (all maths tokens are needed for the
    neighbour clues; filter to numbers afterwards)."""
    df = scores.sort_values(["page", "i"]).reset_index(drop=True)
    g = df.groupby("page", sort=False)
    out = df[["page", "i"]].copy()
    for c in SCORE_COLS:
        out[c] = df[c].astype(float)
    for c in ("nll", "dent"):
        out[f"{c}_prev"] = g[c].shift(1).fillna(0.0).values
        out[f"{c}_next"] = g[c].shift(-1).fillna(0.0).values
        out[f"{c}_page_mean"] = g[c].transform("mean").values
        out[f"{c}_page_rank"] = g[c].rank(pct=True).values
    n = g["i"].transform("count")
    out["pos"] = (g.cumcount() / n).values
    out["page_len"] = n.values
    raw = df["text"].astype(str)
    out["n_digits"] = raw.str.count(r"\d").values
    out["has_dot"] = raw.str.contains(".", regex=False).astype(int).values
    for d in DIGITS:
        out[f"has_{d}"] = raw.str.contains(d, regex=False).astype(int).values
    alt = df["alt"].fillna("").astype(str)
    out["alt_one_digit"] = [int(len(a) == len(r) and sum(x != y for x, y in zip(a, r)) == 1) for a, r in zip(alt, raw)]
    out["is_num"] = (df["type"] == "num").values
    return out


def feature_names(c: pd.DataFrame) -> list[str]:
    return [k for k in c.columns if k not in ("page", "i", "is_num")]


class LogReg:
    """L2-regularised logistic regression (Newton steps), with standardisation. numpy only."""

    def __init__(self, l2: float = 1.0, iters: int = 50):
        self.l2, self.iters = l2, iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogReg":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        w = np.zeros(Z.shape[1])
        reg = self.l2 * np.eye(Z.shape[1])
        reg[0, 0] = 0.0  # no penalty on the intercept
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (p - y) + reg @ w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + reg
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))


def cutoff_for_recall(score: np.ndarray, y: np.ndarray, recall: float) -> float:
    """Largest score cut-off t such that flagging score >= t catches at least `recall` of the positives."""
    pos = np.sort(score[y == 1])[::-1]
    if len(pos) == 0:
        return float("inf")
    k = min(len(pos), int(np.ceil(recall * len(pos))))
    return float(pos[k - 1])


def student_folds(students: pd.Series, k: int) -> np.ndarray:
    """Deterministic fold per student (students sorted, dealt round-robin)."""
    uniq = sorted(students.unique())
    fold_of = {s: j % k for j, s in enumerate(uniq)}
    return students.map(fold_of).values


def run(labels: pd.DataFrame, scores: pd.DataFrame, target_recall: float, k: int, l2: float) -> tuple[pd.DataFrame, dict]:
    c = clues(scores)
    feats = feature_names(c)
    df = labels.merge(c, on=["page", "i"])
    df = df[df["is_num"] & (df["op"] != "ins")].reset_index(drop=True)
    y = df["misread"].values.astype(int)
    X = df[feats].values.astype(float)
    folds = student_folds(df["student"], min(k, df["student"].nunique()))
    p, flag = np.zeros(len(df)), np.zeros(len(df), bool)
    base_flag = np.zeros(len(df), bool)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = LogReg(l2).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
        # cut-off from INNER held-out predictions on the training students (in-sample scores are over-confident
        # and would undershoot the target recall on new students)
        inner = np.zeros(int(tr.sum()))
        ftr = folds[tr]
        for g in np.unique(ftr):
            itr, ite = ftr != g, ftr == g
            inner[ite] = LogReg(l2).fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite])
        flag[te] = p[te] >= cutoff_for_recall(inner, y[tr], target_recall)
        base_flag[te] = df["nll"].values[te] >= cutoff_for_recall(df["nll"].values[tr], y[tr], target_recall)
    out = pd.DataFrame({"page": df["page"], "i": df["i"], "student": df["student"], "fold": folds,
                        "p_misread": np.round(p, 5), "flagged": flag})

    def summ(fl):
        return {"recall": round(float(fl[y == 1].mean()), 4), "precision": round(float(y[fl].mean()), 4),
                "share_flagged": round(float(fl.mean()), 4),
                "flags_per_page": round(float(fl.sum() / df["page"].nunique()), 2)}
    metrics = {"numbers": len(df), "misreads": int(y.sum()), "students": int(df["student"].nunique()),
               "folds": int(len(np.unique(folds))), "target_recall": target_recall,
               "selector_heldout": summ(flag), "nll_cutoff_heldout": summ(base_flag), "features": feats}
    return out, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv and scores*.csv")
    p.add_argument("--target-recall", type=float, default=0.8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1.0)
    a = p.parse_args()
    d = Path(a.dir)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str})
    scores = pd.concat([pd.read_csv(f, dtype={"alts": str, "text": str}) for f in sorted(d.glob("scores*.csv"))],
                       ignore_index=True)
    scored = labels.merge(scores[["page", "i"]], on=["page", "i"])
    if scored["student"].nunique() < 3 or scored["misread"].sum() == 0:  # e.g. a 10-page smoke run (the cut-off needs inner folds)
        (d / "selector.csv").unlink(missing_ok=True)
        print("selector: needs >= 3 students and some misreads among the scored pages; skipped (verifier uses top-k)")
        return
    out, metrics = run(labels, scores, a.target_recall, a.folds, a.l2)
    out.to_csv(d / "selector.csv", index=False)
    (d / "selector_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
