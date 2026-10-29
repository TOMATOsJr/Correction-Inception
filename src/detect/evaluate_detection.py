"""Does reader confidence find misread numbers? Scores (score_tokens.py) vs labels (labels.py). (Phase 2)

Population: number tokens that the alignment matched to a gold number, either equal (negative) or different
(positive = misread). 'ins' numbers are left out (mostly alignment noise / looping), see labels.py.

Per signal (higher = more suspicious):
  AUROC, average precision (base rate = AP of a random ranking), student-clustered bootstrap 95% CI on AUROC
  precision@k / recall@k: flag the k most suspicious numbers on each page (the agents' claim budget is per page)
Baselines: random, and n_pieces (number length in tokens: longer numbers are misread more, for free).

  python evaluate_detection.py --dir results/detect/qwen3b_ft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ["nll", "neg_gap", "ent", "dent", "n_pieces", "random"]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    if hits.sum() == 0:
        return float("nan")
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def at_k(df: pd.DataFrame, col: str, k: int) -> tuple[float, float]:
    """Flag the top-k numbers per page by `col`: (precision over all flags, recall over all misreads)."""
    top = df.sort_values(col, ascending=False, kind="stable").groupby("page", sort=False).head(k)
    return float(top["misread"].mean()), float(top["misread"].sum() / max(1, df["misread"].sum()))


def bootstrap_auroc(df: pd.DataFrame, col: str, n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("student")]
    vals = []
    for _ in range(n):
        b = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(auroc(b["misread"].values, b[col].values))
    return [round(float(np.nanpercentile(vals, 2.5)), 4), round(float(np.nanpercentile(vals, 97.5)), 4)]


def load(d: Path) -> pd.DataFrame:
    labels = pd.read_csv(d / "labels.csv")
    scores = pd.concat([pd.read_csv(f) for f in sorted(d.glob("scores*.csv"))], ignore_index=True)
    df = labels.merge(scores[[c for c in ["page", "i", "nll", "neg_gap", "ent", "dent", "alt", "alts", "n_pieces"] if c in scores]], on=["page", "i"])
    df = df[(df["type"] == "num") & (df["op"] != "ins")].copy()
    df["random"] = np.random.default_rng(0).random(len(df))
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv and scores*.csv")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    d = Path(a.dir)
    df = load(d)
    if df.empty:
        raise SystemExit(f"no scored number tokens in {d}: run labels.py and score_tokens.py on the same transcripts")
    y = df["misread"].values
    res = {"pages": int(df["page"].nunique()), "students": int(df["student"].nunique()), "numbers": len(df),
           "misreads": int(y.sum()), "base_rate": round(float(y.mean()), 4), "signals": {}}
    for col in SIGNALS:
        s = df[col].values.astype(float)
        r = {"auroc": round(auroc(y, s), 4), "auroc_95CI": bootstrap_auroc(df, col, a.boot),
             "avg_precision": round(average_precision(y, s), 4)}
        for k in (1, 3, 5):
            prec, rec = at_k(df, col, k)
            r[f"precision@{k}"], r[f"recall@{k}"] = round(prec, 4), round(rec, 4)
        res["signals"][col] = r
    # does the reader's preferred reading recover the expert's number? (the 'proposed' a claim would carry)
    mis = df[df["misread"] == 1]
    gold_alt = mis.apply(lambda r: _num(r["alt"]) is not None and _num(r["alt"]) == float(r["gold"]), axis=1)
    res["alt_recovers_gold_percent"] = round(100 * float(gold_alt.mean()), 2) if len(mis) else None
    if "alts" in mis:  # is the gold value among the reader's top-3 digit swaps (the verifier's candidates)?
        in_top = mis.apply(lambda r: float(r["gold"]) in {_num(c) for c in str(r["alts"]).split("|")}, axis=1)
        res["alts_top3_recover_gold_percent"] = round(100 * float(in_top.mean()), 2) if len(mis) else None
    (d / "detection_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    cols = ["auroc", "auroc_95CI", "avg_precision", "precision@3", "recall@3"]
    print(pd.DataFrame(res["signals"]).T[cols].to_string())


def _num(s) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


if __name__ == "__main__":
    main()
