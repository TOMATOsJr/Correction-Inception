"""Learned Arbiter: combine every tool's evidence to decide change / keep for each verified number (Phase 4c)

correct.py decides from one number, the checker's probability p_change (with two hard guards). That forces a very high
threshold, so almost nothing changes. But each verified number carries several independent pieces of evidence:

  checker      p_change (best challenger), margin over the second candidate, both option orders agree
  blank image  p_blank: the checker's preference for the challenger WITHOUT the page (guessing from context)
  beams        is the challenger one of the reader's top-k whole numbers, its rank and beam probability
  reader       doubt in the written value (neg_gap; nll / dent / ent from scores.csv when present)
  shape        one-digit change, length change, decimal-point change
  answer key   challenger / written value in the rubric's answer key (the Auditor's silent-correction flags)

A small L2 logistic regression predicts "changing to the challenger fixes the number" from these. Evaluation is by
held-out group (students for EDU, problems for FERMAT): for each fold the model is trained on the other groups, the
change threshold is chosen on THEIR out-of-fold predictions, and both are applied to the held-out fold.

Threshold rules (both reported; `net` decides):
  net          maximise fixes - wrong changes on the training groups; no change at all unless that is positive
  budget       lowest threshold whose false-change rate (correct numbers changed) is <= --max-false-change

Outputs (in --dir/arbiter): arbiter_metrics.json (same counts as correct_metrics.json, per rule, plus the evidence
AUROCs), decisions.csv, and for EDU the transcripts Outputs/{none,corrected} for transcript_diff / grade_impact.

  python arbiter.py --dir results/checker/correct_checker_fe_k10_beams --target-root R --model-dir qwen3b_ft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detection import _num, auroc  # noqa: E402
from selector import LogReg, student_folds  # noqa: E402

CHANGE, KEEP = "change", "keep"
FEATURES = ["p_change", "consistent", "p_second", "margin", "n_cands", "p_blank", "blank_gap", "blank_missing",
            "in_beams", "beam_rank", "beam_p", "beam_p_top", "neg_gap", "one_digit", "len_diff", "dot_change",
            "cand_is_key", "orig_is_key", "key_move", "nll", "dent", "ent"]


def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def _edit1(a: str, b: str) -> bool:
    """The digit strings differ by exactly one substitution, insertion or deletion."""
    a, b = _digits(a), _digits(b)
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:k] + t[k + 1:] == s for k in range(len(t)))


def _p_cands(s) -> list[tuple[str, float]]:
    out = []
    for part in str(s if isinstance(s, str) else "").split("|"):
        if ":" in part:
            c, p = part.rsplit(":", 1)
            try:
                out.append((c, float(p)))
            except ValueError:
                pass
    return out


def evidence(df: pd.DataFrame, beams: pd.DataFrame | None = None, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row of numeric features per verified number (rows aligned with df)."""
    best = df["best"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    pc = df["p_change"].astype(float).fillna(0.0)
    cands = [sorted((p for _, p in _p_cands(s)), reverse=True) for s in df.get("p_cands", pd.Series("", index=df.index))]
    p_second = np.array([c[1] if len(c) > 1 else 0.0 for c in cands])
    pb = df["p_blank"].astype(float)
    X = pd.DataFrame(index=df.index)
    X["p_change"] = pc
    X["consistent"] = df["consistent"].astype(str).str.lower().isin(["true", "1"]).astype(float)
    X["p_second"] = p_second
    X["margin"] = pc - p_second
    X["n_cands"] = [len(c) for c in cands]
    X["blank_missing"] = pb.isna().astype(float)
    X["p_blank"] = pb.fillna(0.0)
    X["blank_gap"] = (pc - pb).fillna(0.0)
    blist = [[c for c in str(b).split("|") if c] if isinstance(b, str) else [] for b in df.get("beams", pd.Series(np.nan, index=df.index))]
    rank = [next((k for k, c in enumerate(bl) if _num(c) is not None and _num(c) == _num(bs)), -1) for bl, bs in zip(blist, best)]
    X["in_beams"] = [float(r >= 0) for r in rank]
    X["beam_rank"] = [r if r >= 0 else 4 for r in rank]
    X["beam_p"], X["beam_p_top"] = 0.0, 0.0
    if beams is not None and len(beams):
        bp = beams.set_index(["page", "i"])
        for j, (k, r) in enumerate(zip(zip(df["page"], df["i"].astype(int)), rank)):
            if k in bp.index:
                row = bp.loc[k]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                ps = [float(x) for x in str(row.get("beam_p", "")).split("|") if x not in ("", "nan")]
                if ps:
                    X.iat[j, X.columns.get_loc("beam_p_top")] = ps[0]
                    if 0 <= r < len(ps):
                        X.iat[j, X.columns.get_loc("beam_p")] = ps[r]
    X["neg_gap"] = df["neg_gap"].astype(float).fillna(0.0) if "neg_gap" in df else 0.0
    X["one_digit"] = [float(_edit1(b, t)) for b, t in zip(best, text)]
    X["len_diff"] = [abs(len(_digits(b)) - len(_digits(t))) if b else 0 for b, t in zip(best, text)]
    X["dot_change"] = [float(b != "" and (("." in b) != ("." in t))) for b, t in zip(best, text)]
    for c in ("cand_is_key", "orig_is_key"):
        X[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(float) if c in df else 0.0
    X["key_move"] = X["cand_is_key"] * (1 - X["orig_is_key"])
    for c in ("nll", "dent", "ent"):
        X[c] = 0.0
    if scores is not None and len(scores):
        sc = scores.drop_duplicates(["page", "i"]).set_index(["page", "i"])
        keys = list(zip(df["page"], df["i"].astype(int)))
        for c in ("nll", "dent", "ent"):
            if c in sc:
                X[c] = [float(sc[c].get(k, 0.0)) for k in keys]
    return X[FEATURES].astype(float)


def labels_of(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """good: changing to the challenger fixes a misread; wrong: changes a misread to another wrong value;
    broken: changes a correct number. (A row with no challenger can only be kept.)"""
    mis = df["misread"].astype(int).values == 1
    best = df["best"].fillna("").astype(str)
    has = best.str.len().values > 0
    good = np.array([m and h and _num(b) is not None and _num(b) == _num(str(g))
                     for m, h, b, g in zip(mis, has, best, df["gold"].astype(str))])
    return good, mis & has & ~good, ~mis & has


def _sweep(p: np.ndarray):
    """Candidate thresholds (distinct scores, high to low) and, for each, the mask of rows at or above it."""
    for t in np.unique(p)[::-1]:
        yield float(t), p >= t


def net_threshold(p: np.ndarray, good: np.ndarray, bad: np.ndarray, min_net: int = 2) -> float:
    """Threshold maximising fixes - bad changes among rows with p >= t (the highest such t on ties);
    1.01 (never change) if no threshold reaches min_net."""
    best_t, best = 1.01, min_net - 1
    for t, m in _sweep(p):
        net = int(good[m].sum()) - int(bad[m].sum())
        if net > best:
            best_t, best = t, net
    return best_t


def budget_threshold(p: np.ndarray, broken: np.ndarray, n_correct: int, max_fc: float) -> float:
    """Lowest threshold whose correct-number changes stay within max_fc of the correct numbers."""
    allowed = int(np.floor(max_fc * max(1, n_correct)))
    if len(p) == 0 or np.ptp(p) == 0:  # uninformative scores (e.g. no positives to learn from): never change
        return 1.01
    best_t = 1.01
    for t, m in _sweep(p):
        if int(broken[m].sum()) > allowed:
            break
        best_t = t
    return best_t


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, l2: float) -> np.ndarray:
    """Out-of-fold probabilities: each group scored by a model trained on the other groups."""
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        if y[~te].sum() == 0:
            out[te] = 0.0
            continue
        out[te] = LogReg(l2).fit(X[~te], y[~te]).predict(X[te])
    return out


def learn_actions(df: pd.DataFrame, X: pd.DataFrame, folds: np.ndarray, max_fc: float = 0.01, l2: float = 1.0,
                  inner: int = 4) -> dict:
    """Held-out decisions per fold under both threshold rules, plus the held-out arbiter score."""
    good, wrong, broken = labels_of(df)
    has = df["best"].fillna("").astype(str).str.len().values > 0
    Xv, y = X.values, good.astype(float)
    score = np.zeros(len(df))
    acts = {"net": np.full(len(df), KEEP, dtype=object), "budget": np.full(len(df), KEEP, dtype=object)}
    taus: dict[str, list] = {"net": [], "budget": []}
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        if y[tr].sum() == 0:
            continue
        score[te] = LogReg(l2).fit(Xv[tr], y[tr]).predict(Xv[te])
        # threshold from the training groups' own out-of-fold scores (in-sample scores are over-confident)
        tr_groups = student_folds(pd.Series(df["student"].values[tr]), min(inner, len(np.unique(df["student"].values[tr]))))
        p_tr = oof_scores(Xv[tr], y[tr], tr_groups, l2)
        t_net = net_threshold(p_tr, good[tr], (wrong | broken)[tr])
        n_ok_tr = int((df["misread"].astype(int).values[tr] == 0).sum())
        t_bud = budget_threshold(p_tr, broken[tr], n_ok_tr, max_fc)
        for rule, t in (("net", t_net), ("budget", t_bud)):
            taus[rule].append(round(t, 4))
            acts[rule][te & has & (score >= t)] = CHANGE
    return {"score": score, "actions": acts, "taus": taus, "good": good, "wrong": wrong, "broken": broken}


def summary(df: pd.DataFrame, act: np.ndarray, good, wrong, broken, misreads_before: int) -> dict:
    ch = act == CHANGE
    n_fix, n_wrong, n_broken = int((ch & good).sum()), int((ch & wrong).sum()), int((ch & broken).sum())
    n_ok = int((df["misread"].astype(int).values == 0).sum())
    return {"verified": len(df), "verified_misreads": int((df["misread"].astype(int) == 1).sum()),
            "fixable": int(good.sum()), "auto_fixed": n_fix, "auto_wrong_fix": n_wrong, "auto_broken_correct": n_broken,
            "net_fixes": n_fix - n_wrong - n_broken, "false_change_rate": round(n_broken / max(1, n_ok), 4),
            "misreads_before": misreads_before, "misreads_after_auto": misreads_before - n_fix + n_broken}


def evidence_aurocs(X: pd.DataFrame, good: np.ndarray, score: np.ndarray) -> dict:
    """How well each piece of evidence alone, and the combined held-out arbiter score, separate good changes."""
    y = good.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    out = {c: round(max(auroc(y, X[c].values), 1 - auroc(y, X[c].values)), 4) for c in X.columns if X[c].std() > 0}
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]))
    return {"arbiter_heldout": round(auroc(y, score), 4), "single_evidence (direction-free)": out}


def weights(X: pd.DataFrame, good: np.ndarray, l2: float = 1.0) -> dict:
    """The model fitted on all rows, for the report: which evidence the Arbiter leans on (standardised units)."""
    if good.sum() == 0:
        return {}
    m = LogReg(l2).fit(X.values, good.astype(float))
    w = dict(zip(X.columns, np.round(m.w[1:], 3)))
    return dict(sorted(w.items(), key=lambda kv: -abs(kv[1])))


def load_dir(d: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    df = pd.concat([pd.read_csv(f, dtype={"gold": str, "best": str, "text": str, "beams": str, "p_cands": str})
                    for f in sorted(d.glob("verify*.csv"))], ignore_index=True)
    bf = sorted(d.glob("beams*.csv"))
    beams = pd.concat([pd.read_csv(f, dtype={"beams": str, "beam_p": str}) for f in bf]) if bf else None
    sf = sorted(d.glob("scores*.csv"))
    scores = pd.concat([pd.read_csv(f, usecols=lambda c: c in ("page", "i", "nll", "dent", "ent")) for f in sf]) if sf else None
    return df, beams, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with verify*.csv (+ beams*.csv, scores*.csv, labels.csv)")
    p.add_argument("--target-root", default=None, help="EDU transcripts: write Outputs/{none,corrected}")
    p.add_argument("--model-dir", default="qwen3b_ft")
    p.add_argument("--max-false-change", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--l2", type=float, default=1.0)
    a = p.parse_args()
    d = Path(a.dir)
    df, beams, scores = load_dir(d)
    X = evidence(df, beams, scores)
    folds = student_folds(df["student"], min(a.folds, df["student"].nunique()))
    res = learn_actions(df, X, folds, a.max_false_change, a.l2)
    before = int(df["misread"].astype(int).sum())
    if (d / "labels.csv").exists():
        lab = pd.read_csv(d / "labels.csv", usecols=["page", "type", "op", "misread"])
        before = int(lab[(lab["type"] == "num") & (lab["op"] != "ins") & lab["page"].isin(df["page"])]["misread"].sum())
    out = d / "arbiter"
    out.mkdir(exist_ok=True)
    m = {"folds": int(len(np.unique(folds))), "taus": res["taus"],
         **{rule: summary(df, res["actions"][rule], res["good"], res["wrong"], res["broken"], before)
            for rule in ("net", "budget")},
         "evidence_auroc": evidence_aurocs(X, res["good"], res["score"]),
         "weights_all_data (standardised features)": weights(X, res["good"])}
    (out / "arbiter_metrics.json").write_text(json.dumps(m, indent=2))
    df[["page", "student", "i"]].assign(fold=folds, score=np.round(res["score"], 5), action_net=res["actions"]["net"],
                                        action_budget=res["actions"]["budget"]).to_csv(out / "decisions.csv", index=False)
    if a.target_root and (d / "labels.csv").exists():
        from correct import TA, write_variants  # noqa: F401
        labels = pd.read_csv(d / "labels.csv", dtype={"gold": str, "gold_raw": str})
        write_variants(out, df, res["actions"]["net"], labels, Path(a.target_root), a.model_dir)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
