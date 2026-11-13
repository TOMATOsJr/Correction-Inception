"""The Auditor on FERMAT: find and undo silent corrections in the reader's own transcripts (exact labels)

FERMAT plants an error in a student's solution (pert_a = what is WRITTEN) next to the correct solution (orig_a). Our
reader transcribed the FERMAT test images (results/pc_run/results/fermat/qwen3b_ft/predictions.jsonl). A number in its
transcript T is, compared with what is written:
  ok        the written value
  silent    the CORRECT value where the student wrote a wrong one: a silent correction (the Auditor's target)
  misread   any other wrong value
Silent corrections are where the reader is confident (it "knows" the right answer), so the misread tools (doubt)
cannot find them. Phases (all resumable, outputs in --out):

  label    (CPU) per-number labels: labels.csv (sample, group, i, start, end, raw, gold = written value, kind)
  score    (GPU) teacher-forced reader confidence of T with the page AND with a blank image (blank = what the reader
           would write from context alone, the signature of a silent correction): scores.csv
  detect   (CPU) learned detector (problem-held-out) for "not what is written", AUROC of every signal for silent and
           misread separately, and the k most suspicious numbers per sample: flagged.csv, detect_metrics.json
  beams    (GPU, reader LoRA) top-k whole-number candidates at each flagged number: beams.csv
  verify   (GPU, A/B checker) "is it A) x or B) c?" for each candidate, both orders, + blank image: verify.csv
  decide   (CPU) the learned Arbiter (arbiter.py, problem-held-out) -> changes; silent corrections / misreads /
           broken numbers before and after, and the silent-correction (reversion) rate of whole answers before/after:
           audit_metrics.json

Groups: problems are hashed into 8 groups (the same problem never in two folds), used as 'student' by the folds.
The red error markers in pert_a are stripped first (fermat_data.strip_red); the transcript never has them.

  python fermat_audit.py label  --pred results/pc_run/results/fermat/qwen3b_ft/predictions.jsonl --out results/checker/audit_fermat
  python fermat_audit.py score  --out ... --adapter local_results/results/reader_lora
  python fermat_audit.py detect --out ... --k 4
  python fermat_audit.py beams  --out ... --adapter local_results/results/reader_lora
  python fermat_audit.py verify --out ... --adapter results/checker/checker_fe
  python fermat_audit.py decide --out ...
"""
from __future__ import annotations

import argparse
from resume import has_rows
import csv
import hashlib
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "finetune"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from fermat_data import PROMPT, reversion, strip_red  # noqa: E402

_TOK = re.compile(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\sA-Za-z\d]")
_NUM = re.compile(r"\d+(?:\.\d+)?")
GROUPS = 8
LABEL_FIELDS = ["page", "student", "i", "start", "end", "raw", "gold", "orig", "kind", "misread", "err_spot"]


def toks(s: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOK.finditer(s or "")]


def is_num(t: str) -> bool:
    return bool(_NUM.fullmatch(t))


def same(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def align_map(a: list[tuple[str, int, int]], b: list[tuple[str, int, int]]) -> dict[int, int]:
    """Token index in a -> aligned token index in b (equal runs, and replace runs of equal length position by position)."""
    out = {}
    sm = SequenceMatcher(None, [t for t, _, _ in a], [t for t, _, _ in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            out.update({i1 + k: j1 + k for k in range(i2 - i1)})
    return out


def group_of(orig_q: str) -> str:
    return f"g{int(hashlib.sha256((orig_q or '').encode()).hexdigest(), 16) % GROUPS}"


def label_sample(idx: int, transcript: str, pert: str, orig: str, group: str) -> list[dict]:
    """Per number of the transcript: what is written there (gold) and whether the transcript silently corrected it."""
    T, W, O = toks(transcript), toks(strip_red(pert)), toks(orig)
    t2w, w2o = align_map(T, W), align_map(W, O)
    rows = []
    for ti, (t, s, e) in enumerate(T):
        if not is_num(t) or ti not in t2w:
            continue
        wi = t2w[ti]
        w = W[wi][0]
        if not is_num(w):
            continue
        o = O[w2o[wi]][0] if wi in w2o else w
        err_spot = is_num(o) and not same(o, w)
        if same(t, w):
            kind = "ok"
        elif err_spot and same(t, o):
            kind = "silent"
        else:
            kind = "misread"
        rows.append({"page": str(idx), "student": group, "i": ti, "start": s, "end": e, "raw": t, "gold": w,
                     "orig": o if is_num(o) else "", "kind": kind, "misread": int(kind != "ok"), "err_spot": int(err_spot)})
    return rows


def read_preds(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def cmd_label(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for r in read_preds(Path(a.pred)):
        T = strip_red(r.get("output") or "")  # the reader copied the markers from its training targets 64 times
        g = group_of(r.get("orig_q", ""))
        rows += label_sample(int(r["idx"]), T, r.get("pert_a") or "", r.get("orig_a") or "", g)
        samples.append({"page": str(r["idx"]), "student": g, "transcript": T, "pert_a": strip_red(r.get("pert_a") or ""),
                        "orig_a": r.get("orig_a") or "", "has_error": bool(r.get("has_error"))})
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, LABEL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    summary = {"samples": len(samples), "numbers": len(rows), "kinds": kinds,
               "error_spot_numbers": sum(r["err_spot"] for r in rows),
               "silent_share_of_error_spots": round(kinds.get("silent", 0) / max(1, sum(r["err_spot"] for r in rows)), 4)}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------ GPU helpers
def samples_of(out: Path) -> dict[str, dict]:
    return {s["page"]: s for s in (json.loads(x) for x in (out / "samples.jsonl").open(encoding="utf-8"))}


def fermat_images(pages: set[str]):
    """idx (position in our FERMAT test split, as eval_fermat.py) -> page image."""
    from fermat_data import load_fermat
    ds = load_fermat()
    test = ds.filter(lambda r: r["split"] == "test")
    return {p: test[int(p)]["image"].convert("RGB") for p in sorted(pages, key=int)}


def shard_of(items: list, shard: str | None) -> list:
    if not shard:
        return items
    i, n = map(int, shard.split("/"))
    return items[i::n]


def number_features(pieces: list[dict], s: int, e: int) -> dict:
    """The number's reader features from its token pieces (weakest piece wins), as score_tokens.aggregate."""
    ps = [p for p in pieces if p["start"] < e and p["end"] > s]
    if not ps:
        return {}
    weak = min(ps, key=lambda p: p["gap"])
    return {"nll": max(-p["lp"] for p in ps), "neg_gap": -weak["gap"], "ent": max(p["ent"] for p in ps),
            "dent": max(p["dent"] for p in ps)}


SCORE_FIELDS = ["page", "i", "nll", "neg_gap", "ent", "dent", "nll_blank", "neg_gap_blank", "dent_blank", "alts"]


def cmd_score(a):
    import pandas as pd
    from PIL import Image
    from score_tokens import TokenScorer
    out = Path(a.out)
    labels = pd.read_csv(out / "labels.csv", dtype={"page": str, "raw": str, "gold": str})
    samples = samples_of(out)
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    path = out / f"scores{suffix}.csv"
    done = set(pd.read_csv(path, dtype={"page": str})["page"]) if has_rows(path) else set()
    pages = shard_of(sorted(set(labels["page"]), key=int), a.shard)
    todo = [p for p in pages if p not in done][: a.limit]
    print(f"{len(todo)} samples to score ({len(done)} done)")
    images = fermat_images(set(todo))
    scorer = TokenScorer(a.reader, a.adapter, a.max_pixels, a.device)
    by_page = {p: g for p, g in labels.groupby("page")}
    new = not has_rows(path)
    from tqdm import tqdm
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SCORE_FIELDS)
        if new:
            w.writeheader()
            f.flush()
        for p in tqdm(todo, desc="scoring"):
            T, img = samples[p]["transcript"], images[p]
            page, _ = scorer.token_features(PROMPT, img, T, a.max_answer_tokens)
            blank, _ = scorer.token_features(PROMPT, Image.new("RGB", img.size, "white"), T, a.max_answer_tokens)
            rows = []
            for _, r in by_page[p].iterrows():
                s, e = int(r["start"]), int(r["end"])
                fp, fb = number_features(page, s, e), number_features(blank, s, e)
                if not fp:
                    continue
                ps = [q for q in page if q["start"] < e and q["end"] > s and q.get("alt_digits")
                      and q["start"] >= s and q["end"] <= e]
                alts = ""
                if ps:
                    wd = max(ps, key=lambda q: (q["dent"], -q["gap"]))
                    alts = "|".join(T[s:wd["start"]] + d + T[wd["end"]:e] for d in wd["alt_digits"])
                rows.append({"page": p, "i": int(r["i"]), "nll": round(fp["nll"], 5), "neg_gap": round(fp["neg_gap"], 5),
                             "ent": round(fp["ent"], 5), "dent": round(fp["dent"], 5),
                             "nll_blank": round(fb.get("nll", 0.0), 5), "neg_gap_blank": round(fb.get("neg_gap", 0.0), 5),
                             "dent_blank": round(fb.get("dent", 0.0), 5), "alts": alts})
            w.writerows(rows)
            f.flush()


# ------------------------------------------------------------------ detect (CPU)
DETECT_FEATURES = ["nll", "neg_gap", "ent", "dent", "nll_blank", "neg_gap_blank", "dent_blank", "blank_minus_page",
                   "after_equals", "repeats", "n_digits", "pos", "is_last"]


def detect_features(labels, scores, samples):
    import numpy as np
    df = labels.merge(scores, on=["page", "i"])
    df["blank_minus_page"] = df["nll_blank"] - df["nll"]  # small: the value is as predictable without the page
    after, reps, pos, last = [], [], [], []
    for _, r in df.iterrows():
        T = samples[r["page"]]["transcript"]
        before = T[max(0, int(r["start"]) - 4):int(r["start"])]
        after.append(float("=" in before))
        reps.append(float(len(re.findall(rf"(?<![\d.]){re.escape(str(r['raw']))}(?![\d.])", T)) - 1))
        pos.append(int(r["start"]) / max(1, len(T)))
    df["after_equals"], df["repeats"], df["pos"] = after, reps, pos
    df["n_digits"] = df["raw"].astype(str).str.count(r"\d")
    mx = df.groupby("page")["i"].transform("max")
    df["is_last"] = (df["i"] == mx).astype(float)
    return df, df[DETECT_FEATURES].astype(float).fillna(0.0).values


def cmd_detect(a):
    import numpy as np
    import pandas as pd
    from evaluate_detection import auroc
    from selector import LogReg, student_folds
    out = Path(a.out)
    labels = pd.read_csv(out / "labels.csv", dtype={"page": str, "raw": str, "gold": str})
    scores = pd.concat([pd.read_csv(f, dtype={"page": str, "alts": str}) for f in sorted(out.glob("scores*.csv"))])
    df, X = detect_features(labels, scores, samples_of(out))
    y = df["misread"].values.astype(float)
    folds = student_folds(df["student"], min(GROUPS, df["student"].nunique()))
    p = np.zeros(len(df))
    for f in np.unique(folds):
        te = folds == f
        if y[~te].sum():
            p[te] = LogReg(1.0).fit(X[~te], y[~te]).predict(X[te])
    df["p_flag"] = p
    # which numbers the fix step checks: the learned detector, or the blank-image signal alone (a silently corrected
    # number is as predictable without the page as with it: small nll_blank - nll), which worked better on FERMAT
    rank_col = {"learned": "p_flag", "blank": "neg_blank_minus_page", "doubt": "dent"}[a.select]
    df["neg_blank_minus_page"] = -df["blank_minus_page"]
    flagged = df.sort_values(rank_col, ascending=False, kind="stable").groupby("page", sort=False).head(a.k)
    flagged.to_csv(out / "flagged.csv", index=False)
    met = {"numbers": len(df), "by_kind": df["kind"].value_counts().to_dict(), "k_per_sample": a.k, "select": a.select,
           "flagged": len(flagged)}
    for kind in ("silent", "misread"):
        m = df["kind"].isin([kind, "ok"])
        yy = (df.loc[m, "kind"] == kind).astype(int).values
        if 0 < yy.sum() < len(yy):
            met[f"auroc_{kind}_vs_ok"] = {"learned_detector": round(auroc(yy, df.loc[m, "p_flag"].values), 4),
                                          **{c: round(auroc(yy, df.loc[m, c].astype(float).fillna(0).values), 4)
                                             for c in DETECT_FEATURES}}
        met[f"{kind}_caught_in_top_k"] = int((flagged["kind"] == kind).sum())
        met[f"{kind}_total"] = int((df["kind"] == kind).sum())
    (out / "detect_metrics.json").write_text(json.dumps(met, indent=2))
    print(json.dumps(met, indent=2))


# ------------------------------------------------------------------ beams / verify (GPU)
def cmd_beams(a):
    import pandas as pd
    from beam_candidates import BeamReader, distinct_candidates
    from score_tokens import TokenScorer
    out = Path(a.out)
    flagged = pd.read_csv(out / "flagged.csv", dtype={"page": str, "raw": str, "gold": str})
    samples = samples_of(out)
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    path = out / f"beams{suffix}.csv"
    done = set(pd.read_csv(path, dtype={"page": str})["page"]) if has_rows(path) else set()
    todo = [p for p in shard_of(sorted(set(flagged["page"]), key=int), a.shard) if p not in done][: a.limit]
    images = fermat_images(set(todo))
    reader = BeamReader(TokenScorer(a.reader, a.adapter, a.max_pixels, a.device), PROMPT, a.beams, 8, 2048)
    by_page = {p: g for p, g in flagged.groupby("page")}
    new = not has_rows(path)
    from tqdm import tqdm
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["page", "i", "written", "beams", "beam_p"])
        if new:
            w.writeheader()
            f.flush()
        for p in tqdm(todo, desc="beams"):
            T = samples[p]["transcript"]
            rows = []
            for _, r in by_page[p].iterrows():
                s, e = int(r["start"]), int(r["end"])
                cands = distinct_candidates(reader.continuations(images[p], T[:s]), T[s:e], a.keep)
                rows.append({"page": p, "i": int(r["i"]), "written": T[s:e], "beams": "|".join(c for c, _ in cands),
                             "beam_p": "|".join(f"{q:.4f}" for _, q in cands)})
            w.writerows(rows)
            f.flush()


VERIFY_FIELDS = ["page", "student", "i", "start", "end", "text", "gold", "misread", "kind", "neg_gap", "beams", "cands",
                 "p_cands", "best", "p_change", "consistent", "p_blank", "cand_is_key", "orig_is_key", "propose"]


def cmd_verify(a):
    import pandas as pd
    from PIL import Image
    from choice_prompt import masked_line
    from score_tokens import TokenScorer
    from verify_choice import ChoiceAsker, candidates, decide
    out = Path(a.out)
    flagged = pd.read_csv(out / "flagged.csv", dtype={"page": str, "raw": str, "gold": str, "alts": str})
    bf = sorted(out.glob("beams*.csv"))
    beams = {}
    for f in bf:
        for _, b in pd.read_csv(f, dtype={"page": str, "beams": str}).iterrows():
            beams[(b["page"], int(b["i"]))] = [c for c in str(b["beams"] if pd.notna(b["beams"]) else "").split("|") if c]
    samples = samples_of(out)
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    path = out / f"verify{suffix}.csv"
    done = set(pd.read_csv(path, dtype={"page": str})["page"]) if has_rows(path) else set()
    todo = [p for p in shard_of(sorted(set(flagged["page"]), key=int), a.shard) if p not in done][: a.limit]
    images = fermat_images(set(todo))
    asker = ChoiceAsker(TokenScorer(a.reader, a.adapter, a.max_pixels, a.device))
    by_page = {p: g for p, g in flagged.groupby("page")}
    new = not has_rows(path)
    from tqdm import tqdm
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, VERIFY_FIELDS)
        if new:
            w.writeheader()
            f.flush()
        for p in tqdm(todo, desc="verifying"):
            T, img = samples[p]["transcript"], images[p]
            rows = []
            for _, r in by_page[p].iterrows():
                s, e = int(r["start"]), int(r["end"])
                x = T[s:e]
                bm = beams.get((p, int(r["i"])), [])
                cands = candidates(x, r.get("alts"), a.max_cands, first=tuple(bm))
                line = masked_line(T, s, e)
                scored = [(c, *asker.challenge(img, line, x, c)) for c in cands]
                dec = decide(scored)
                p_blank = float("nan")
                if dec["best"] and dec["p_change"] > 0.5:
                    p_blank = round(asker.challenge(Image.new("RGB", img.size, "white"), line, x, dec["best"])[0], 5)
                rows.append({"page": p, "student": r["student"], "i": int(r["i"]), "start": s, "end": e, "text": x,
                             "gold": r["gold"], "misread": int(r["misread"]), "kind": r["kind"], "neg_gap": r["neg_gap"],
                             "beams": "|".join(bm), "cands": "|".join(cands),
                             "p_cands": "|".join(f"{c}:{pc:.4f}" for c, pc, _ in scored), "best": dec["best"],
                             "p_change": dec["p_change"], "consistent": dec["consistent"], "p_blank": p_blank,
                             "cand_is_key": False, "orig_is_key": False, "propose": dec["propose"]})
            w.writerows(rows)
            f.flush()


# ------------------------------------------------------------------ decide (CPU)
def apply_changes(T: str, changes: list[tuple[int, int, str, str]]) -> str:
    for s, e, old, new in sorted(changes, reverse=True):
        if T[s:e] == old:
            T = T[:s] + new + T[e:]
    return T


def cmd_decide(a):
    import numpy as np
    import pandas as pd
    from arbiter import CHANGE, evidence, evidence_aurocs, learn_actions, summary, weights
    from selector import student_folds
    out = Path(a.out)
    df = pd.concat([pd.read_csv(f, dtype={"page": str, "gold": str, "best": str, "text": str, "beams": str, "p_cands": str})
                    for f in sorted(out.glob("verify*.csv"))], ignore_index=True)
    bf = sorted(out.glob("beams*.csv"))
    beams = pd.concat([pd.read_csv(f, dtype={"page": str, "beams": str, "beam_p": str}) for f in bf]) if bf else None
    sc = pd.concat([pd.read_csv(f, dtype={"page": str}) for f in sorted(out.glob("scores*.csv"))])
    X = evidence(df, beams, sc)
    folds = student_folds(df["student"], min(a.folds, df["student"].nunique()))
    res = learn_actions(df, X, folds, a.max_false_change)
    labels = pd.read_csv(out / "labels.csv", dtype={"page": str, "raw": str, "gold": str})
    labels = labels[labels["page"].isin(df["page"])]
    samples = samples_of(out)
    met = {"samples": int(df["page"].nunique()), "taus": res["taus"],
           "evidence_auroc": evidence_aurocs(X, res["good"], res["score"]),
           "weights_all_data (standardised features)": weights(X, res["good"])}
    # the simple rule of correct.py (checker probability + both orders + guards, threshold per fold for <= 1% false
    # changes), which beat the learned Arbiter on EDU
    from correct import actions as simple_actions, pick_tau
    simple = np.full(len(df), "keep", dtype=object)
    for f in np.unique(folds):
        te = folds == f
        t = pick_tau(df[~te], 0.35, 0.15, a.max_false_change)
        simple[te] = simple_actions(df[te], t, 0.35, 0.15)
    res["actions"]["simple"] = np.where(simple == CHANGE, CHANGE, "keep")
    for rule in ("simple", "net", "budget"):
        act = res["actions"][rule]
        ch = act == CHANGE
        m = summary(df, act, res["good"], res["wrong"], res["broken"], int(labels["misread"].sum()))
        for kind in ("silent", "misread"):
            k = df["kind"].values == kind
            m[f"{kind}_before"] = int((labels["kind"] == kind).sum())
            m[f"{kind}_checked"] = int(k.sum())
            m[f"{kind}_fixable"] = int((res["good"] & k).sum())
            m[f"{kind}_fixed"] = int((ch & res["good"] & k).sum())
        m["silent_after"] = m["silent_before"] - m["silent_fixed"]
        # whole answers: the FERMAT silent-correction (reversion) rate, before and after the changes
        changes: dict[str, list] = {}
        for (_, r), c in zip(df.iterrows(), ch):
            if c:
                changes.setdefault(r["page"], []).append((int(r["start"]), int(r["end"]), str(r["text"]), str(r["best"])))
        rev_b, rev_a = [], []
        for p in df["page"].unique():
            s = samples[p]
            if not s["has_error"]:
                continue
            before = reversion(s["transcript"], s["pert_a"], s["orig_a"])
            after = reversion(apply_changes(s["transcript"], changes.get(p, [])), s["pert_a"], s["orig_a"])
            if before is not None and after is not None:
                rev_b.append(before)
                rev_a.append(after)
        m["answers_evaluable"] = len(rev_b)
        m["silent_correction_rate_before_percent"] = round(100 * float(np.mean(rev_b)), 2) if rev_b else None
        m["silent_correction_rate_after_percent"] = round(100 * float(np.mean(rev_a)), 2) if rev_a else None
        met[rule] = m
    (out / "audit_metrics.json").write_text(json.dumps(met, indent=2))
    df[["page", "student", "i", "kind"]].assign(score=np.round(res["score"], 5), action=res["actions"]["simple"],
                                                action_net=res["actions"]["net"]).to_csv(
        out / "decisions.csv", index=False)
    print(json.dumps(met, indent=2))


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("label", "score", "detect", "beams", "verify", "decide"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="results/checker/audit_fermat")
        s.add_argument("--pred", default="results/pc_run/results/fermat/qwen3b_ft/predictions.jsonl")
        s.add_argument("--reader", default="Qwen/Qwen2.5-VL-3B-Instruct")
        s.add_argument("--adapter", default=None)
        s.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
        s.add_argument("--max-answer-tokens", type=int, default=2048)
        s.add_argument("--k", type=int, default=4, help="detect: numbers checked per sample")
        s.add_argument("--select", choices=["learned", "blank", "doubt"], default="learned",
                       help="detect: rank numbers by the learned detector, the blank-image signal, or digit doubt")
        s.add_argument("--beams", type=int, default=6)
        s.add_argument("--keep", type=int, default=4)
        s.add_argument("--max-cands", type=int, default=4)
        s.add_argument("--max-false-change", type=float, default=0.01)
        s.add_argument("--folds", type=int, default=4)
        s.add_argument("--device", type=int, default=0)
        s.add_argument("--shard", default=None)
        s.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    {"label": cmd_label, "score": cmd_score, "detect": cmd_detect, "beams": cmd_beams, "verify": cmd_verify,
     "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
