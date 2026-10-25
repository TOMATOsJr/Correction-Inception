"""No-API test of the 'derivation as an error-correcting code' idea.

Students reuse their own values, so a value usually appears several times in an answer. A misread corrupts ONE
occurrence, leaving an 'orphan' number whose visually-confusable variant appears repeatedly elsewhere.

Detector (on the Gemini-2.5-Pro transcript only, no gold, no LLM):
  flag number n if   count(n) == 1   and   some confusable variant v of n has count(v) >= 2
  confusable = same length with one digit swapped among visually similar pairs, or a sign flip, or a dropped/added '.'.
Ground truth: aligned numeric tokens that differ between the Gemini transcript and the expert transcript.
Reports precision, recall, and the precision of a random baseline that flags the same number of single-occurrence numbers.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_recognition import DEFAULT_EDU_REPO, RECTIFIED, load_env, load_markdown  # noqa: E402

NUM = re.compile(r"-?\d+(?:\.\d+)?")
SIMILAR = [set("38"), set("17"), set("56"), set("49"), set("02"), set("69"), set("35"), set("57"), set("01"), set("26")]


def confusable(a: str, b: str) -> bool:
    if a == b:
        return False
    if a.lstrip("-") == b.lstrip("-"):
        return True  # sign flip
    if a.replace(".", "") == b.replace(".", ""):
        return True  # decimal point
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and any({diff[0][0], diff[0][1]} == s for s in SIMILAR)


def numbers(md: str) -> list[str]:
    return [n for n in NUM.findall(md) if len(n.lstrip("-").replace(".", "")) >= 1]


def main():
    load_env(DEFAULT_EDU_REPO)
    data = Path(os.environ.get("EDU_DATA", "D:/lma/data/EDU-CIRCUIT-HW_v1"))
    items = pd.read_csv(data / "Screenshot_output_anon" / "set_splitting" / "obsetf_involved_data.csv", dtype=str)
    tgt_root = data / "Observationset_Final" / "v6_Gemini_2p5"
    tp = fp = fn = singles_total = 0
    rand_hits, rand_flags = 0, 0
    examples = []
    rng = random.Random(0)
    n_samples = 0
    for _, r in items.iterrows():
        hw, st, qid = r["Homework ID"], r["Student ID"], r["Question ID"]
        gold = load_markdown(data / RECTIFIED, hw, st, qid, "gemini-2.5-pro")
        tgt = load_markdown(tgt_root, hw, st, qid, "gemini-2.5-pro")
        if gold is None or tgt is None:
            continue
        n_samples += 1
        g, t = numbers(gold), numbers(tgt)
        wrong = set()  # indices in t whose number differs from the aligned gold number
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, g, t, autojunk=False).get_opcodes():
            if tag == "replace":
                wrong.update(range(j1, j2))
        cnt = Counter(t)
        flagged = {j for j, n in enumerate(t)
                   if cnt[n] == 1 and any(cnt[v] >= 2 and confusable(n, v) for v in cnt)}
        singles = [j for j, n in enumerate(t) if cnt[n] == 1]
        singles_total += len(singles)
        tp += len(flagged & wrong)
        fp += len(flagged - wrong)
        fn += len(wrong - flagged)
        if flagged and singles:  # random baseline: same number of flags among single-occurrence numbers
            pick = rng.sample(singles, min(len(flagged), len(singles)))
            rand_flags += len(pick)
            rand_hits += len(set(pick) & wrong)
        for j in sorted(flagged & wrong)[:1]:
            examples.append({"id": f"{hw}/{st}/{qid}", "gemini": t[j],
                             "repeated_variant": [v for v in cnt if cnt[v] >= 2 and confusable(t[j], v)]})
    res = {"samples": n_samples, "true_numeric_misreads": tp + fn, "flags": tp + fp,
           "precision_percent": round(100 * tp / max(1, tp + fp), 1),
           "recall_percent": round(100 * tp / max(1, tp + fn), 1),
           "random_baseline_precision_percent": round(100 * rand_hits / max(1, rand_flags), 1),
           "examples": examples[:12]}
    out = ROOT / "results" / "baseline_reference" / "redundancy_detector_gemini25.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
