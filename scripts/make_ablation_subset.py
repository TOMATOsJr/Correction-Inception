"""Pick the exam subset for the ablations: N exams that have an expert transcript, a rubric and a transcript
from EVERY target transcriber (so the transcriber comparison is on identical exams), spread evenly over
students (round-robin over a seeded shuffle, so no single student dominates the bootstrap CIs).

  python scripts/make_ablation_subset.py --edu-data D --n 80 --out results/ablation/subset_items.csv \
      --target local_results/results/Outputs/qwen3b_ft:qwen3b_ft \
      --target D/Observationset_Final/v6_Gemini_2p5:gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
import pandas as pd  # noqa: E402

from evaluate_recognition import RECTIFIED, load_markdown  # noqa: E402
from grade_impact import rubric_path  # noqa: E402

ITEMS = ROOT / "src" / "eval" / "items_with_expert_transcript.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edu-data", required=True)
    p.add_argument("--target", action="append", required=True, help="ROOT:MODEL_DIR (repeatable)")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = Path(a.edu_data)
    targets = [tuple(t.rsplit(":", 1)) for t in a.target]

    df = pd.read_csv(ITEMS, dtype=str)
    by_student, dropped = defaultdict(list), 0
    for _, r in df.iterrows():
        hw, st, qid = r["Homework ID"], r["Student ID"], r["Question ID"]
        ok = (rubric_path(data, qid).exists() and load_markdown(data / RECTIFIED, hw, st, qid, "gemini-2.5-pro") is not None
              and all(load_markdown(Path(root), hw, st, qid, mdir) is not None for root, mdir in targets))
        if ok:
            by_student[st].append(r)
        else:
            dropped += 1
    usable = sum(len(v) for v in by_student.values())
    print(f"{usable} exams usable by every target ({dropped} dropped: missing transcript/rubric/expert)")

    rng = random.Random(a.seed)
    for rows in by_student.values():
        rng.shuffle(rows)
    picked, students = [], sorted(by_student)
    while len(picked) < min(a.n, usable):
        for st in students:
            if by_student[st] and len(picked) < a.n:
                picked.append(by_student[st].pop())
    out = pd.DataFrame(picked).sort_values(["Homework ID", "Student ID", "Question ID"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} exams from {out['Student ID'].nunique()} students -> {a.out}")
    print(out["Student ID"].value_counts().to_string())


if __name__ == "__main__":
    main()
