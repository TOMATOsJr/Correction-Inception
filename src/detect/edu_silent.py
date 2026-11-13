"""Silent corrections in real exams (EDU-CIRCUIT-HW), found through the rubric's answer key

A silent correction moves the transcript towards the expected answer. With the expert transcript (what is written)
and the rubric's answer-key numbers (notes and deduction rules, verify_choice.key_numbers), every transcript number
that differs from the expert's is either:
  toward_key   the transcript shows an answer-key value and the expert's (written) value is not one: silent-correction
               like, it would give marks the student did not earn (the Auditor's target)
  away_key     the expert's value is an answer-key value and the transcript's is not: a misread that costs marks
  other        neither (or both) in the key
The share of correct numbers that are key values is the base rate: "toward_key" only means something above it.

  analyse  (CPU) per transcriber (labels.csv from labels.py): counts, rates, and the pages involved
  blank    (GPU) teacher-forced reader confidence of the transcript with a BLANK image (the page scores already exist):
           a silent correction is as predictable without the page as with it. Then `analyse --blank` reports how well
           nll_blank - nll separates toward_key misreads from correct numbers and from other misreads

  python edu_silent.py analyse --labels results/correction_600k/qwen_ftreader/labels.csv --edu-data E --name qwen3b_ft \\
      --out results/checker/edu_silent
  python edu_silent.py blank --labels ... --edu-data E --images E --target-root R --model-dir qwen3b_ft \\
      --adapter local_results/results/reader_lora --out results/checker/edu_silent/qwen3b_ft
  python edu_silent.py analyse ... --scores results/correction_600k/qwen_ftreader --blank
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_detection import _num  # noqa: E402


def qid_of(stem: str) -> str:
    return re.sub(r"_[1-9]$", "", stem)


def key_sets(labels, edu_data: Path) -> dict[str, set]:
    from grade_impact import rubric_path
    from verify_choice import key_numbers
    keys = {}
    for pid in labels["page"].unique():
        q = qid_of(pid.split("|")[2])
        if q not in keys:
            rp = rubric_path(edu_data, q)
            keys[q] = key_numbers(json.loads(rp.read_text(encoding="utf-8"))) if rp.exists() else set()
    return keys


def categorise(labels, keys: dict[str, set]):
    """Adds in_key_x, in_key_gold, category (ok / toward_key / away_key / other) to the numbers of labels.csv."""
    df = labels[(labels["type"] == "num") & (labels["op"] != "ins")].copy()
    q = df["page"].map(lambda p: qid_of(p.split("|")[2]))
    xv = df["raw"].astype(str).map(_num)
    gv = df["gold_raw"].fillna("").astype(str).map(_num)
    df["in_key_x"] = [v is not None and v in keys.get(k, set()) for v, k in zip(xv, q)]
    df["in_key_gold"] = [v is not None and v in keys.get(k, set()) for v, k in zip(gv, q)]
    df["has_key"] = [bool(keys.get(k)) for k in q]
    cat = []
    for m, kx, kg in zip(df["misread"].astype(int), df["in_key_x"], df["in_key_gold"]):
        cat.append("ok" if m == 0 else "toward_key" if kx and not kg else "away_key" if kg and not kx else "other")
    df["category"] = cat
    return df


def blank_aurocs(df, scores_dir: Path, blank_dir: Path) -> dict:
    import pandas as pd
    from evaluate_detection import auroc
    sc = pd.concat([pd.read_csv(f, usecols=["page", "i", "nll", "dent"]) for f in sorted(scores_dir.glob("scores*.csv"))])
    bl = pd.concat([pd.read_csv(f) for f in sorted(blank_dir.glob("blank_scores*.csv"))])
    m = df.merge(sc.drop_duplicates(["page", "i"]), on=["page", "i"]).merge(bl.drop_duplicates(["page", "i"]), on=["page", "i"])
    m["blank_minus_page"] = m["nll_blank"] - m["nll"]
    out = {"numbers_scored": len(m)}
    for pos, neg in (("toward_key", "ok"), ("toward_key", "away_key"), ("toward_key", "other"), ("away_key", "ok")):
        s = m[m["category"].isin([pos, neg])]
        y = (s["category"] == pos).astype(int).values
        if 0 < y.sum() < len(y):
            res = {}
            for c in ("blank_minus_page", "nll_blank", "nll", "dent"):
                a = auroc(y, s[c].astype(float).values)
                res[c] = {"auroc": round(a, 4), "direction": "high" if a >= 0.5 else "low",
                          "direction_free": round(max(a, 1 - a), 4)}
            out[f"{pos}_vs_{neg} (n={int(y.sum())} vs {int(len(y) - y.sum())})"] = res
    return out


def cmd_analyse(a):
    import pandas as pd
    labels = pd.read_csv(a.labels, dtype={"gold": str, "gold_raw": str, "raw": str})
    df = categorise(labels, key_sets(labels, Path(a.edu_data)))
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    ok = df[df["category"] == "ok"]
    mis = df[df["misread"] == 1]
    res = {"name": a.name, "pages": int(df["page"].nunique()), "numbers": len(df),
           "numbers_on_pages_with_a_key": int(df["has_key"].sum()), "misreads": len(mis),
           "categories": df["category"].value_counts().to_dict(),
           "base_rate_correct_numbers_in_key": round(float(ok["in_key_x"].mean()), 4) if len(ok) else None,
           "misreads_whose_transcript_value_is_in_key": round(float(mis["in_key_x"].mean()), 4) if len(mis) else None,
           "misreads_whose_written_value_is_in_key": round(float(mis["in_key_gold"].mean()), 4) if len(mis) else None,
           "pages_with_toward_key": int(df.loc[df["category"] == "toward_key", "page"].nunique())}
    if a.blank:
        res["blank_image"] = blank_aurocs(df, Path(a.scores), out)
    (out / "edu_silent_metrics.json").write_text(json.dumps(res, indent=2))
    df[["page", "student", "i", "raw", "gold_raw", "category", "in_key_x", "in_key_gold"]].to_csv(out / "numbers.csv",
                                                                                                  index=False)
    print(json.dumps(res, indent=2))


def cmd_blank(a):
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    from labels import page_text
    from run_ocr import DEFAULT_EDU_REPO, DEFAULT_PROMPT
    from score_tokens import TokenScorer, aggregate, jobs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    path = out / f"blank_scores{suffix}.csv"
    done = set(pd.read_csv(path)["page"]) if path.exists() else set()
    prompt = Path(a.prompt_file or Path(DEFAULT_EDU_REPO) / DEFAULT_PROMPT).read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{problem_statement}", "")
    a.images = a.images or a.edu_data
    todo = [j for j in jobs(a) if j[0] not in done]
    print(f"{len(todo)} pages to score with a blank image ({len(done)} done)")
    scorer = TokenScorer(a.reader, a.adapter, a.max_pixels, a.device)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["page", "i", "nll_blank", "neg_gap_blank", "dent_blank"])
        if new:
            w.writeheader()
        for pid, md, img in tqdm(todo, desc="blank scoring"):
            answer = page_text(md)
            size = Image.open(img).size
            pieces, _ = scorer.token_features(prompt, Image.new("RGB", size, "white"), answer, a.max_answer_tokens)
            rows = [{"page": pid, "i": r["i"], "nll_blank": r["nll"], "neg_gap_blank": r["neg_gap"], "dent_blank": r["dent"]}
                    for r in aggregate(answer, pieces) if r["type"] == "num"]
            w.writerows(rows)
            f.flush()


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyse")
    an.add_argument("--labels", required=True)
    an.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    an.add_argument("--name", required=True)
    an.add_argument("--out", default="results/checker/edu_silent")
    an.add_argument("--blank", action="store_true", help="also report the blank-image signal (needs `blank` first)")
    an.add_argument("--scores", default="results/correction_600k/qwen_ftreader", help="folder with the page scores*.csv")
    bl = sub.add_parser("blank")
    bl.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    bl.add_argument("--images", default=None)
    bl.add_argument("--target-root", required=True)
    bl.add_argument("--model-dir", required=True)
    bl.add_argument("--out", required=True, help="results/checker/edu_silent/<name>")
    bl.add_argument("--reader", default="Qwen/Qwen2.5-VL-3B-Instruct")
    bl.add_argument("--adapter", default=None)
    bl.add_argument("--prompt-file", default=None)
    bl.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    bl.add_argument("--max-answer-tokens", type=int, default=3072)
    bl.add_argument("--device", type=int, default=0)
    bl.add_argument("--shard", default=None)
    bl.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    {"analyse": cmd_analyse, "blank": cmd_blank}[a.cmd](a)


if __name__ == "__main__":
    main()
