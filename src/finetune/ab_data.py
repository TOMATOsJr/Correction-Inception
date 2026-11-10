"""Training / test pairs for the A/B checker: "is the number marked [?] written as A) x or B) y?"

The checker is the Advocate's and the Auditor's main tool (src/detect/verify_choice.py asks exactly this question,
from src/detect/choice_prompt.py). The untuned reader answers it at chance (it picks the true value over the written
one 45% of the time), so here it gets training pairs whose right answer is always "what is written on the page":

  silent     FERMAT: the student wrote a wrong value (pert_a) where the correct solution (orig_a) has another.
             Options: written (wrong) vs correct. Right answer: the written, wrong one. This is the
             silent-correction test: a reader that prefers "what makes the maths right" fails it.
  lookalike  any written number vs a look-alike digit swap of it (3<->8, 1<->7, ...). Right answer: the written
             one. Balances 'silent' so the checker cannot learn "always pick the mathematically wrong option".
  misread    EDU-CIRCUIT-HW: the transcript says x where the expert transcript says g (only near misses: one digit
             changed, dropped or added; the rest is mostly alignment noise). Options x vs g. Right answer: g.
             The line shown is the transcript's, as at test time.

Every pair is emitted in both option orders (A and B are each right half the time).
Splits: FERMAT by problem (fermat_data.split_of: train / dev / test); EDU by student (whole students only).

  python ab_data.py fermat --out results/checker/pairs                 # needs HF_TOKEN (FERMAT is gated)
  python ab_data.py edu --labels results/detect/qwen3b_ft_ftreader/labels.csv --target-root R --model-dir qwen3b_ft \\
      --images <edu_data> --train-students 3 --out results/checker/pairs

Output: pairs_<source>_<split>.jsonl (quotes student work: git-ignored) and pairs_summary.json (counts only).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "detect"))
from choice_prompt import CONFUSE, masked_line  # noqa: E402

_TOK = re.compile(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\sA-Za-z\d]")
LOOKALIKE_PER_ROW = 2


def tokens_with_offsets(s: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOK.finditer(s or "")]


def is_num(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t))


def same_value(a: str, b: str) -> bool:
    try:
        return float(a.replace(",", "")) == float(b.replace(",", ""))
    except ValueError:
        return a == b


def lookalike(text: str, rng: random.Random) -> str | None:
    """One digit of `text` swapped for a look-alike digit, or None if no digit has one."""
    spots = [(j, d) for j, ch in enumerate(text) if ch in CONFUSE for d in CONFUSE[ch]]
    rng.shuffle(spots)
    for j, d in spots:
        c = text[:j] + d + text[j + 1:]
        if not (len(c) > 1 and c[0] == "0" and c[1] != ".") and not same_value(c, text):  # no '07'
            return c
    return None


def digit_edit(a: str, b: str) -> int:
    """Levenshtein distance between the two numbers' digit strings."""
    a, b = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def both_orders(base: dict, right: str, wrong: str) -> list[dict]:
    """The pair as two examples: right answer as option A, then as option B."""
    return [{**base, "a": right, "b": wrong, "answer": "A"}, {**base, "a": wrong, "b": right, "answer": "B"}]


def fermat_pairs(idx: int, pert: str, orig: str, split: str, rng: random.Random) -> list[dict]:
    """Pairs from one FERMAT row. 'silent': written wrong value vs the correct one; 'lookalike': unchanged numbers."""
    p, o = tokens_with_offsets(pert), tokens_with_offsets(orig)
    sm = SequenceMatcher(None, [t for t, _, _ in o], [t for t, _, _ in p], autojunk=False)
    out, kept = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            kept += [p[j] for j in range(j1, j2) if is_num(p[j][0])]
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for (ot, _, _), (pt, s, e) in zip(o[i1:i2], p[j1:j2]):
                if is_num(ot) and is_num(pt) and not same_value(ot, pt):
                    base = {"source": "fermat", "kind": "silent", "split": split, "fermat_idx": idx,
                            "line": masked_line(pert, s, e), "written": pt}
                    out += both_orders(base, pt, ot)
    rng.shuffle(kept)
    n = 0
    for t, s, e in kept:
        if n >= LOOKALIKE_PER_ROW:
            break
        c = lookalike(t, rng)
        if c:
            base = {"source": "fermat", "kind": "lookalike", "split": split, "fermat_idx": idx,
                    "line": masked_line(pert, s, e), "written": t}
            out += both_orders(base, t, c)
            n += 1
    return out


def edu_pairs(rows: list[dict], answer: str, image: str, split: str, rng: random.Random) -> list[dict]:
    """Pairs from one EDU page. rows: labels.csv rows (numbers, not 'ins') with start/end/raw/misread/gold_raw."""
    out, kept = [], []
    for r in rows:
        s, e, x = int(r["start"]), int(r["end"]), str(r["raw"])
        base = {"source": "edu", "split": split, "student": r["student"], "page": r["page"], "image": image,
                "line": masked_line(answer, s, e)}
        if int(r["misread"]) == 1:
            g = str(r.get("gold_raw") or "")
            if g and not same_value(g, x) and digit_edit(g, x) <= 1:
                out += both_orders({**base, "kind": "misread", "written": g}, g, x)
        else:
            kept.append((base, x))
    rng.shuffle(kept)
    n_mis = sum(1 for q in out if q["kind"] == "misread") // 2
    for base, x in kept[: max(LOOKALIKE_PER_ROW, n_mis)]:  # about as many look-alike pairs as misread pairs
        c = lookalike(x, rng)
        if c:
            out += both_orders({**base, "kind": "lookalike", "written": x}, x, c)
    return out


def pick_train_students(labels, n: int) -> list[str]:
    """The n students with the most near-miss misreads (most training signal); ties by name. Deterministic."""
    if n <= 0:
        return []
    mis = labels[(labels["type"] == "num") & (labels["misread"] == 1)]
    near = mis[[digit_edit(str(g), str(x)) <= 1 for g, x in zip(mis["gold_raw"].fillna(""), mis["raw"])]]
    counts = near.groupby("student").size().reindex(sorted(labels["student"].unique()), fill_value=0)
    return sorted(counts.sort_values(ascending=False, kind="stable").index[:n].tolist())


def write(out: Path, name: str, pairs: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"pairs_{name}.jsonl", "w", encoding="utf-8") as f:
        for q in pairs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def update_summary(out: Path, key: str, value: dict) -> None:
    path = out / "pairs_summary.json"
    s = json.loads(path.read_text()) if path.exists() else {}
    s[key] = value
    path.write_text(json.dumps(s, indent=2))


def count(pairs: list[dict]) -> dict:
    c: dict[str, int] = {}
    for q in pairs:
        c[q["kind"]] = c.get(q["kind"], 0) + 1
    return c


def cmd_fermat(a) -> None:
    from fermat_data import load_fermat
    ds = load_fermat()
    rng = random.Random(0)
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for i, r in enumerate(ds.remove_columns([c for c in ds.column_names if c == "image"])):
        by_split[r["split"]] += fermat_pairs(i, r["pert_a"] or "", r["orig_a"] or "", r["split"], rng)
    out = Path(a.out)
    for split, pairs in by_split.items():
        write(out, f"fermat_{split}", pairs)
    update_summary(out, "fermat", {s: count(p) for s, p in by_split.items()})
    print(json.dumps({s: count(p) for s, p in by_split.items()}, indent=2))


def cmd_edu(a) -> None:
    import pandas as pd
    from labels import page_files, page_text
    from score_tokens import page_index
    labels = pd.read_csv(a.labels, dtype={"gold": str, "gold_raw": str, "raw": str, "student": str})
    labels = labels[(labels["type"] == "num") & (labels["op"] != "ins")]
    train_students = a.students.split(",") if a.students else pick_train_students(labels, a.train_students)
    idx = page_index(Path(a.images))
    rng = random.Random(0)
    by_split: dict[str, list[dict]] = {"train": [], "test": []}
    missing = 0
    for pid, g in labels.groupby("page", sort=True):
        hw, st, stem = pid.split("|")
        qid = re.sub(r"_[1-9]$", "", stem)
        md = page_files(Path(a.target_root), hw, st, qid, a.model_dir).get(stem)
        img = idx.get((hw, st, stem))
        if md is None or img is None:
            missing += 1
            continue
        split = "train" if st in train_students else "test"
        by_split[split] += edu_pairs(g.to_dict("records"), page_text(md), str(img), split, rng)
    out = Path(a.out)
    for split, pairs in by_split.items():
        write(out, f"edu_{split}", pairs)
    summary = {"train_students": train_students, "pages_missing": missing,
               **{s: count(p) for s, p in by_split.items()}}
    update_summary(out, "edu", summary)
    print(json.dumps(summary, indent=2))


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(ROOT / "src" / "eval"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fermat")
    f.add_argument("--out", default="results/checker/pairs")
    e = sub.add_parser("edu")
    e.add_argument("--labels", required=True, help="labels.csv from src/detect/labels.py")
    e.add_argument("--target-root", required=True)
    e.add_argument("--model-dir", required=True)
    e.add_argument("--images", required=True, help="folder holding Screenshot_output_anon")
    e.add_argument("--train-students", type=int, default=3, help="students used for training (0 = none)")
    e.add_argument("--students", default="", help="explicit comma-separated training students (overrides the count)")
    e.add_argument("--out", default="results/checker/pairs")
    a = p.parse_args()
    {"fermat": cmd_fermat, "edu": cmd_edu}[a.cmd](a)


if __name__ == "__main__":
    main()
