"""Does the checker pick what is WRITTEN? Accuracy of the A/B question on held-out pairs (ab_data.py), per kind.

For each pair (both option orders): p_right = mean over the two orders of P(right letter) / (P(A) + P(B)).
  accuracy      share of pairs with p_right > 0.5
  both_orders   share of pairs answered right in BOTH orders (what verify_choice calls 'consistent' + right)
  mean_p        mean p_right

What each kind measures:
  edu/misread     the transcript misread a number and the true value is offered: does the checker pick the truth?
                  (untuned fine-tuned reader: 45%; a fix can only be net positive well above 50%)
  edu/lookalike   the transcript is right and a look-alike is offered: does the checker keep it? (1 - false change)
  fermat/silent   the student's wrong value vs the correct one: does it keep the wrong, written one? (silent correction)
  fermat/lookalike

Test pairs: FERMAT test split (problems never trained on), EDU students not used for training.

  python eval_checker.py --pairs results/checker/pairs --tag checker_fe --adapter results/checker/checker_fe
  python eval_checker.py --pairs results/checker/pairs --tag base                      # untuned model
Writes results/checker/eval/<tag>/{checker_metrics.json, pairs.csv} and updates results/checker/eval/summary.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "detect"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from choice_prompt import ab_token_ids, encode  # noqa: E402
from train_checker import ImageSource  # noqa: E402


def pair_groups(pairs: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for q in pairs:
        key = (q["source"], q["kind"], q.get("fermat_idx"), q.get("page"), q["line"], q["written"], frozenset((q["a"], q["b"])))
        groups.setdefault(key, []).append(q)
    return [g for g in groups.values() if len(g) == 2]


def sample(groups: list[list[dict]], per_kind: int, seed: int = 0) -> list[list[dict]]:
    """At most per_kind pairs per source/kind, same sample for every model (fixed seed, sorted input)."""
    rng = random.Random(seed)
    by: dict[str, list] = {}
    for g in sorted(groups, key=lambda g: json.dumps(g[0], sort_keys=True)):
        by.setdefault(f"{g[0]['source']}/{g[0]['kind']}", []).append(g)
    out = []
    for k in sorted(by):
        v = by[k]
        rng.shuffle(v)
        out += v[:per_kind] if per_kind else v
    return out


def summarize(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    res = {}
    for k, v in sorted(by.items()):
        n = len(v)
        res[k] = {"pairs": n, "accuracy": round(sum(r["p_right"] > 0.5 for r in v) / n, 4),
                  "both_orders": round(sum(r["both_right"] for r in v) / n, 4),
                  "mean_p": round(sum(r["p_right"] for r in v) / n, 4)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="results/checker/pairs")
    p.add_argument("--sources", default="fermat,edu")
    p.add_argument("--tag", required=True, help="name of this model in the results (base, reader, checker_f, ...)")
    p.add_argument("--reader", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--per-kind", type=int, default=400, help="pairs per source/kind (0 = all)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default="results/checker/eval")
    a = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import pandas as pd
    import torch
    from run_ocr import HFReader
    from tqdm import tqdm

    sources = [s for s in a.sources.split(",") if s]
    pairs = []
    for s in sources:
        f = Path(a.pairs) / f"pairs_{s}_test.jsonl"
        if f.exists():
            pairs += [json.loads(line) for line in f.open(encoding="utf-8")]
        else:
            print(f"warning: {f} missing, skipped", file=sys.stderr)
    groups = sample(pair_groups(pairs), a.per_kind)
    print(f"{a.tag}: {len(groups)} test pairs")

    r = HFReader(a.reader, a.adapter, a.max_pixels, max_new_tokens=1, device=a.device)  # as the verifier loads it
    ab = ab_token_ids(r.proc.tokenizer)
    images = ImageSource("fermat" in sources)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)
    for g in tqdm(groups, desc=f"eval {a.tag}"):
        if deadline and time.time() > deadline:
            print("time budget reached: partial results", file=sys.stderr)
            break
        img = images(g[0])
        ps = []
        for q in g:
            enc = encode(r.proc, img, q["line"], q["a"], q["b"]).to(r.model.device)
            with torch.no_grad():
                logits = r.model(**enc, logits_to_keep=1).logits[0, -1].float()
            pa = float(torch.softmax(logits[ab], -1)[0])
            ps.append(pa if q["answer"] == "A" else 1 - pa)
        q = g[0]
        rows.append({"group": f"{q['source']}/{q['kind']}", "student": q.get("student", ""),
                     "p_right": sum(ps) / 2, "both_right": all(x > 0.5 for x in ps), "p_order1": ps[0], "p_order2": ps[1]})
    pd.DataFrame(rows).to_csv(out / "pairs.csv", index=False)  # no student text: probabilities only
    m = {"tag": a.tag, "adapter": a.adapter, "seconds": round(time.time() - t0), "by_kind": summarize(rows)}
    (out / "checker_metrics.json").write_text(json.dumps(m, indent=2))
    sp = Path(a.out) / "summary.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s[a.tag] = m["by_kind"]
    sp.write_text(json.dumps(s, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
