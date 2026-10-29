"""Forced-choice verification of flagged numbers: 'is this number written as A) 72 or B) 78?' (Phase 3)

For the k most suspicious numbers per page (score_tokens.py, by neg_gap) we build a short candidate list from
APPEARANCE only, never from the maths or the rubric (that would invite a silent correction):
  1. the reader's top other digits at the number's weakest digit ('alts' column of scores.csv)
  2. look-alike digit swaps (CONFUSE), one digit at a time
Each candidate c is put against the transcript's reading x as a two-option question, asked twice with the options
swapped (VLMs favour the first option). The answer is read from the probabilities of the tokens 'A' and 'B' after
the question (one forward pass), not from generated text. Per candidate:
  p_c = mean over both orders of P(c) / (P(A) + P(B));  consistent = both orders pick the same winner
The best challenger is the candidate with the highest p_c. Decision:
  propose x -> c   if consistent and p_c > 0.5;   otherwise keep x (an inconsistent pair is a TA case).

The question shows the number's line with the number masked, so the reader looks for it on the page without
being handed a full worked solution to 'repair'. There are no word boxes, so the whole page image is used (no crop).

Outputs verify.csv (one row per flagged number) and verify_metrics.json:
  candidate_recall   flagged misreads whose gold value is in the candidate list (upper bound for any fix)
  fix_rate           flagged misreads where the proposal equals gold
  wrong_fix_rate     flagged misreads changed to a value that is still wrong
  false_change_rate  flagged correct numbers the verifier would change
  auroc_p_change     best p_c as a misread detector over the flagged numbers (compare: neg_gap on the same rows)

  python verify_choice.py --dir results/detect/qwen3b_ft_ftreader --target-root R --model-dir qwen3b_ft \\
      --adapter results/pc_run/results/reader_lora --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detection import _num, auroc  # noqa: E402
from labels import page_text  # noqa: E402

CONFUSE = {"0": "68", "1": "72", "2": "7", "3": "85", "4": "9", "5": "63", "6": "05", "7": "12", "8": "30", "9": "4"}
QUESTION = ("Look only at the student's handwriting in the image. In this line of their work:\n\n{line}\n\n"
            "the number marked [?] is written either as\nA) {a}\nB) {b}\n"
            "Which one did the student actually write, even if it is mathematically wrong? Answer with A or B.")


def candidates(text: str, alts, max_cands: int) -> list[str]:
    """Reader alternatives first, then single look-alike digit swaps; distinct values, never the reading itself."""
    out, seen = [], {_num(text)}
    pool = [c for c in ("" if pd.isna(alts) else str(alts)).split("|") if c]
    pool += [text[:j] + d + text[j + 1:] for j, ch in enumerate(text) if ch in CONFUSE for d in CONFUSE[ch]]
    for c in pool:
        v = _num(c)
        if v is not None and v not in seen:
            seen.add(v)
            out.append(c)
    return out[:max_cands]


def masked_line(answer: str, start: int, end: int, width: int = 80) -> str:
    """The number's line (clipped to ~width chars each side) with the number replaced by [?]."""
    lo = answer.rfind("\n", 0, start) + 1
    hi = answer.find("\n", end)
    hi = len(answer) if hi < 0 else hi
    lo, hi = max(lo, start - width), min(hi, end + width)
    return answer[lo:start] + "[?]" + answer[end:hi]


class ChoiceAsker:
    """Two-option questions to the local reader; answers read from P('A') vs P('B') after the prompt."""

    def __init__(self, scorer):
        self.s = scorer
        self.ab = [scorer.tok.convert_tokens_to_ids(x) for x in ("A", "B")]

    def p_first(self, image, line: str, a: str, b: str) -> float:
        torch, proc = self.s.torch, self.s.proc
        q = QUESTION.format(line=line, a=a, b=b)
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[image], return_tensors="pt").to(self.s.model.device)
        with torch.no_grad():
            logits = self.s.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        return float(torch.softmax(logits[self.ab], -1)[0])

    def challenge(self, image, line: str, x: str, c: str) -> tuple[float, bool]:
        """p(c) averaged over both orders, and whether both orders agree on the winner."""
        p1 = 1 - self.p_first(image, line, x, c)  # c is option B
        p2 = self.p_first(image, line, c, x)      # c is option A
        return (p1 + p2) / 2, (p1 > 0.5) == (p2 > 0.5)


def decide(scored: list[tuple[str, float, bool]]) -> dict:
    """Best challenger among (candidate, p_c, consistent) and whether to propose it."""
    best, p_best, cons = "", 0.0, True
    for c, pc, ok in scored:
        if pc > p_best:
            best, p_best, cons = c, pc, ok
    return {"best": best, "p_change": round(p_best, 5), "consistent": cons, "propose": bool(cons and p_best > 0.5)}


def metrics(df: pd.DataFrame) -> dict:
    df = df.assign(propose=df["propose"].astype(bool), consistent=df["consistent"].astype(bool))
    mis, ok = df[df["misread"] == 1], df[df["misread"] == 0]

    def is_gold(r, v):
        return v is not None and v == float(r["gold"])
    gold_in = mis.apply(lambda r: any(is_gold(r, _num(c)) for c in str(r["cands"]).split("|") if c), axis=1)
    fixed = mis.apply(lambda r: r["propose"] and is_gold(r, _num(r["best"])), axis=1).astype(bool)
    rate = (lambda s: round(float(s.mean()), 4) if len(s) else None)
    y = df["misread"].values
    return {"flagged": len(df), "flagged_misreads": len(mis),
            "candidate_recall": rate(gold_in), "fix_rate": rate(fixed),
            "wrong_fix_rate": rate(mis["propose"] & ~fixed), "false_change_rate": rate(ok["propose"]),
            "inconsistent_rate": rate(~df["consistent"]),
            "auroc_p_change": round(auroc(y, df["p_change"].values.astype(float)), 4),
            "auroc_neg_gap_same_rows": round(auroc(y, df["neg_gap"].values.astype(float)), 4)}


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv and scores*.csv (same transcripts)")
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--images", default=None, help="folder holding Screenshot_output_anon (default: --edu-data)")
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reader", default="hf:Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--k", type=int, default=3, help="numbers checked per page (most suspicious by neg_gap)")
    p.add_argument("--max-cands", type=int, default=4)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard", default=None, help="i/n: every n-th page starting at i")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.images = a.images or a.edu_data
    from PIL import Image
    from score_tokens import TokenScorer, jobs

    d = Path(a.dir)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str})
    scores = pd.concat([pd.read_csv(f, dtype={"alts": str}) for f in sorted(d.glob("scores*.csv"))])
    if "alts" not in scores:
        raise SystemExit(f"{d}: scores have no 'alts' column; re-run score_tokens.py from this branch")
    df = labels.merge(scores[["page", "i", "neg_gap", "alts"]], on=["page", "i"])
    df = df[(df["type"] == "num") & (df["op"] != "ins")]
    flagged = df.sort_values("neg_gap", ascending=False, kind="stable").groupby("page", sort=False).head(a.k)
    by_page = {pg: g for pg, g in flagged.groupby("page")}

    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    out_path = d / f"verify{suffix}.csv"
    done = set(pd.read_csv(out_path)["page"]) if out_path.exists() else set()
    a.limit, limit = None, a.limit  # limit counts verified pages, not all pages
    todo = [j for j in jobs(a) if j[0] in by_page and j[0] not in done][:limit]
    print(f"{len(todo)} pages to verify ({len(done)} done before), k={a.k}")
    asker = ChoiceAsker(TokenScorer(a.reader.split(":", 1)[-1], a.adapter, a.max_pixels, a.device))
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)
    for pid, md, img in tqdm(todo, desc="verifying"):
        if deadline and time.time() > deadline:
            print("time budget reached: stopping (re-run to continue)", file=sys.stderr)
            break
        answer, image, rows = page_text(md), Image.open(img).convert("RGB"), []
        for _, r in by_page[pid].iterrows():
            s, e = int(r["start"]), int(r["end"])
            x = answer[s:e]
            cands = candidates(x, r["alts"], a.max_cands)
            line = masked_line(answer, s, e)
            scored = [(c, *asker.challenge(image, line, x, c)) for c in cands]
            rows.append({"page": pid, "student": r["student"], "i": r["i"], "text": x, "gold": r["gold"],
                         "misread": r["misread"], "neg_gap": r["neg_gap"], "cands": "|".join(cands),
                         **decide(scored)})
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    res = pd.concat([pd.read_csv(f, dtype={"gold": str}) for f in sorted(d.glob("verify*.csv"))])
    m = metrics(res)
    (d / "verify_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
