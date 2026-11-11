"""Top-k whole-number candidates for doubtful numbers: beam search by the reader, from the page (Phase 3a)

The A/B checker can only fix a misread if the true value is one of the options. The old candidates (the reader's top
digits at ONE position, plus look-alike digit swaps) contain it for only ~34% of misreads: they can never propose
5.9 for 59, 12 for 123, or 540 for 54. Here the reader reads the page up to the number (the transcript so far is fed
in, teacher-forced) and then GENERATES the number with a digit-by-digit beam search: its k most probable complete numbers, which may
differ in several digits, the decimal point or the length.

Appearance only, as before: the reader sees the page and the transcript up to the number, never the rubric and never
the lines after it, so "what makes the later maths work" cannot leak in.

Which numbers: the same as verify_choice.py (--select topk --k N: most doubtful by neg_gap per page, or the selector).

Output (in --dir): beams[.<shard>].csv  page, i, written, beams ('|'-joined distinct values, best first, never the
written value), beam_p (their sequence probabilities). verify_choice.py --beams uses them as the first candidates.

  python beam_candidates.py --dir results/checker/correct_x --target-root R --model-dir qwen3b_ft \\
      --adapter local_results/results/reader_lora --select topk --k 10 --beams 6 --keep 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_detection import _num  # noqa: E402
from labels import page_text  # noqa: E402

FIELDS = ["page", "i", "written", "beams", "beam_p"]
_LEAD = re.compile(r"^\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)")


def leading_number(text: str) -> str | None:
    """The number a continuation starts with ('5.9 V = ...' -> '5.9'), or None."""
    m = _LEAD.match(text or "")
    return m.group(1) if m else None


def distinct_candidates(continuations: list[tuple[str, float]], written: str, k: int) -> list[tuple[str, float]]:
    """Leading numbers of the beams, best first, distinct by value, never the written value; probabilities of
    beams that land on the same value are added up."""
    seen: dict[float, int] = {}
    out: list[list] = []
    wv = _num(written)
    for text, p in continuations:
        c = leading_number(text)
        v = _num(c) if c else None
        if v is None or v == wv:
            continue
        if v in seen:
            out[seen[v]][1] += p
        else:
            seen[v] = len(out)
            out.append([c, p])
    return [(c, p) for c, p in out[:k]]


NUM_CHARS = list("0123456789.,")


def expand(beams: list[tuple[str, float]], allowed_logp: list[list[float]], end_logp: list[float],
           width: int) -> tuple[list[tuple[float, int, str, int]], list[tuple[str, float]]]:
    """One step of the number beam search, on plain lists (no torch).

    beams[b] = (text so far, log-prob); allowed_logp[b][j] = log P(NUM_CHARS[j] next); end_logp[b] = log P(the number
    ends here) = log(1 - P(any digit, '.' or ',')). Returns the `width` best extensions as (score, beam, text, char
    index) and the numbers that end here as (text, score). A number starts with a digit or '.', has one '.' at most,
    and ends on a digit."""
    ext, done = [], []
    for b, (txt, sc) in enumerate(beams):
        if txt and txt[-1].isdigit():
            done.append((txt, sc + end_logp[b]))
        for j, ch in enumerate(NUM_CHARS):
            if ch == "," and (not txt or not txt[-1].isdigit() or "." in txt):
                continue
            if ch == "." and ("." in txt or (txt and not txt[-1].isdigit())):
                continue
            if ch.isdigit() is False and txt and txt[-1] in ".,":
                continue
            ext.append((sc + allowed_logp[b][j], b, txt + ch, j))
    ext.sort(key=lambda x: -x[0])
    return ext[:width], done


class BeamReader:
    """Top-k complete numbers at a position, read from the page.

    The page and the transcript up to the number are read ONCE (one prefill pass at batch 1, like a scoring pass).
    Then a digit-by-digit beam search runs on the cached keys/values, expanded to the beams: each beam may only add a
    digit, '.' or ',', or end the number. HF generate() with beams copied the page image and the whole prefill per
    beam and ran an 11 GB GPU out of memory."""

    def __init__(self, scorer, prompt: str, beams: int, max_new_tokens: int, max_prefix_tokens: int):
        self.s, self.prompt, self.k = scorer, prompt, beams
        self.max_new, self.max_prefix = max_new_tokens, max_prefix_tokens
        self.char_ids = scorer.torch.tensor(scorer.tok.convert_tokens_to_ids(NUM_CHARS), device=scorer.model.device)

    def window(self, prefix: str) -> list[int]:
        """Token ids of the last ~max_prefix tokens of the transcript before the number, starting at a line start
        when one is near (the whole page stays in view as the image)."""
        enc = self.s.tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if len(ids) <= self.max_prefix:
            return ids
        cut = offs[len(ids) - self.max_prefix][0]
        nl = prefix.find("\n", cut)
        if 0 <= nl < cut + 300:
            cut = nl + 1
        return self.s.tok(prefix[cut:], add_special_tokens=False)["input_ids"]

    def _step_logp(self, logits):
        """Per beam: log P of each NUM_CHARS token, and log P(the number ends) = log(1 - their total)."""
        torch = self.s.torch
        lp = torch.log_softmax(logits.float(), -1)
        allowed = lp[:, self.char_ids]
        end = torch.log1p(-allowed.exp().sum(-1).clamp(max=1 - 1e-6))
        return allowed.tolist(), end.tolist()

    def continuations(self, image, prefix: str) -> list[tuple[str, float]]:
        """Up to k complete numbers at the position after `prefix`, best first, with probabilities."""
        torch, proc = self.s.torch, self.s.proc
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": self.prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[image], return_tensors="pt")
        ids = self.window(prefix)
        if ids:
            ans = torch.tensor([ids])
            shape = inputs["input_ids"].shape
            for key, v in list(inputs.items()):  # as score_tokens.py: every per-position tensor grows
                if key != "input_ids" and torch.is_tensor(v) and v.shape == shape:
                    fill = torch.ones_like(ans) if key == "attention_mask" else torch.zeros_like(ans)
                    inputs[key] = torch.cat([v, fill.to(v.dtype)], 1)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ans], 1)
        inputs = inputs.to(self.s.model.device)
        with torch.no_grad():
            out = self.s.model(**inputs, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            allowed, end = self._step_logp(out.logits[:, -1])
            beams: list[tuple[str, float]] = [("", 0.0)]
            finished: list[tuple[str, float]] = []
            for _ in range(self.max_new):
                keep, done = expand(beams, allowed, end, self.k)
                finished += done
                finished.sort(key=lambda x: -x[1])
                # scores only fall: stop once no open beam can beat the k-th finished number
                if not keep or (len(finished) >= self.k and keep[0][0] < finished[self.k - 1][1]):
                    beams = []
                    break
                cache.reorder_cache(torch.tensor([b for _, b, _, _ in keep], device=self.s.model.device))
                step = self.char_ids[torch.tensor([j for _, _, _, j in keep], device=self.s.model.device)][:, None]
                out = self.s.model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                allowed, end = self._step_logp(out.logits[:, -1])
                beams = [(t, sc) for sc, _, t, _ in keep]
            finished += [(t, sc + end[b]) for b, (t, sc) in enumerate(beams) if t and t[-1].isdigit()]
        finished.sort(key=lambda x: -x[1])
        return [(t, math.exp(sc)) for t, sc in finished]


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder with labels.csv and scores*.csv (beams.csv is written here)")
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--images", default=None)
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reader", default="hf:Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", help="the verbatim reader's LoRA (transcription), not the A/B checker")
    p.add_argument("--prompt-file", default=None, help="transcription prompt (default: paper v6)")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--select", choices=["auto", "selector", "topk"], default="topk")
    p.add_argument("--k", type=int, default=10, help="topk: numbers per page")
    p.add_argument("--beams", type=int, default=6, help="beam search width")
    p.add_argument("--keep", type=int, default=4, help="distinct candidate numbers kept per number")
    p.add_argument("--max-new-tokens", type=int, default=8, help="digits / '.' / ',' per number (single tokens)")
    p.add_argument("--max-prefix-tokens", type=int, default=2048, help="transcript context before the number")
    p.add_argument("--exclude-students", default="")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard", default=None, help="i/n: every n-th page starting at i")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.images = a.images or a.edu_data
    from PIL import Image
    from run_ocr import DEFAULT_EDU_REPO, DEFAULT_PROMPT
    from score_tokens import TokenScorer, jobs
    from verify_choice import flagged_numbers

    d = Path(a.dir)
    flagged = flagged_numbers(d, a.select, a.k)
    skip = {x for x in a.exclude_students.split(",") if x}
    if skip:
        flagged = flagged[~flagged["student"].astype(str).isin(skip)]
    by_page = {pg: g for pg, g in flagged.groupby("page")}
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    out_path = d / f"beams{suffix}.csv"
    import pandas as pd
    done = set(pd.read_csv(out_path)["page"]) if out_path.exists() else set()
    a.limit, limit = None, a.limit
    todo = [j for j in jobs(a) if j[0] in by_page and j[0] not in done][:limit]
    print(f"{len(todo)} pages to beam-search ({len(done)} done before); width {a.beams}, keep {a.keep}")
    prompt = Path(a.prompt_file or Path(DEFAULT_EDU_REPO) / DEFAULT_PROMPT).read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{problem_statement}", "")
    reader = BeamReader(TokenScorer(a.reader.split(":", 1)[-1], a.adapter, a.max_pixels, a.device), prompt,
                        a.beams, a.max_new_tokens, a.max_prefix_tokens)
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)
    new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for pid, md, img in tqdm(todo, desc="beams"):
            if deadline and time.time() > deadline:
                print("time budget reached: stopping (re-run to continue)", file=sys.stderr)
                break
            answer, image = page_text(md), Image.open(img).convert("RGB")
            rows = []
            for _, r in by_page[pid].iterrows():
                s, e = int(r["start"]), int(r["end"])
                conts = reader.continuations(image, answer[:s])
                cands = distinct_candidates(conts or [], answer[s:e], a.keep)
                rows.append({"page": pid, "i": int(r["i"]), "written": answer[s:e],
                             "beams": "|".join(c for c, _ in cands),
                             "beam_p": "|".join(f"{q:.4f}" for _, q in cands)})
            w.writerows(rows)  # one page at a time, so a restart resumes at the next page
            f.flush()
    print(json.dumps({"pages_done": len(done) + len(todo), "file": str(out_path)}))


if __name__ == "__main__":
    main()
