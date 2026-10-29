"""Teacher-forced token confidence: how much does our local reader believe each maths token of ANY transcript? (Phase 1)

The page image + transcription prompt go in as usual, but instead of generating, the transcript under test is fed
as the answer and one forward pass gives the reader's probability for every token it contains. Works for
transcripts we did not produce (Gemini / GPT-5.1 released ones have no logits).

Per answer token (from the full next-token distribution):
  lp     log p(token actually in the transcript)
  gap    p(token) - p(best OTHER token)          (< 0: the reader would have written something else)
  ent    entropy of the distribution
  dent   for digit tokens: entropy over the 10 digit tokens only (3-vs-8 confusion, ignores formatting mass)
Per maths token (labels.math_tokens, same indices as labels.csv), aggregated over its pieces (weakest piece wins):
  nll = max -lp,  neg_gap = max -gap,  ent = max ent,  dent = max dent
  alt  = the token with its weakest piece replaced by the reader's preferred piece (a ready-made 'proposed' reading)

  python score_tokens.py --target-root R --model-dir qwen3b_ft --out results/detect/qwen3b_ft \\
      --reader hf:Qwen/Qwen2.5-VL-3B-Instruct --adapter results/pc_run/results/reader_lora
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from labels import GOLD_MODEL_DIR, ITEMS, math_tokens, page_files, page_text  # noqa: E402
from evaluate_recognition import RECTIFIED  # noqa: E402
from run_ocr import DEFAULT_EDU_REPO, DEFAULT_PROMPT, sanitize  # noqa: E402

FIELDS = ["page", "i", "text", "type", "nll", "neg_gap", "ent", "dent", "alt", "n_pieces"]


def page_index(images_root: Path) -> dict:
    """(hw, st, page stem) -> page image, same naming as run_ocr."""
    return {(img.parent.parent.name, img.parent.name, sanitize(img.name)): img
            for img in (images_root / "Screenshot_output_anon").glob("Homework*/student_*/*.png")}


class TokenScorer:
    def __init__(self, model_id: str, adapter: str | None, max_pixels: int, device: int = 0):
        import torch
        from run_ocr import HFReader

        self.torch = torch
        r = HFReader(model_id, adapter, max_pixels, max_new_tokens=1, device=device)
        self.model, self.proc = r.model, r.proc
        tok = self.proc.tokenizer
        self.tok = tok
        self.digit_ids = torch.tensor(tok.convert_tokens_to_ids(list("0123456789")), device=self.model.device)

    def token_features(self, prompt: str, image, answer: str, max_answer_tokens: int) -> tuple[list[dict], bool]:
        """Per answer-token features + char offsets in `answer`; second value: answer was truncated."""
        torch = self.torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt")
        enc = self.tok(answer, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"][:max_answer_tokens], enc["offset_mapping"][:max_answer_tokens]
        truncated = len(enc["input_ids"]) > max_answer_tokens
        n = len(ids)
        if n == 0:
            return [], truncated
        ans = torch.tensor([ids])
        shape = inputs["input_ids"].shape
        for k, v in list(inputs.items()):  # every per-position tensor grows (attention mask, mm_token_type_ids, ...)
            if k != "input_ids" and torch.is_tensor(v) and v.shape == shape:
                fill = torch.ones_like(ans) if k == "attention_mask" else torch.zeros_like(ans)  # answer = text
                inputs[k] = torch.cat([v, fill.to(v.dtype)], 1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ans], 1)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            # keep logits only for the answer: position P-1+k predicts answer token k
            logits = self.model(**inputs, logits_to_keep=n + 1).logits[0, :n]
        target = inputs["input_ids"][0, -n:]
        rows = []
        for s in range(0, n, 256):  # chunked: a full float32 vocab row per token is ~0.6 MB
            lp_all = torch.log_softmax(logits[s:s + 256].float(), -1)
            tgt = target[s:s + 256]
            lp = lp_all.gather(1, tgt[:, None])[:, 0]
            p_all = lp_all.exp()
            top2 = p_all.topk(2, -1)
            first_is_tgt = top2.indices[:, 0] == tgt
            other_p = torch.where(first_is_tgt, top2.values[:, 1], top2.values[:, 0])
            other_id = torch.where(first_is_tgt, top2.indices[:, 1], top2.indices[:, 0])
            ent = -(p_all * lp_all).sum(-1)
            dig = lp_all[:, self.digit_ids]
            dig = torch.log_softmax(dig, -1)
            dent = -(dig.exp() * dig).sum(-1)
            is_digit = (tgt[:, None] == self.digit_ids[None]).any(-1)
            for k in range(len(tgt)):
                rows.append({"start": offs[s + k][0], "end": offs[s + k][1], "lp": float(lp[k]),
                             "gap": float(lp[k].exp() - other_p[k]), "ent": float(ent[k]),
                             "dent": float(dent[k]) if bool(is_digit[k]) else 0.0,
                             "alt_piece": self.tok.decode([int(other_id[k])])})
        return rows, truncated


def aggregate(answer: str, pieces: list[dict]) -> list[dict]:
    """Per maths token: features of its weakest model piece (pieces overlapping its character span)."""
    out = []
    for t in math_tokens(answer):
        ps = [p for p in pieces if p["start"] < t["end"] and p["end"] > t["start"]]
        if not ps:  # past the truncation point
            continue
        weak = min(ps, key=lambda p: p["gap"])
        alt = answer[t["start"]:weak["start"]] + weak["alt_piece"] + answer[weak["end"]:t["end"]] \
            if weak["start"] >= t["start"] and weak["end"] <= t["end"] else ""
        out.append({"i": t["i"], "text": t["raw"], "type": t["type"], "nll": round(max(-p["lp"] for p in ps), 5),
                    "neg_gap": round(-weak["gap"], 5), "ent": round(max(p["ent"] for p in ps), 5),
                    "dent": round(max(p["dent"] for p in ps), 5), "alt": alt.strip(), "n_pieces": len(ps)})
    return out


def jobs(a) -> list[tuple[str, Path, Path]]:
    """(page id, transcript page file, page image) for every expert-labelled page with a matching image."""
    idx = page_index(Path(a.images))
    items = pd.read_csv(ITEMS, dtype=str)[["Homework ID", "Student ID", "Question ID"]].values.tolist()
    out = []
    for hw, st, qid in items:
        gold = page_files(Path(a.edu_data) / RECTIFIED, hw, st, qid, GOLD_MODEL_DIR)
        tgt = page_files(Path(a.target_root), hw, st, qid, a.model_dir)
        if not tgt or set(gold) != set(tgt):
            continue
        for stem, path in tgt.items():
            img = idx.get((hw, st, stem))
            if img is not None:
                out.append((f"{hw}|{st}|{stem}", path, img))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        out = out[i::n]
    return out[: a.limit] if a.limit else out


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--images", default=None, help="folder holding Screenshot_output_anon (default: --edu-data)")
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reader", default="hf:Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--prompt-file", default=None, help="prompt the transcript was produced with (default: paper v6)")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--max-answer-tokens", type=int, default=3072, help="looping transcripts are cut here")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard", default=None, help="i/n: every n-th page starting at i (one process per GPU)")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.images = a.images or a.edu_data
    from PIL import Image

    prompt = Path(a.prompt_file or Path(DEFAULT_EDU_REPO) / DEFAULT_PROMPT).read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{problem_statement}", "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f".{a.shard.replace('/', 'of')}" if a.shard else ""
    path, meta_path = out / f"scores{suffix}.csv", out / f"score_pages{suffix}.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(line)["page"] for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [j for j in jobs(a) if j[0] not in done]
    print(f"{len(todo)} pages to score ({len(done)} done before)")
    scorer = TokenScorer(a.reader.split(":", 1)[-1], a.adapter, a.max_pixels, a.device)
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f, open(meta_path, "a", encoding="utf-8") as fm:
        w = csv.DictWriter(f, FIELDS)
        if new_file:
            w.writeheader()
        for pid, md, img in tqdm(todo, desc="scoring"):
            if deadline and time.time() > deadline:
                print("time budget reached: stopping (re-run to continue)", file=sys.stderr)
                break
            t0, answer = time.time(), page_text(md)
            pieces, truncated = scorer.token_features(prompt, Image.open(img).convert("RGB"), answer,
                                                      a.max_answer_tokens)
            rows = aggregate(answer, pieces)
            w.writerows({"page": pid, **r} for r in rows)
            f.flush()
            fm.write(json.dumps({"page": pid, "n_answer_tokens": len(pieces), "truncated": truncated,
                                 "n_math_tokens": len(rows), "seconds": round(time.time() - t0, 2)}) + "\n")
            fm.flush()


if __name__ == "__main__":
    main()
