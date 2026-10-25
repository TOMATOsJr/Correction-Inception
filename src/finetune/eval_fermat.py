"""Evaluate any reader on FERMAT: silent-correction (error-reversion) rate + token error rate vs pert_a.

  python eval_fermat.py --model hf:Qwen/Qwen2.5-VL-3B-Instruct --split test --name qwen3b_base      # GPU
  python eval_fermat.py --model hf:Qwen/Qwen2.5-VL-3B-Instruct --adapter reader_lora --split test --name qwen3b_ft
  python eval_fermat.py --model gemini-3.5-flash --split test --name gemini35flash --limit 100         # API
Outputs: <out>/<name>/predictions.jsonl (resumable) and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from fermat_data import PROMPT, cer, load_fermat, reversion  # noqa: E402


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--out", default=str(ROOT / "results" / "fermat"))
    a = p.parse_args()

    ds = load_fermat()
    ds = ds.filter(lambda r: r["split"] == a.split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    out_dir = Path(a.out) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["idx"]] = r

    if a.model.startswith("hf:"):
        from run_ocr import HFReader
        reader = HFReader(a.model[3:], a.adapter, a.max_pixels, 2048)
        read = lambda img: reader(PROMPT, img)  # noqa: E731
    else:
        from llm import Cache, call_llm
        cache = Cache(ROOT / "results" / "llm_cache.sqlite")
        read = lambda img: call_llm(PROMPT, a.model, images=[img], temperature=0.0, cache=cache)  # noqa: E731

    from tqdm import tqdm
    with open(pred_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(tqdm(ds, desc=a.name)):
            if i in done:
                continue
            out = read(r["image"].convert("RGB"))
            rec = {"idx": i, "orig_q": r["orig_q"], "has_error": truthy(r.get("has_error")),
                   "handwriting_style": r.get("handwriting_style"), "output": out,
                   "pert_a": r["pert_a"], "orig_a": r["orig_a"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[i] = rec

    recs = [r for r in done.values() if not str(r["output"]).startswith("Error:")]
    rev = [reversion(r["output"], r["pert_a"], r["orig_a"]) for r in recs if r["has_error"]]
    ev = [x for x in rev if x is not None]
    by_style = {}
    for r in recs:
        if r["has_error"]:
            v = reversion(r["output"], r["pert_a"], r["orig_a"])
            if v is not None:
                by_style.setdefault(str(r["handwriting_style"]), []).append(v)
    m = {
        "name": a.name, "model": a.model, "adapter": a.adapter, "split": a.split,
        "n": len(recs), "n_api_errors": len(done) - len(recs),
        "n_erroneous_evaluable": len(ev),
        "silent_correction_rate_percent": round(100 * sum(ev) / len(ev), 2) if ev else None,
        "token_error_rate_vs_written_percent": round(100 * sum(cer(r["output"], r["pert_a"]) for r in recs) / max(1, len(recs)), 2),
        "silent_correction_by_handwriting_style": {k: [round(100 * sum(v) / len(v), 1), len(v)] for k, v in by_style.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
