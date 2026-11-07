"""Correction pipeline on one local GPU (Windows or Linux); the single-machine version of run_detect.sh.

For each experiment:
  labels (CPU) -> token scores (GPU) -> detection metrics -> selector (CPU) -> forced-choice check (GPU)
  -> change / keep / TA + corrected transcripts (CPU) -> what-if sweep (CPU) -> maths diff (CPU) [-> grading, --grade]
Experiments with the same transcripts and reader share one set of scores. Everything is resumable: re-run the same
command after a crash. At the end, correction_metrics.zip holds the numbers only (no student text): safe to share.

  python scripts/run_correction.py --data C:\\lma\\data --smoke                 (5 pages, a few minutes)
  python scripts/run_correction.py --data C:\\lma\\data --experiments topk3     (cheapest full run)
  python scripts/run_correction.py --data C:\\lma\\data --experiments baseline
  python scripts/run_correction.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt"
READER = "hf:Qwen/Qwen2.5-VL-3B-Instruct"
GRADER = "hf-text:Qwen/Qwen2.5-7B-Instruct"
MDIR = {"qwen": "qwen3b_ft", "gemini": "gemini-2.5-pro"}
SHARED = ["labels.csv", "label_pages.csv", "labels_summary.json", "selector.csv", "selector_metrics.json"]

# name: transcripts, fine-tuned reader?, numbers to check, extra verify_choice.py args, what it tests
EXPERIMENTS = {
    "topk3": ("qwen", True, "topk", [], "3 most doubtful numbers per page: the cheap first answer"),
    "baseline": ("qwen", True, "auto", [], "learned selector, all guards: the planned pipeline"),
    "base_reader": ("qwen", False, "auto", [], "does the fine-tuned reader beat the untuned one?"),
    "crop": ("qwen", True, "topk", ["--crop-band", "0.3"], "zoom: only 30% of the page height around the number"),
    "more_cands": ("qwen", True, "topk", ["--max-cands", "6"], "does a longer candidate list raise candidate recall?"),
    "other": ("qwen", True, "topk", ["OTHER"], "Gemini's value as a candidate (flattering: experts corrected Gemini)"),
    "gemini": ("gemini", True, "topk", [], "same pipeline on Gemini 2.5's transcripts (few misreads)"),
}


def find(roots: list[Path], pattern: str, up: int = 1) -> Path | None:
    """Shallowest match of `pattern` under the data folder."""
    for depth in range(6):
        hits = sorted(h for r in roots for h in glob(str(r / ("*/" * depth) / pattern)))
        if hits:
            p = Path(hits[0])
            for _ in range(up):
                p = p.parent
            return p
    return None


def resolve_data(data: Path) -> dict:
    p = {"edu": find([data], "Rectified_recognized_markdown_done_Anon"),
         "images": find([data], "Screenshot_output_anon/Homework*/student_*/*.png", up=4),
         "qwen": find([data], "qwen3b_ft", up=0), "lora": find([data], "reader_lora", up=0)}
    p["gemini"] = p["edu"] / "Observationset_Final" / "v6_Gemini_2p5" if p["edu"] else None
    return p


def preflight(a, paths: dict, needs_lora: bool):
    problems = []
    for k, what in [("edu", "edu_text"), ("images", "page images"), ("qwen", "our 3B transcripts"), ("gemini", "Gemini transcripts")]:
        ok = paths[k] is not None and Path(paths[k]).is_dir()
        print(f"  {what:20s} {paths[k]}  {'ok' if ok else 'MISSING'}")
        problems += [] if ok else [what]
    if needs_lora and not a.no_adapter and paths["lora"] is None:
        problems.append("reader_lora (or pass --no-adapter)")
    if paths["edu"] and not (paths["edu"] / "Screenshot_output_anon" / "rubric_outputs").is_dir():
        problems.append("rubrics (answer-key guard)")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU                  {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
        elif not a.allow_cpu:
            problems.append("no CUDA GPU (reinstall torch with the cu128 build; --allow-cpu only for tests)")
    except ImportError:
        problems.append("torch is not installed")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2**30
    print(f"  free disk            {free:.0f} GB")
    if free < 12 and not a.allow_cpu:
        problems.append(f"only {free:.0f} GB free (the reader download alone is ~7 GB)")
    if problems:
        sys.exit("preflight failed: " + "; ".join(problems))


class Runner:
    def __init__(self, out: Path):
        self.out, self.times = out, []
        self.env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

    def step(self, name: str, script: str, *args, optional: bool = False):
        cmd = [sys.executable, script, *[str(x) for x in args]]
        print(f"\n--- {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        code = subprocess.run(cmd, cwd=ROOT, env=self.env).returncode
        self.times.append({"step": name, "seconds": round(time.time() - t0, 1), "exit": code})
        (self.out / "timing.json").write_text(json.dumps(self.times, indent=1))
        if code and not optional:
            sys.exit(f"step failed ({code}): {name}. Fix it and re-run the same command; finished work is kept.")


def scored_dir(r: Runner, a, paths, transcripts: str, ft: bool, out: Path) -> Path:
    """labels + scores + detection metrics + selector, shared by experiments on the same transcripts and reader."""
    d = out / f"{transcripts}_{'ftreader' if ft else 'basereader'}"
    common = ["--edu-data", paths["edu"], "--target-root", paths[transcripts], "--model-dir", MDIR[transcripts]]
    r.step(f"{d.name}: labels", "src/detect/labels.py", *common, "--out", d)
    lim = ["--limit", a.score_limit] if a.score_limit else []
    ad = ["--adapter", paths["lora"]] if ft else []
    r.step(f"{d.name}: token scores (GPU)", "src/detect/score_tokens.py", *common, "--images", paths["images"],
           "--prompt-file", PROMPT, "--out", d, "--reader", a.reader, "--device", 0, "--max-pixels", a.max_pixels, *ad, *lim)
    r.step(f"{d.name}: detection metrics", "src/detect/evaluate_detection.py", "--dir", d, "--boot", a.boot)
    r.step(f"{d.name}: selector", "src/detect/selector.py", "--dir", d, "--target-recall", a.target_recall)
    return d


def run_experiment(r: Runner, a, paths, name: str, out: Path, done: dict):
    transcripts, ft, select, extra, _ = EXPERIMENTS[name]
    ft = ft and not a.no_adapter
    key = (transcripts, ft)
    if key not in done:
        done[key] = scored_dir(r, a, paths, transcripts, ft, out)
    src, d = done[key], out / key[0] / name
    d.mkdir(parents=True, exist_ok=True)
    for f in SHARED + [p.name for p in src.glob("scores*.csv")]:  # verify*.csv stays per experiment
        if (src / f).exists():
            shutil.copy2(src / f, d / f)
    if select != "auto" or not (src / "selector.csv").exists():
        (d / "selector.csv").unlink(missing_ok=True)
    extra = ["--other-root", paths["gemini"], "--other-model-dir", MDIR["gemini"]] if extra == ["OTHER"] else extra
    common = ["--target-root", paths[transcripts], "--model-dir", MDIR[transcripts]]
    ad = ["--adapter", paths["lora"]] if ft else []
    lim = ["--limit", a.verify_limit] if a.verify_limit else []
    r.step(f"{name}: forced-choice check (GPU)", "src/detect/verify_choice.py", "--dir", d, "--edu-data", paths["edu"],
           "--images", paths["images"], *common, "--select", "auto", "--k", a.k, "--reader", a.reader, "--device", 0,
           "--max-pixels", a.max_pixels, *ad, *extra, *lim)
    r.step(f"{name}: decisions", "src/detect/correct.py", "--dir", d, *common,
           "--max-false-change", a.max_false_change)
    r.step(f"{name}: what-if sweep", "src/detect/sweep_correct.py", "--dir", d)
    for v in ("none", "corrected", "corrected_ta"):
        r.step(f"{name}: maths diff {v}", "src/eval/transcript_diff.py", "--data-root", paths["edu"],
               "--target-root", d / "Outputs" / v, "--target-model-dir", MDIR[transcripts],
               "--name", f"{name}_{v}", "--out", d / "transcript_diff", optional=True)
        if a.grade:
            r.step(f"{name}: grading {v} (GPU)", "src/eval/grade_impact.py", "--data-root", paths["edu"],
                   "--target-root", d / "Outputs" / v, "--target-model-dir", MDIR[transcripts], "--grader", GRADER,
                   "--name", f"correction_{name}_{v}", "--workers", 8, optional=True)


def summary(out: Path):
    rows = []
    for f in sorted(out.glob("*/*/correct_metrics.json")):
        h = json.loads(f.read_text())["heldout"]
        v = json.loads((f.parent / "verify_metrics.json").read_text())
        rows.append({"experiment": f.parent.name, **{k: h[k] for k in (
            "verified", "auto_fixed", "auto_wrong_fix", "auto_broken_correct", "ta_per_page", "misreads_before",
            "misreads_after_auto")}, "candidate_recall": v["candidate_recall"]})
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print("\n=== summary (held-out students)")
    for row in rows:
        print(row)
    keep = ("*_metrics.json", "*_metrics.csv", "labels_summary.json", "summary.json", "timing.json")
    with zipfile.ZipFile(out / "correction_metrics.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for pat in keep:
            for f in out.rglob(pat):
                z.write(f, f.relative_to(out))
    print(f"\nsend this file (numbers only, no student text): {out / 'correction_metrics.zip'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="folder holding the unzipped data (searched, the layout does not matter)")
    p.add_argument("--out", default=str(ROOT / "results" / "correction"))
    p.add_argument("--experiments", default="topk3", help="comma list, see --list")
    p.add_argument("--list", action="store_true")
    p.add_argument("--smoke", action="store_true", help="5 pages, separate output folder; checks everything runs")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--target-recall", type=float, default=0.8)
    p.add_argument("--max-false-change", type=float, default=0.01)
    p.add_argument("--grade", action="store_true", help="also grade with the 7B grader (4-bit, slow)")
    p.add_argument("--no-adapter", action="store_true", help="use the untuned reader everywhere")
    p.add_argument("--reader", default=READER)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28, help="lower this if the GPU runs out of memory")
    p.add_argument("--score-limit", type=int)
    p.add_argument("--verify-limit", type=int)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--allow-cpu", action="store_true", help="tests only: very slow with a real model")
    a = p.parse_args()
    if a.list:
        for n, e in EXPERIMENTS.items():
            print(f"{n:12s} {e[4]}")
        return
    if not a.data:
        p.error("--data is required")
    if a.smoke:
        a.out, a.score_limit, a.verify_limit, a.boot = str(Path(a.out) / "smoke"), a.score_limit or 5, a.verify_limit or 5, 50
    names = [n.strip() for n in a.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiments {unknown}; see --list")
    paths = resolve_data(Path(a.data))
    print("preflight")
    preflight(a, paths, any(EXPERIMENTS[n][1] for n in names))
    out = Path(a.out)
    r = Runner(out)
    r.step("unit tests", "src/detect/test_detect.py")
    done: dict = {}
    for n in names:
        print(f"\n===== experiment {n}: {EXPERIMENTS[n][4]}")
        run_experiment(r, a, paths, n, out, done)
    summary(out)


if __name__ == "__main__":
    main()
