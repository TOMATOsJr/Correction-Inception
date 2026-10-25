"""One-shot runner for everything that needs a GPU (Kaggle notebooks 01 + 02), on ONE machine.

  python scripts/run_all.py --data C:/lma                 # full run
  python scripts/run_all.py --data C:/lma --limit 3       # smoke test (~15 min + downloads), writes results_smoke/
  python scripts/run_all.py --data C:/lma --hours 10      # stop starting new exam pages so everything ends in ~10 h
  python scripts/run_all.py --data C:/lma --preflight     # only the checks

--data expects: <data>/edu_text/edu_data, <data>/edu_text/edu_repo, <data>/images/Screenshot_output_anon
(or pass --edu-data / --edu-repo / --images explicitly, e.g. on Kaggle).

Design (limited machine access):
  * preflight first (2-5 min): libraries, GPU, 4-bit kernels, HF token + FERMAT access, disk, data, downloads.
    Anything wrong stops the run BEFORE hours are spent.
  * most valuable results first: train -> FERMAT fine-tuned -> FERMAT base -> exams (fine-tuned, base) -> scores.
  * a failed step never kills the run: steps that depend on it are skipped, the rest continue.
  * every step is skipped if already done, so re-running the same command resumes.
  * everything is logged (results*/logs/), and results are zipped at the end -> local_results.zip.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# test-only overrides (CPU code tests with a tiny model): LMA_READER=<local dir>, LMA_GRADER=mock, LMA_ALLOW_CPU=1
READER = os.environ.get("LMA_READER", "Qwen/Qwen2.5-VL-3B-Instruct")
GRADER = os.environ.get("LMA_GRADER", "Qwen/Qwen2.5-7B-Instruct")
ITEMS = ROOT / "src" / "eval" / "items_with_expert_transcript.csv"
PROMPT = ROOT / "src" / "eval" / "prompts" / "Initial_prompt_wo_problem_statement_v6.txt"


class Run:
    def __init__(self, a):
        self.a = a
        self.out = ROOT / ("results_smoke" if a.limit else "results")
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.logs / "run_all.log", "a", encoding="utf-8")
        self.status = {}  # step -> ok / failed / skipped (dependency) / done-before
        self.t0 = time.time()

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        try:
            print(line, flush=True)
        except Exception:  # never let a console encoding problem stop the run
            print(line.encode("ascii", "replace").decode(), flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def step(self, name, done_file, cmd, needs=(), env=None):
        try:
            self._step(name, done_file, cmd, needs, env)
        except Exception as e:  # a bug in the runner itself must not stop the other steps
            self.status[name] = f"FAILED (runner error: {type(e).__name__}: {e})"
            self.say(f"FAILED {name}: runner error {type(e).__name__}: {e}")

    def _step(self, name, done_file, cmd, needs=(), env=None):
        if any(not str(self.status.get(n, "")).startswith(("ok", "done-before")) for n in needs):
            self.status[name] = "skipped (needs " + ", ".join(needs) + ")"
            self.say(f"SKIP {name}: a step it needs did not succeed")
            return
        if done_file and Path(done_file).exists():
            self.status[name] = "done-before"
            self.say(f"done before: {name}")
            return
        self.say(f"START {name}")
        t = time.time()
        log = self.logs / f"{name}.log"
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.ctime()} :: {' '.join(map(str, cmd))}\n")
            f.flush()
            p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env={**os.environ, **(env or {})},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                f.write(line)
                f.flush()
                if not self.a.quiet and ("\r" not in line or line.strip().endswith("it/s]")):
                    sys.stdout.write(line)
            p.wait()
        mins = (time.time() - t) / 60
        if p.returncode == 0 and (not done_file or Path(done_file).exists()):
            self.status[name] = "ok"
            self.say(f"OK {name} ({mins:.0f} min)")
        else:
            self.status[name] = f"FAILED (exit {p.returncode}, see {log.relative_to(ROOT)})"
            self.say(f"FAILED {name} after {mins:.0f} min -> {log}")
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            self.say("last lines:\n    " + "\n    ".join(tail))


# ------------------------------------------------------------------ preflight
def preflight(a, r: Run, need_gpu=True):
    problems, py = [], sys.executable
    r.say("PREFLIGHT")

    def check(ok, what, fix=""):
        r.say(("  ok    " if ok else "  FAIL  ") + what + ("" if ok else f"  ->  {fix}"))
        if not ok:
            problems.append(what)

    for mod in ["torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "pandas",
                "qwen_vl_utils", "dotenv", "tqdm"]:
        try:
            __import__(mod)
            check(True, f"import {mod}")
        except Exception as e:
            check(False, f"import {mod} ({type(e).__name__}: {str(e)[:120]})", "pip install -r requirements-gpu.txt")
    import torch
    gpu = torch.cuda.is_available()
    check(gpu or not need_gpu, f"CUDA GPU visible: {torch.cuda.get_device_name(0) if gpu else 'NONE'}",
          "install the CUDA 12.8 PyTorch build (docs/LOCAL_RUN.md step 5) and a recent NVIDIA driver")
    if gpu:
        free, total = torch.cuda.mem_get_info()
        check(total > 14e9, f"GPU memory {total / 1e9:.1f} GB (free {free / 1e9:.1f} GB)", "needs a 16 GB GPU")
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # what the scripts will use
        try:  # real 4-bit kernel on this GPU (bitsandbytes is the most fragile piece)
            import bitsandbytes as bnb
            lin = bnb.nn.Linear4bit(64, 64, compute_dtype=dt, quant_type="nf4").cuda()
            y = lin(torch.randn(2, 64, device="cuda", dtype=dt))
            r.say("  ok    bitsandbytes 4-bit kernel runs on this GPU" if bool(torch.isfinite(y).all())
                  else "  WARN  bitsandbytes 4-bit kernel gives non-finite values")
        except Exception as e:  # not fatal: training falls back to plain LoRA; only the 7B grader needs 4-bit
            a.no_4bit = True
            r.say(f"  WARN  bitsandbytes 4-bit kernel failed ({type(e).__name__}: {str(e)[:150]}): training will use "
                  "plain LoRA; grade-impact steps will fail and can be redone later from the saved transcripts")
        try:
            x = torch.randn(256, 256, device="cuda", dtype=dt)
            check(bool(torch.isfinite(x @ x).all()), f"{str(dt)[6:]} matmul (sm_{''.join(map(str, torch.cuda.get_device_capability()))})")
        except Exception as e:
            check(False, f"matmul ({str(e)[:150]})", "PyTorch build does not support this GPU: use the cu128 build")

    for name, path in [("edu_data", a.edu_data), ("edu_repo", a.edu_repo), ("images", Path(a.images) / "Screenshot_output_anon")]:
        check(Path(path).exists(), f"{name}: {path}", "unzip the data (docs/LOCAL_RUN.md step 4)")
    if Path(a.images).exists():
        import pandas as pd
        items = pd.read_csv(ITEMS, dtype=str)
        pngs = list((Path(a.images) / "Screenshot_output_anon").glob("Homework*/student_*/*.png"))
        check(len(pngs) > 1000, f"{len(pngs)} exam page images found (expected ~1290)", "re-extract edu_circuit_hw_images.zip")
        check(len(items) == 293, f"{len(items)} exams with expert transcripts in the item list")
    rect = Path(a.edu_data) / "Rectified_recognized_markdown_done_Anon"
    check(rect.exists() and any(rect.rglob("*_markdown.md")), "expert transcripts present", "re-extract edu_text.zip")
    cache = ROOT / "results" / "llm_cache.sqlite"
    r.say(f"  {'ok  ' if cache.exists() else 'note'}  grade cache {'found' if cache.exists() else 'missing: expert transcripts get re-graded (+~30 min)'}")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_free = shutil.disk_usage(hf_home).free / 1e9
    check(free_gb > 10, f"disk free at project: {free_gb:.0f} GB")
    check(hf_free > 35 or a.limit, f"disk free for model/data downloads ({hf_home}): {hf_free:.0f} GB", "need ~35 GB")

    try:  # token + gated FERMAT access, without downloading the data yet
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN")
        check(bool(tok) or bool(os.environ.get("LMA_FAKE_FERMAT")), "HF_TOKEN set", "put HF_TOKEN=hf_... in .env")
        if tok and not os.environ.get("LMA_FAKE_FERMAT"):
            HfApi().dataset_info("ai4bharat/FERMAT", token=tok)
            check(True, "FERMAT access with this token")
    except Exception as e:
        check(False, f"FERMAT access ({type(e).__name__}: {str(e)[:150]})",
              "accept the terms at huggingface.co/datasets/ai4bharat/FERMAT with the token's account")

    if not problems and not a.mock:  # download now (also in --preflight) so network problems show up now
        try:
            from huggingface_hub import snapshot_download
            for repo in [m for m in (READER, GRADER) if m != "mock" and not Path(m).exists()]:
                r.say(f"  downloading {repo} (once, several GB) ...")
                snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
            check(True, "models downloaded")
            if not os.environ.get("LMA_FAKE_FERMAT"):
                r.say("  downloading FERMAT (once, ~5 GB) ...")
                snapshot_download("ai4bharat/FERMAT", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
                check(True, "FERMAT downloaded")
        except Exception as e:
            check(False, f"download ({type(e).__name__}: {str(e)[:150]})", "check internet / disk, then re-run")
    return problems


# ------------------------------------------------------------------ main plan
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="folder with edu_text/ and images/ (see docstring)")
    p.add_argument("--edu-data")
    p.add_argument("--edu-repo")
    p.add_argument("--images")
    p.add_argument("--limit", type=int, default=0, help="smoke test: N samples per step, writes results_smoke/")
    p.add_argument("--hours", type=float, default=0, help="time budget; exam reading stops early to fit it")
    p.add_argument("--preflight", action="store_true", help="only run the checks")
    p.add_argument("--skip-base", action="store_true", help="skip the untuned-reader steps")
    p.add_argument("--mock", action="store_true", help="plumbing test without GPU (fake models, no training)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    if a.data:
        a.edu_data = a.edu_data or f"{a.data}/edu_text/edu_data"
        a.edu_repo = a.edu_repo or f"{a.data}/edu_text/edu_repo"
        a.images = a.images or f"{a.data}/images"
        os.environ.setdefault("HF_HOME", f"{a.data}/hf_cache")  # all downloads in one deletable folder
    if not (a.edu_data and a.edu_repo and a.images):
        p.error("give --data, or all of --edu-data --edu-repo --images")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    os.environ.update(EDU_DATA=str(a.edu_data), EDU_REPO=str(a.edu_repo), HF_BATCH="8", HF_TOKEN_BUDGET="16000",
                      PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", TOKENIZERS_PARALLELISM="false")

    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    r = Run(a)
    r.say(f"run_all.py  python={sys.executable}  out={r.out}  limit={a.limit or 'full'}  hours={a.hours or 'unlimited'}")
    problems = preflight(a, r, need_gpu=not (a.mock or os.environ.get("LMA_ALLOW_CPU")))
    if problems:
        r.say(f"PREFLIGHT FAILED ({len(problems)}): fix the lines marked FAIL above and re-run. Nothing else was started.")
        sys.exit(2)
    r.say("PREFLIGHT PASSED")
    if a.preflight:
        return

    py, out, lim = sys.executable, r.out, (["--limit", str(a.limit)] if a.limit else [])
    tag = "_smoke" if a.limit else ""
    reader = "mock" if a.mock else f"hf:{READER}"
    grader = "mock" if a.mock or GRADER == "mock" else f"hf-text:{GRADER}"
    lora = out / "reader_lora"
    deadline = r.t0 + a.hours * 3600 - 75 * 60 if a.hours else 0  # keep ~75 min for scoring + zip

    def read_exams(task, adapter, needs, share):
        """Resumable; with --hours, gets `share` of the time left before the deadline."""
        env = {}
        if deadline:
            env["LMA_DEADLINE"] = str(time.time() + max(0, deadline - time.time()) * share)
        r.step(f"read_{task}", None,
               [py, "src/eval/run_ocr.py", "--split", "observation", "--items", ITEMS, "--task", task, "--model", reader,
                "--model-dir", task, *(["--adapter", lora] if adapter else []), "--data-root", a.images,
                "--outputs-root", out / "Outputs", "--prompt-file", PROMPT, *lim], needs=needs, env=env)

    def score(task):
        r.step(f"diff_{task}", out / "transcript_diff" / task / "metrics.json",
               [py, "src/eval/transcript_diff.py", "--target-root", out / "Outputs" / task, "--target-model-dir", task,
                "--name", task, "--out", out / "transcript_diff"], needs=[f"read_{task}"])
        r.step(f"grade_{task}", ROOT / "results" / "grade_impact" / f"B1_{task}_qwen7b{tag}" / "metrics.json",
               [py, "src/eval/grade_impact.py", "--target-root", out / "Outputs" / task, "--target-model-dir", task,
                "--grader", grader, "--name", f"B1_{task}_qwen7b{tag}", "--workers", "16", *lim], needs=[f"read_{task}"])

    if not a.mock:
        # 1) the key mid-submission result first: train, then silent-correction before/after on FERMAT
        train_cmd = [py, "src/finetune/train_reader.py", "--model", READER, "--out", lora, "--epochs", "1", *lim]
        if not getattr(a, "no_4bit", False):
            r.step("train", lora / "adapter_model.safetensors", train_cmd)
        if r.status.get("train") not in ("ok", "done-before"):  # 4-bit QLoRA failed or unavailable: plain LoRA
            first = r.status.get("train", "4-bit unavailable")
            shutil.rmtree(lora, ignore_errors=True)  # never resume from a 4-bit checkpoint
            r.status.pop("train", None)
            r.step("train", lora / "adapter_model.safetensors", train_cmd, env={"LMA_NO_QUANT": "1"})
            if r.status["train"] == "ok":
                r.status["train"] = f"ok (plain-LoRA fallback; QLoRA: {first})"
        r.step("fermat_ft", out / "fermat" / "qwen3b_ft" / "metrics.json",
               [py, "src/finetune/eval_fermat.py", "--model", f"hf:{READER}", "--adapter", lora, "--split", "test",
                "--name", "qwen3b_ft", "--out", out / "fermat", *lim], needs=["train"])
        if not a.skip_base:
            r.step("fermat_base", out / "fermat" / "qwen3b_base" / "metrics.json",
                   [py, "src/finetune/eval_fermat.py", "--model", f"hf:{READER}", "--split", "test",
                    "--name", "qwen3b_base", "--out", out / "fermat", *lim])
        # 2) exams: fine-tuned reader (needed by the agents later), then the untuned one, then scores
        read_exams("qwen3b_ft", True, ["train"], 0.5)
        if not a.skip_base:
            read_exams("qwen3b_base", False, [], 1.0)
        score("qwen3b_ft")
        if not a.skip_base:
            score("qwen3b_base")
    else:
        read_exams("qwen3b_base", False, [], 1.0)
        score("qwen3b_base")

    # 3) summary + zip (transcripts as markdown, adapter without optimizer checkpoints)
    summary = {"finished": time.ctime(), "hours": round((time.time() - r.t0) / 3600, 2), "steps": r.status}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    zpath = ROOT / f"local_results{tag}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in list(out.rglob("*")) + list((ROOT / "results" / "grade_impact").rglob("*")):
            if not f.is_file() or f.suffix == ".png" or "checkpoint-" in f.as_posix() or f.suffix == ".sqlite":
                continue
            z.write(f, f.relative_to(ROOT).as_posix())
    r.say("SUMMARY\n" + json.dumps(r.status, indent=2))
    failed = [k for k, v in r.status.items() if not str(v).startswith(("ok", "done"))]
    r.say(f"{'ALL DONE' if not failed else 'FINISHED WITH PROBLEMS in: ' + ', '.join(failed)}  ->  upload {zpath}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
