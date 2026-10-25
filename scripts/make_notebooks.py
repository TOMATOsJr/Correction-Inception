"""Generate the Kaggle notebooks in notebooks/ (run: python scripts/make_notebooks.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIND = ("import os, glob, shutil, zipfile, sys\n"
        "IN = os.environ.get('KAGGLE_INPUT', '/kaggle/input')\n"
        "WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')\n"
        "def find(pattern):\n"
        "    hits = sorted(glob.glob(f'{IN}/**/{pattern}', recursive=True) + glob.glob(f'{WORK}/_in/**/{pattern}', recursive=True))\n"
        "    return hits[0] if hits else None\n"
        "# Kaggle usually unzips uploads; if it kept a .zip, extract it ourselves\n"
        "if not (find('src/eval/llm.py') and find('edu_data') and find('Screenshot_output_anon/Homework*')):\n"
        "    for z in glob.glob(f'{IN}/**/*.zip', recursive=True):\n"
        "        zipfile.ZipFile(z).extractall(f'{WORK}/_in/' + os.path.basename(z)[:-4])\n"
        "llm, EDU_DATA, EDU_REPO = find('src/eval/llm.py'), find('edu_data'), find('edu_repo')\n"
        "hw = find('Screenshot_output_anon/Homework*')\n"
        "IMAGES = os.path.dirname(os.path.dirname(hw)) if hw else None\n"
        "if not (llm and EDU_DATA and EDU_REPO and IMAGES):\n"
        "    for root, dirs, files in os.walk(IN):\n"
        "        if root.count(os.sep) - IN.count(os.sep) < 4: print(root, dirs[:5], files[:5])\n"
        "    raise SystemExit('Missing inputs: attach lma-src, edu-text AND the images dataset via Add Input. Found: '\n"
        "                     + str(dict(src=llm, edu_data=EDU_DATA, edu_repo=EDU_REPO, images=IMAGES)))\n"
        "shutil.copytree(os.path.dirname(os.path.dirname(llm)), f'{WORK}/src', dirs_exist_ok=True)\n"
        "os.environ.update(EDU_DATA=EDU_DATA, EDU_REPO=EDU_REPO)\n"
        "os.chdir(WORK)\n"
        "for rel in ['results/llm_cache.sqlite']:  # reuse grades cached by notebook 03 (attach its output as an input)\n"
        "    old = find(rel)\n"
        "    if old and not os.path.exists(f'{WORK}/{rel}'):\n"
        "        os.makedirs(os.path.dirname(f'{WORK}/{rel}'), exist_ok=True); shutil.copy(old, f'{WORK}/{rel}'); print('reusing cache', old)\n"
        "DRY = bool(os.environ.get('DRY_JUDGE'))  # local dry run: mock models, no GPU\n"
        "if not DRY:\n"
        "    from kaggle_secrets import UserSecretsClient\n"
        "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n"
        "    os.system(f'{sys.executable} -m pip install -q -U transformers accelerate peft bitsandbytes qwen-vl-utils datasets python-dotenv')\n"
        "    os.system('nvidia-smi --query-gpu=name,memory.total --format=csv')\n"
        "READER = 'mock' if DRY else 'hf:Qwen/Qwen2.5-VL-3B-Instruct'\n"
        "GRADER = 'mock' if DRY else 'hf-text:Qwen/Qwen2.5-7B-Instruct'\n"
        "os.environ.update(HF_BATCH='8', HF_TOKEN_BUDGET='16000')\n"
        "LIMIT = os.environ.get('DRY_LIMIT', '')\n"
        "ITEMS = 'src/eval/items_with_expert_transcript.csv'  # the 293 exams that have an expert transcript\n"
        "print('code:', llm, '| data:', EDU_DATA, '| images:', IMAGES)")

SETUP_MD = ("md", "## Setup\nKaggle settings: **Accelerator = GPU T4 x2**, **Internet = On**. Add-ons → Secrets → `HF_TOKEN` "
                  "(you must have accepted FERMAT's terms on Hugging Face).\n\nAdd Inputs: datasets **lma-src**, **edu-text**, "
                  "**edu-circuit-hw-images**, and (to reuse cached grades) the **output of notebook 03**.")
SETUP = [SETUP_MD, ("code", FIND)]


def score_edu(task):
    """Judge-free scoring of a reader's EDU transcripts: maths diff vs expert + grade impact (B1 metric)."""
    return [
        ("md", f"### Score `{task}` transcripts (no LLM judge: Qwen-7B failed as a judge, kappa 0.03)\n"
               "1) exact maths comparison with the expert transcript; 2) grade impact with the Qwen-7B grader "
               "(agrees with TAs 71%; expert-transcript grades are reused from notebook 03's cache)."),
        ("code", f"!python src/eval/transcript_diff.py --target-root {{WORK}}/Outputs/{task} --target-model-dir {task} --name {task}\n"
                 f"!python src/eval/grade_impact.py --target-root {{WORK}}/Outputs/{task} --target-model-dir {task} "
                 f"--grader {{GRADER}} --name B1_{task}_qwen7b --workers 16 {{LIMIT}}"),
    ]


def transcribe(task, adapter=""):
    """Two processes, one per T4, each reading half the pages. Logs: ocr_<task>_<gpu>.log. Resumable."""
    return ("code", "import subprocess, time\n"
                    f"cmd = (f'{{sys.executable}} src/eval/run_ocr.py --split observation --items {{ITEMS}} --task {task} --model {{READER}} '\n"
                    f"       f'--model-dir {task} {adapter}--data-root {{IMAGES}} --outputs-root {{WORK}}/Outputs {{LIMIT}} '\n"
                    "       '--prompt-file src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt')\n"
                    "procs = []\n"
                    "for g in (0, 1):\n"
                    f"    log = open(f'{{WORK}}/ocr_{task}_{{g}}.log', 'w')\n"
                    "    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(g)}\n"
                    "    procs.append(subprocess.Popen(cmd + f' --shard {g}/2', shell=True, env=env, stdout=log, stderr=log))\n"
                    f"done = lambda: len(glob.glob(f'{{WORK}}/Outputs/{task}/**/*_markdown.md', recursive=True))\n"
                    "t0 = time.time()\n"
                    "while any(p.poll() is None for p in procs):\n"
                    "    time.sleep(60 if not DRY else 1)\n"
                    "    m = (time.time() - t0) / 60\n"
                    "    print(f'{m:.0f} min: {done()} pages done', flush=True)\n"
                    "print('exit codes:', [p.returncode for p in procs], '| pages:', done())\n"
                    f"for g in (0, 1): print(open(f'{{WORK}}/ocr_{task}_{{g}}.log').read()[-600:])\n"
                    "if any(p.returncode for p in procs): raise SystemExit('a reader process failed: see the log tails above')")


def pack(name):
    return ("code", "# results + transcripts (markdown only, not the copied page images)\n"
                    f"with zipfile.ZipFile(f'{{WORK}}/{name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
                    "    for f in glob.glob(f'{WORK}/results/**/*', recursive=True) + glob.glob(f'{WORK}/Outputs/**/*.md', recursive=True):\n"
                    "        if os.path.isfile(f): z.write(f, os.path.relpath(f, WORK))\n"
                    f"print(os.path.getsize(f'{{WORK}}/{name}.zip') // 1024, 'KB')")


NB1 = SETUP + [
    ("md", "## 1. Untuned Qwen2.5-VL-3B on FERMAT test: how often does it silently fix student mistakes?"),
    ("code", "if not DRY:\n"
             "    !python src/finetune/eval_fermat.py --model hf:Qwen/Qwen2.5-VL-3B-Instruct --split test --name qwen3b_base --out {WORK}/results/fermat {LIMIT}"),
    ("md", "## 2. Untuned Qwen2.5-VL-3B transcribes the 293 EDU-CIRCUIT-HW exams that have an expert transcript\n"
           "Resumable: finished pages are skipped if you re-run."),
    transcribe("qwen3b_base"),
    *score_edu("qwen3b_base"),
    ("md", "## 3. Download `reader_baselines.zip` from the Output tab"),
    pack("reader_baselines"),
]

NB2 = SETUP + [
    ("md", "## 1. QLoRA fine-tune the faithful reader on FERMAT train (image → what was written, errors kept)\n"
           "Kaggle sessions stop after ~12 h. Checkpoints are saved every 50 steps, and re-running this cell resumes."),
    ("code", "if not DRY:\n"
             "    !python src/finetune/train_reader.py --out {WORK}/reader_lora --epochs 1"),
    ("md", "## 2. Evaluate on FERMAT test (compare with qwen3b_base from notebook 01)"),
    ("code", "if not DRY:\n"
             "    !python src/finetune/eval_fermat.py --model hf:Qwen/Qwen2.5-VL-3B-Instruct --adapter {WORK}/reader_lora "
             "--split test --name qwen3b_ft --out {WORK}/results/fermat {LIMIT}"),
    ("md", "## 3. Transcribe the 293 EDU exams with the fine-tuned reader, then score"),
    ("code", "ADAPTER = '' if DRY else f'--adapter {WORK}/reader_lora'"),
    transcribe("qwen3b_ft", "{ADAPTER} "),
    *score_edu("qwen3b_ft"),
    pack("reader_finetuned"),
    ("code", "if os.path.isdir(f'{WORK}/reader_lora'): shutil.make_archive(f'{WORK}/reader_lora_adapter', 'zip', WORK, 'reader_lora')"),
]


NB5 = SETUP + [
    ("md", "## Smoke test of `scripts/run_all.py` on a real GPU (3 samples per step, ~30-40 min incl. downloads)\n"
           "Same script that runs the full job on the local RTX 5070 Ti. Must end with **ALL DONE**."),
    ("code", "CODE = os.path.dirname(os.path.dirname(os.path.dirname(llm)))  # dataset root holding src/ and scripts/\n"
             "shutil.copytree(f'{CODE}/scripts', f'{WORK}/scripts', dirs_exist_ok=True)\n"
             "shutil.copy(f'{CODE}/requirements-gpu.txt', WORK)\n"
             "if not DRY:\n"
             "    os.system(f'{sys.executable} -m pip install -q -r {WORK}/requirements-gpu.txt')\n"
             "rc = os.system(f'{sys.executable} {WORK}/scripts/run_all.py --edu-data {EDU_DATA} --edu-repo {EDU_REPO} '\n"
             "               f'--images {IMAGES} --limit 3')\n"
             "print('exit code', rc)\n"
             "if rc: raise SystemExit('run_all reported problems: read the PREFLIGHT / FAILED lines above')"),
    pack("smoke_results"),
]


NB3 = [
    ("md", "## Free open judge + grader (no paid API)\n"
           "Settings: **Accelerator = GPU T4 x2**, **Internet = On**. Add Inputs (private Kaggle Datasets): "
           "`lma_src.zip` and `edu_text.zip` (from `D:/lma/kaggle_upload`). No HF token needed.\n\n"
           "Judge/grader = **Qwen2.5-7B-Instruct (4-bit)**, batched on both T4s; our scripts call it through the "
           "hf-text backend. Because the paper's Gemini judge verdicts ship with the data, we **calibrate** "
           "this open judge against them (agreement / Cohen's kappa)."),
    ("code", "import os, glob, shutil, zipfile, sys\n"
             "IN = os.environ.get('KAGGLE_INPUT', '/kaggle/input')\n"
             "WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')\n"
             "def find(pattern):\n"
             "    hits = sorted(glob.glob(f'{IN}/**/{pattern}', recursive=True) + glob.glob(f'{WORK}/_in/**/{pattern}', recursive=True))\n"
             "    return hits[0] if hits else None\n"
             "# Kaggle usually unzips uploads; if it kept a .zip, extract it ourselves\n"
             "if not (find('src/eval/llm.py') and find('edu_data')):\n"
             "    for z in glob.glob(f'{IN}/**/*.zip', recursive=True):\n"
             "        zipfile.ZipFile(z).extractall(f'{WORK}/_in/' + os.path.basename(z)[:-4])\n"
             "llm, EDU_DATA, EDU_REPO = find('src/eval/llm.py'), find('edu_data'), find('edu_repo')\n"
             "if not (llm and EDU_DATA and EDU_REPO):\n"
             "    for root, dirs, files in os.walk(IN):\n"
             "        if root.count(os.sep) - IN.count(os.sep) < 4: print(root, dirs[:5], files[:5])\n"
             "    raise SystemExit('Missing inputs: attach BOTH datasets (lma-src and edu-text) via Add Input. Found: '\n"
             "                     + str(dict(src=llm, edu_data=EDU_DATA, edu_repo=EDU_REPO)))\n"
             "shutil.copytree(os.path.dirname(os.path.dirname(llm)), f'{WORK}/src', dirs_exist_ok=True)\n"
             "os.environ.update(EDU_DATA=EDU_DATA, EDU_REPO=EDU_REPO)\n"
             "os.chdir(WORK)\n"
             "# Re-runs: attach the previous run's output as an input; its answer caches are reused (failed calls aren't cached)\n"
             "for rel in ['results/llm_cache.sqlite', 'results/judge_calibration/llm_cache.sqlite']:\n"
             "    old = find(rel)\n"
             "    if old and not os.path.exists(f'{WORK}/{rel}'):\n"
             "        os.makedirs(os.path.dirname(f'{WORK}/{rel}'), exist_ok=True); shutil.copy(old, f'{WORK}/{rel}'); print('reusing cache', old)\n"
             "print('code:', llm, '\\ndata:', EDU_DATA, '\\nrepo:', EDU_REPO)\n"
             "if shutil.which('nvidia-smi'): os.system('nvidia-smi --query-gpu=name,memory.total --format=csv')"),
    ("md", "### Judge = Qwen2.5-7B-Instruct, 4-bit, batched transformers (one copy per T4; use **GPU T4 x2**).\n"
           "No vLLM: its current wheels don't install cleanly on Kaggle's T4 image."),
    ("code", "if not os.environ.get('DRY_JUDGE'):\n"
             "    os.system(f'{sys.executable} -m pip install -q -U accelerate bitsandbytes python-dotenv')\n"
             "os.environ.update(HF_BATCH='8', HF_TOKEN_BUDGET='16000')  # lower the budget if you see 'CUDA out of memory'\n"
             "JUDGE = os.environ.get('DRY_JUDGE') or 'hf-text:Qwen/Qwen2.5-7B-Instruct'\n"
             "WORKERS = 16\n"
             "LIMIT = os.environ.get('DRY_LIMIT', '')  # dry runs only: '--limit 5'\n"
             "print('judge =', JUDGE, '| workers =', WORKERS)"),
    ("md", "## 1. Judge calibration: open judge vs the paper's Gemini judge on the SAME Gemini-2.5-Pro transcripts (513 samples)"),
    ("code", "!python src/eval/evaluate_recognition.py --transcripts-root {EDU_DATA}/Observationset_Final --task v6_Gemini_2p5 "
             "--model-dir gemini-2.5-pro --judge {JUDGE} --workers {WORKERS} {LIMIT} --out {WORK}/results/judge_calibration\n"
             "!python src/eval/calibrate_judge.py --ours {WORK}/results/judge_calibration/v6_Gemini_2p5/step10_rechecked.csv --shipped gemini-2.5-pro"),
    ("md", "## 2. Grade impact (B1): does transcription change the grade? Same open grader on expert vs model transcripts\n"
           "Also reports grader validity vs TA labels and the grader noise floor (`--repeats 1`)."),
    ("code", "for folder, mdir in [('v6_Gemini_2p5', 'gemini-2.5-pro'), ('v6_Gemini_3', 'gemini-3-pro-preview'), ('v6_gpt5p1', 'gpt-5.1')]:\n"
             "    !python src/eval/grade_impact.py --target-root {EDU_DATA}/Observationset_Final/{folder} --target-model-dir {mdir} "
             "--grader {JUDGE} --name B1_{folder}_qwen7b --repeats 1 --workers {WORKERS} {LIMIT}"),
    ("code", "shutil.make_archive(f'{WORK}/open_judge_results', 'zip', WORK, 'results')\n"
             "print(os.path.getsize(f'{WORK}/open_judge_results.zip') // 1024, 'KB')"),
]


def cell(kind, text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def write(name, cells, title):
    nb = {"cells": [cell("md", f"# {title}\nTeam Dropouts · Correction Inception")] + [cell(k, t) for k, t in cells],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote notebooks/" + name)


write("01_reader_baselines_kaggle.ipynb", NB1, "01 · Reader baselines (untuned Qwen2.5-VL-3B)")
write("02_finetune_faithful_reader_kaggle.ipynb", NB2, "02 · Fine-tune the faithful reader (QLoRA)")
write("05_smoke_test_run_all_kaggle.ipynb", NB5, "05 · Smoke test of the one-shot local runner")
write("03_open_judge_grader_kaggle.ipynb", NB3, "03 · Free open judge + grader (calibration and grade impact)")
