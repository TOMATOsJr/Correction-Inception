"""Build notebooks_kaggle/detect_misreads_kaggle.ipynb: misread detection (labels -> token scores -> metrics -> A/B check).

Inputs (private Kaggle Datasets, any names: the notebook finds them by content):
  edu_text.zip                -> edu_data/ (expert + Gemini transcripts)
  edu_circuit_hw_images.zip   -> Screenshot_output_anon/ (page images)
  our 3B outputs              -> qwen3b_ft/ (transcripts) + reader_lora/ (adapter)

  python scripts/make_detect_notebook.py
"""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("md", "# Misread detection on Kaggle\n\n"
           "Does our reader's token confidence find the numbers a transcriber misread?\n\n"
           "**Settings:** Accelerator *GPU T4 x2*, Internet *on*. **Add Input:** the 3 private datasets "
           "(`edu_text`, page images, `qwen3b_ft` + `reader_lora`). Folder names don't matter, cell 2 finds them.\n\n"
           "**Outputs:** per run folder `results/detect/<run>/`: `scores.csv` (confidence of every maths token), "
           "`labels.csv` (misread = 1 where the expert disagrees), `detection_metrics.json` (how many misreads "
           "the top-k flags catch), `verify_metrics.json` (A/B check). Only the `*_metrics.json` files may leave "
           "Kaggle: the CSVs quote student work."),
    ("md", "## 1. Code + packages"),
    ("code", "import os, sys, glob, json, shutil, subprocess, zipfile, base64, io\n"
             "IN = os.environ.get('KS_INPUT', '/kaggle/input')\n"
             "WORK = os.environ.get('KS_WORK', '/kaggle/working')\n"
             "REPO = os.environ.get('KS_REPO') or f'{WORK}/ci'\n"
             "\n"
             "def sh(cmd):\n"
             "    print('$', cmd, flush=True)\n"
             "    r = subprocess.run(cmd, shell=True, cwd=REPO)\n"
             "    if r.returncode: raise RuntimeError(f'failed ({r.returncode}): {cmd}')\n"
             "\n"
             "# the code is embedded below (src/detect + the src/eval modules it imports), no git needed\n"
             "if not os.environ.get('KS_REPO'):\n"
             "    zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall(REPO)\n"
             "print(sorted(glob.glob(f'{REPO}/src/*/*.py')))\n"
             "if os.path.exists('/kaggle'):\n"
             "    sh('pip install -q -U \"transformers>=4.51\" accelerate peft')"),
    ("md", "## 2. Find the data"),
    ("code", "roots = [IN, f'{WORK}/_in']\n"
             "\n"
             "def find(pattern, up=1, required=True):\n"
             "    # shallowest match first; a recursive glob over /kaggle/input walks every file and takes ~15 min\n"
             "    hits = []\n"
             "    for depth in range(7):\n"
             "        hits = sorted(h for r in roots for h in glob.glob(f'{r}/' + '*/' * depth + pattern))\n"
             "        if hits: break\n"
             "    if not hits and not required: return None\n"
             "    if not hits: raise FileNotFoundError(f'nothing matches {pattern} under {roots}')\n"
             "    p = hits[0]\n"
             "    for _ in range(up): p = os.path.dirname(p)\n"
             "    return p\n"
             "\n"
             "# Kaggle usually unzips uploads; if it kept a .zip, extract it\n"
             "if find('Rectified_recognized_markdown_done_Anon', required=False) is None:\n"
             "    for depth in range(4):\n"
             "        for z in glob.glob(f'{IN}/' + '*/' * depth + '*.zip'):\n"
             "            zipfile.ZipFile(z).extractall(f'{WORK}/_in/' + os.path.basename(z)[:-4])\n"
             "EDU = find('Rectified_recognized_markdown_done_Anon')                  # expert transcripts\n"
             "IMG = find('Screenshot_output_anon/Homework*/student_*/*.png', up=4)    # page images\n"
             "# optional: without them only Gemini is scored, with the untuned base reader\n"
             "QWEN = find('qwen3b_ft', up=0, required=False)                          # our 3B transcripts\n"
             "LORA = os.environ.get('KS_LORA') or find('reader_lora', up=0, required=False)  # fine-tuned reader\n"
             "GEMINI = f'{EDU}/Observationset_Final/v6_Gemini_2p5'\n"
             "for k, v in dict(EDU=EDU, IMG=IMG, QWEN=QWEN, LORA=LORA, GEMINI=GEMINI).items():\n"
             "    print(f'{k:6s} {v}  exists={bool(v) and os.path.isdir(v)}')\n"
             "print(len(glob.glob(f'{IMG}/Screenshot_output_anon/Homework*/student_*/*.png')), 'page images')"),
    ("md", "## 3. Settings\n\n`LIMIT = None` scores all 293 pages (~8 s/page on a T4, ~40 min per transcriber); "
           "`LIMIT = 5` is a smoke test. Use *Save Version -> Run All* so it runs in the background."),
    ("code", "LIMIT = None\n"
             "K = 0                                           # A/B check of the top-K numbers per page (0 = skip)\n"
             "READER = os.environ.get('KS_READER', 'Qwen/Qwen2.5-VL-3B-Instruct')\n"
             "PROMPT = 'src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt'\n"
             "OUT = f'{WORK}/results/detect'\n"
             "# (name, transcripts, model dir, use the fine-tuned adapter?)\n"
             "FT = bool(LORA) and not os.environ.get('KS_NO_ADAPTER')\n"
             "tag = 'ftreader' if FT else 'basereader'\n"
             "RUNS = ([(f'qwen3b_ft_{tag}', QWEN, 'qwen3b_ft', FT)] if QWEN else []) + \\\n"
             "       [(f'gemini25_{tag}', GEMINI, 'gemini-2.5-pro', FT)]\n"
             "# ablation (does fine-tuning help detection?): add ('qwen3b_ft_basereader', QWEN, 'qwen3b_ft', False)\n"
             "print('runs:', [r[0] for r in RUNS])"),
    ("md", "## 4. Run: labels (CPU) -> token scores (GPU) -> metrics -> A/B check"),
    ("code", "sh('python src/detect/test_detect.py')\n"
             "lim = f'--limit {LIMIT}' if LIMIT else ''\n"
             "for name, root, mdir, ft in RUNS:\n"
             "    d, ad = f'{OUT}/{name}', (f'--adapter {LORA}' if ft else '')\n"
             "    common = f'--edu-data {EDU} --target-root {root} --model-dir {mdir}'\n"
             "    print(f'\\n===== {name}')\n"
             "    sh(f'python src/detect/labels.py {common} --out {d}')\n"
             "    sh(f'python src/detect/score_tokens.py {common} --images {IMG} --prompt-file {PROMPT} --out {d} '\n"
             "       f'--reader hf:{READER} {ad} {lim}')\n"
             "    sh(f'python src/detect/evaluate_detection.py --dir {d}')\n"
             "    if K:\n"
             "        sh(f'python src/detect/verify_choice.py --dir {d} {common} --images {IMG} '\n"
             "           f'--reader hf:{READER} --k {K} {ad} {lim}')"),
    ("md", "## 5. Results\n\n**recall@3** = share of all misreads caught by flagging the 3 least confident numbers "
           "per page; **precision@3** = share of flags that are real misreads; beat the `random` row."),
    ("code", "import pandas as pd\n"
             "for name, *_ in RUNS:\n"
             "    d = f'{OUT}/{name}'\n"
             "    m = json.load(open(f'{d}/detection_metrics.json'))\n"
             "    print(f\"\\n===== {name}: {m['pages']} pages, {m['numbers']} numbers, {m['misreads']} misreads \"\n"
             "          f\"(base rate {m['base_rate']})\")\n"
             "    t = pd.DataFrame(m['signals']).T\n"
             "    t['caught@3'] = (pd.to_numeric(t['recall@3']) * m['misreads']).round().astype(int)\n"
             "    print(t[['auroc', 'auroc_95CI', 'avg_precision', 'precision@3', 'recall@3', 'caught@3']].to_string())\n"
             "    if os.path.exists(f'{d}/verify_metrics.json'):\n"
             "        print('A/B check:', json.load(open(f'{d}/verify_metrics.json')))"),
    ("md", "## 6. Download the metrics (no student text)\n\nOutput tab -> `detect_metrics.zip`."),
    ("code", "with zipfile.ZipFile(f'{WORK}/detect_metrics.zip', 'w') as z:\n"
             "    for f in glob.glob(f'{OUT}/*/*_metrics.json') + glob.glob(f'{OUT}/*/labels_summary.json'):\n"
             "        z.write(f, os.path.relpath(f, OUT))\n"
             "print(zipfile.ZipFile(f'{WORK}/detect_metrics.zip').namelist())"),
]


CODE = ["src/detect/labels.py", "src/detect/score_tokens.py", "src/detect/evaluate_detection.py",
        "src/detect/verify_choice.py", "src/detect/test_detect.py", "src/eval/run_ocr.py",
        "src/eval/evaluate_recognition.py", "src/eval/llm.py", "src/eval/edu_compat.py",
        "src/eval/items_with_expert_transcript.csv", "src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt"]


def code_cell() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(ROOT / f, f)
    b = base64.b64encode(buf.getvalue()).decode()
    lines = "\n".join(b[i:i + 120] for i in range(0, len(b), 120))
    return ("# embedded project code (generated by scripts/make_detect_notebook.py; regenerate after code changes)\n"
            f"# files: {', '.join(CODE)}\n"
            f"CODE_ZIP = '''\n{lines}\n'''")


def main():
    CELLS.insert(2, ("code", code_cell()))
    nb ={"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = ROOT / "notebooks_kaggle" / "detect_misreads_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
