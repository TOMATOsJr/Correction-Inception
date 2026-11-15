# Correction Inception

Detecting and verifying the transcription errors that change grades when AI grades handwritten exams.

**Team Dropouts:** Prabhav (2023115008), Jayant (2023115016), Saloni (2023115001), Sampath (2023102033). LMA Group Project 2026.

**Final report:** [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) · **Repository:** https://github.com/TOMATOsJr/Correction-Inception

AI grading of handwritten answers runs in two steps: a vision-language model (VLM) transcribes the page, then an LLM grades the transcript. Reading mistakes turn into wrong grades in two ways:
- **misreads** cost students marks they earned;
- **silent corrections** ("correction inception") give students marks they did not earn, because the reader quietly fixed their mistake.

We measure this grade damage, fine-tune a small reader to transcribe verbatim, detect likely errors, and try to correct them automatically. The correction uses tools (a fine-tuned A/B checker, beam candidates, a blank-image guard, an answer-key guard, page consistency) and two opposing agents: an **Advocate** (looks for misreads) and an **Auditor** (looks for silent corrections).

> **Where the code is.** `main` holds the base pipeline, this README, the final report and the summaries of the final results. The later work lives on branches, which build on each other (see [Branches](#branches)). The most complete code is on **`ab_checker`**; the ablation study is on **`ablation-results`**. To run a part, check out its branch: `git checkout ab_checker`.

---

## Headline results

| | Result | Source |
|---|---|---|
| SER of 6 VLMs (paper method) | 37.6–85.6% (paper: 37.62–85.43%) | `results/baseline_reference/reference_paper_metrics.json` |
| TCE (transcription-caused grading error): Gemini 2.5 Pro / our reader | 8.1% / 28.6% | `results/kaggle_run2/`, `results/pc_run/` |
| FERMAT answers silently corrected: base → our QLoRA reader | 22.2% → **14.0%** | `results/checker/CHECKER_RESULTS.md` §5 |
| FERMAT token error vs what is written: base → our reader | 34.9% → **8.5%** | same |
| A/B checker (`checker_fe`): keeps a correct EDU number / picks the truth on an EDU misread | 99.5% / 57.1% (base: 90.2% / 53.1%) | same, §1 |
| Misread detection AUROC (Qwen / Gemini) | 0.75 / 0.83 | `misread_detection` branch, `results/detect/` |
| Learned selector at 80% recall (Qwen) | 44% of numbers flagged (confidence alone: 64%) | `forced_choice` branch, `docs/DETECT_AND_CORRECT_PLAN.md` |
| FERMAT silent corrections found by the blank-image signal | AUROC 0.916; 16 of 18 in the top 4 per answer | `results/checker/CHECKER_RESULTS.md` §5 |
| Best automatic correction on unseen students (304 misreads) | net **+2** (5 fixed, 0 changed to another wrong value, 3 broken) | same, §2 |
| Tool-calling agents vs the same tools without agents | −7 vs +1 (Advocate alone: −15) | same, §8 |
| Ablation TCE (qwen): no correction / agents / B2 self-correction | 28.2 / 28.2 / 25.0 | `results/ablation/ABLATION_RESULTS.md` |

**In short:**
- **Works:** the fine-tuned models (reader and checker) and error detection.
- **Doesn't yet work:** fully automatic correction, with or without agents. The report explains why.

**Correction to the mid-submission.** `docs/mid_submission.md` and `context.md` report FERMAT silent-correction rates of 82.5% (base) and 76.3% (ours). Those counted FERMAT's red `\textcolor{red}{}` error markers as text. With the markers stripped, the rates are 22.2% and 14.0%.

---

## Branches

How the branches build on each other:

```
main (25 Oct)  evaluation pipeline, reader fine-tuning, baselines, mid-submission, final report
 └─ agent_interaction (27-29 Oct)  interactive Advocate/Auditor protocol, 214-exam agent run
     ├─ ablation (30 Oct)  ablation scripts
     │   └─ ablation-results (15 Nov)  80-exam ablation run on ADA (job 80460), first report draft
     └─ misread_detection (29-30 Oct)  per-number labels, teacher-forced confidence, detection metrics
         ├─ colab-results (30 Oct)  50-exam Colab agent run and detection runs
         └─ forced_choice (29 Oct - 8 Nov)  learned selector, forced-choice verification, guards, correction
             └─ ab_checker (11-15 Nov)  fine-tuned A/B checker, beams, learned Arbiter, consistency,
                                         Auditor on FERMAT/EDU, tool-calling agents   <- most complete code
```

| Branch | Adds (main files) | Results | How to run |
|---|---|---|---|
| **`main`** | `src/eval/`: OCR with any backend, paper evaluation (SER/AEC), maths diff, TCE (`grade_impact.py`), LLM backends. `src/finetune/`: FERMAT split, QLoRA reader training, FERMAT eval. `src/agents/core.py`, `agents.py`: first claim/tool/Arbiter design. `src/analysis/`: no-GPU analyses. Kaggle notebooks. | `results/baseline_reference`, `kaggle_run1/2`, `pc_run`, `transcript_diff`; copies of the final summaries in `results/checker/`, `results/ablation/` | [§4.1](#41-main-evaluation-tce-reader) |
| **`agent_interaction`** | `src/agents/protocol.py`: interactive rounds (r1 → tools → r2 → tools → r3; counter / withdraw / revise / concede), `crop_reread` and consistency tools, B2 self-correction. `scripts/run_agents_all.sh`, `run_agents_ada.slurm`, `setup_ada.sh`, `download_edu_zip.sh`. | 3-exam smoke and 214-exam agent run | [§4.2](#42-agent_interaction-generation-1-agents) |
| **`misread_detection`** | `src/detect/labels.py` (per-number misread labels vs the expert transcript), `score_tokens.py` (teacher-forced nll / entropy / gap), `evaluate_detection.py`; `scripts/run_detect.sh` | `results/detect/*/detection_metrics.json` | [§4.3](#43-misread_detection--forced_choice-detection-and-correction) |
| **`colab-results`** | Colab copies of the agent and detection runs | 50-exam agent run, detection per transcriber/reader | results only |
| **`forced_choice`** | `src/detect/selector.py` (learned selector, folds by student), `verify_choice.py` (A/B forced choice, both orders), `correct.py` (change / keep / TA, blank-image and answer-key guards), `sweep_correct.py`; `scripts/run_correction.py`; `docs/DETECT_AND_CORRECT_PLAN.md` | `results/correction_600k/`, `results/correction_600k_crop/` | [§4.3](#43-misread_detection--forced_choice-detection-and-correction) |
| **`ab_checker`** | `src/finetune/ab_data.py`, `train_checker.py`, `eval_checker.py` (fine-tuned A/B checker); `src/detect/choice_prompt.py`, `beam_candidates.py` (digit beam search on the KV cache), `arbiter.py` (learned Arbiter), `consistency.py`, `fermat_audit.py` (Auditor on FERMAT), `edu_silent.py` (silent corrections via the answer key), `resume.py`; `src/agents/ledger_agents.py` (tool-calling Advocate/Auditor); `scripts/run_checker.sh`, `job.sh`; `docs/CHECKER.md`. Also fixes the FERMAT red-marker leak (`fermat_data.strip_red`). | `results/checker/CHECKER_RESULTS.md` and the `metrics.json` files beside it | [§4.4](#44-ab_checker-checker-correction-arbiter-auditor-agents) |
| **`ablation`** | `scripts/run_ablation.sh`, `docs/ABLATION.md` | – | [§4.5](#45-ablation-results-ablation-study) |
| **`ablation-results`** | `scripts/ablation_full.slurm`, `collect_ablation.py`, `make_ablation_subset.py`; `docs/ABLATION_RUN_2026-11-15.md` | `results/ablation/ABLATION_RESULTS.md`, `results/grade_impact/abl_*/` | [§4.5](#45-ablation-results-ablation-study) |

---

## 1. Requirements

- Linux and an NVIDIA GPU with at least 11 GB. Tested on 2× RTX 2080 Ti (IIIT-H ADA), a T4 (Kaggle/Colab) and an RTX 5070 Ti.
- Python 3.10–3.12 (3.11 recommended; 3.9 cannot install `transformers==5.20.2`).
- About 40 GB of disk for models, data and the Hugging Face cache. Put the cache on scratch space, not a small home quota.
- A Hugging Face account that has accepted the FERMAT terms (`ai4bharat/FERMAT` is gated).

```bash
python3.11 -m venv ~/lma-venv && source ~/lma-venv/bin/activate
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt        # transformers, peft, accelerate, bitsandbytes (pinned)
echo "HF_TOKEN=hf_..." > .env              # your own token; .env is git-ignored, never commit it
```

- **CPU-only analyses:** `pip install -r requirements.txt` is enough.
- **ADA one-time setup** (on `agent_interaction` and later branches): `bash scripts/setup_ada.sh [DATA_DIR] [cu121|cu124]`.
- **No paid APIs needed.** Every model we run is open (Qwen2.5-VL-3B, Qwen2.5-7B). The Gemini/GPT/Claude transcripts we compare against were released with the dataset.

## 2. Data

| Data | How to get it | Where it goes |
|---|---|---|
| EDU-CIRCUIT-HW: page images, expert transcripts, rubrics, TA grades, released VLM transcripts (`Observationset_Final/v6_*`) | `bash download_edu_zip.sh` (on `agent_interaction` and later), or https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL | `$LMA_DATA/edu_text/edu_data/` (ADA layout) |
| The paper's code and prompts | cloned by `scripts/setup.ps1` / `setup_ada.sh` | `external/` or `$LMA_DATA/edu_text/edu_repo/` |
| FERMAT | downloaded automatically on first use (~5 GB; needs `HF_TOKEN`) | Hugging Face cache |
| Our reader's transcripts (`qwen3b_ft`) and LoRA adapter (`reader_lora`) | regenerate them (§4.1), or get them from the team | `results/Outputs/qwen3b_ft/`, `results/reader_lora/` (or `local_results/results/…`) |

```bash
export LMA_DATA=$HOME/lma_data               # or the repo root if the data sits there
export HF_HOME=/scratch/$USER/hf_cache       # on ADA: node-local scratch
```

**Student data.** Transcripts, per-number `labels.csv`, `scores*.csv` and `verify*.csv`, and agent ledgers all quote student work. They are git-ignored; never commit them.

## 3. Quick check (CPU)

```bash
python -m pytest src/agents src/detect -q                 # unit tests (src/detect exists from misread_detection on)
python src/eval/evaluate_recognition.py --reference       # recomputes the paper's SER/AEC table from released verdicts
```

## 4. Reproducing each experiment

Every GPU stage is **resumable**: if a run is killed, run the same command again and it continues. Times are for 2× RTX 2080 Ti.

### 4.1 `main`: evaluation, TCE, reader

| Experiment | Command | Output | Time |
|---|---|---|---|
| Paper SER/AEC from released verdicts | `python src/eval/evaluate_recognition.py --reference` | `results/baseline_reference/` | seconds |
| Transcribe with any model | `python src/eval/run_ocr.py --split debug --task <task> --model <model>` (our reader: `--model hf:Qwen/Qwen2.5-VL-3B-Instruct --model-dir qwen3b_ft --adapter results/reader_lora`) | `results/Outputs/<model>/` | 2–3 h for our reader |
| Fine-tune the verbatim reader (QLoRA on FERMAT) | `python src/finetune/train_reader.py`, or `notebooks/02_finetune_faithful_reader_kaggle.ipynb` | `results/reader_lora/` | 10–20 h on a T4 |
| FERMAT silent-correction eval | `python src/finetune/eval_fermat.py --model hf:Qwen/Qwen2.5-VL-3B-Instruct --adapter results/reader_lora --name qwen3b_ft` | `results/pc_run/results/fermat/*/metrics.json` | ~1 h |
| Maths diff vs the expert transcript | `python src/eval/transcript_diff.py --data-root <edu_data> --target-root <dir> --target-model-dir <model> --name <name>` | `results/transcript_diff/<name>/` | minutes |
| Grade impact (TCE) | `python src/eval/grade_impact.py --data-root <edu_data> --target-root <dir> --target-model-dir <model> --grader hf-text:Qwen/Qwen2.5-7B-Instruct --name <name>` | `results/grade_impact/<name>/metrics.json` | 10–30 min per transcriber |
| Everything in one go | `python scripts/run_all.py` (Kaggle: `notebooks/05_smoke_test_run_all_kaggle.ipynb`) | `results/` | – |

### 4.2 `agent_interaction`: generation-1 agents

```bash
git checkout agent_interaction
LMA_LIMIT=3 sbatch scripts/run_agents_ada.slurm     # smoke test on 3 exams
sbatch scripts/run_agents_ada.slurm                 # 7B agents on GPU 0, the 3B re-reader on GPU 1
bash scripts/run_agents_all.sh                      # all 293 exams; LMA_SELF_CORRECT=1 adds the B2 baseline
```

### 4.3 `misread_detection` / `forced_choice`: detection and correction

```bash
git checkout forced_choice
LMA_LIMIT=10 bash scripts/run_detect.sh             # smoke test
bash scripts/run_detect.sh                          # labels -> scores -> detection -> selector -> forced choice -> decisions
LMA_GRADE=1 bash scripts/run_detect.sh              # + TCE of the corrected transcripts
python scripts/run_correction.py                    # the same on one local GPU (Windows or Linux)
```

Output goes to `results/detect/` and `results/correction*/`; the plan and metrics are in `docs/DETECT_AND_CORRECT_PLAN.md`. Takes several hours.

### 4.4 `ab_checker`: checker, correction, Arbiter, Auditor, agents

`job.sh` is the SLURM job, and it runs `scripts/run_checker.sh`. Choose the stages with `STAGES` (guide: `docs/CHECKER.md`).

```bash
git checkout ab_checker
LMA_SMOKE=1 sbatch job.sh                           # tiny run of every stage (~20-30 min) -> results/checker_smoke
sbatch job.sh                                       # test, pairs, train, eval, correct (~6-10 h)
```

The later stages are independent, so they can run as two jobs on two nodes:
```bash
STAGES="edu_silent_blank audit_fix agents" sbatch -J ckA -o out_A.txt -e out_A.txt job.sh
```
```bash
STAGES="scale consistency arbiter agents" LMA_K=20 LMA_CHECKER=checker_fe \
  LMA_AGENTS_RUN=correct_checker_fe_k20_beams sbatch -J ckB -o out_B.txt -e out_B.txt job.sh
```

| Stage | What it does | GPU / time |
|---|---|---|
| `test` | unit tests | CPU, seconds |
| `pairs`, `train` | build A/B pairs; fine-tune `checker_f` and `checker_fe` (one per GPU) | 2 GPUs, ~3–4 h |
| `eval`, `clean_eval` | checker accuracy on held-out pairs (with and without FERMAT red markers) | 2 GPUs, ~40 min each |
| `correct` | correction with each checker, 3 numbers per page, unseen students | 2 GPUs, ~1 h |
| `scale` | `LMA_K` numbers per page (10 by default) with beam candidates | 2 GPUs, 5–10 h |
| `arbiter`, `consistency`, `edu_silent` | learned Arbiter; page consistency; EDU answer-key analysis | CPU, minutes |
| `audit`, `audit_fix` | Auditor detectors and fixes on FERMAT | 2 GPUs, 0.5–2 h |
| `edu_silent_blank` | blank-image scores for the EDU transcripts | 2 GPUs, ~30 min |
| `agents` | tool-calling Advocate/Auditor (7B, 4-bit, on GPU 0) over a correction run's ledger | 1 GPU, a few hours |

All results land in `results/checker/CHECKER_RESULTS.md`, which every stage regenerates. `LMA_GRADE=1` adds TCE for the corrected transcripts.

### 4.5 `ablation-results`: ablation study

```bash
git checkout ablation-results
sbatch scripts/ablation_full.slurm                  # -A research -p u22 -G 2 --time=3-00:00:00, ~3.5 h
STAGES="verify ablate" bash scripts/run_ablation.sh # only some stages: subset agents verify ablate score collect
```

Output: `results/ablation/ABLATION_RESULTS.md`. Details are in `docs/ABLATION.md` and `docs/ABLATION_RUN_2026-11-15.md`.

### Running on IIIT-H ADA

- Submit with `sbatch <script>`, check with `squeue -u $USER`, and follow the job's `-o` log with `tail -f`.
- Partition `u22` has a 1-hour default time limit, so always pass `--time`. `sinteractive` sessions end after 6 h, so use `sbatch` for long runs.
- Put the Hugging Face cache on the node's `/scratch` (`HF_HOME=/scratch/$USER/hf_cache`). The 7B and the 3B together need ~25 GB, which does not fit in the home quota. The `ab_checker` agents stage does this by itself.

## 5. Models

| Role | Model |
|---|---|
| Transcriber (ours) | Qwen2.5-VL-3B-Instruct + QLoRA (`reader_lora`) |
| Transcribers (reference) | Gemini 2.5 Pro / 3 Pro, GPT-5.1, Claude Sonnet 4.5, Qwen3-VL; released transcripts only, no API calls |
| Grader, agents, B2 self-correction | Qwen2.5-7B-Instruct, 4-bit nf4 |
| Re-reader, token scoring, beams | Qwen2.5-VL-3B-Instruct + `reader_lora` |
| A/B checker | Qwen2.5-VL-3B-Instruct + `checker_fe` (LoRA trained from `reader_lora`) |

## 6. Repository layout (union over the branches)

```
src/eval/        transcription (run_ocr.py), paper evaluation (evaluate_recognition.py), maths diff
                 (transcript_diff.py), grade impact / TCE (grade_impact.py), LLM backends (llm.py)
src/finetune/    FERMAT data + split, QLoRA reader training, FERMAT eval; A/B checker data, training, eval
src/detect/      labels, token scoring, detection metrics, selector, forced choice, beams, correction,
                 learned Arbiter, consistency, FERMAT / EDU silent-correction audits
src/agents/      claims, tools, Arbiter (core.py), protocol.py (generation 1), ledger_agents.py (generation 2)
src/analysis/    no-GPU analyses on the released data
scripts/         runners: run_all.py, setup_ada.sh, run_agents_*.sh, run_detect.sh, run_correction.py,
                 run_checker.sh, run_ablation.sh, ablation_full.slurm
job.sh           SLURM job for the A/B checker pipeline (ab_checker)
notebooks/       Kaggle notebooks (reader baselines, fine-tuning, open grader, smoke test)
results/         metrics.json and summaries for every run (no student data)
docs/            FINAL_REPORT.md, proposal, mid-submission, plans (DETECT_AND_CORRECT_PLAN.md, CHECKER.md, ABLATION.md)
```

## Credits and licences

Evaluation prompts and parsing come from EDU-CIRCUIT-HW (Apache-2.0), https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL. FERMAT is used under CC BY 4.0. Qwen models are used under their respective licences.
