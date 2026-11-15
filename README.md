# Correction Inception

Detecting and verifying transcription errors that change grades when AI grades handwritten exams.

**Team Dropouts:** Prabhav (2023115008), Jayant (2023115016), Saloni (2023115001), Sampath (2023102033). LMA Group Project 2026.
**Report:** [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)

AI grading of handwritten answers runs in two steps: a vision-language model (VLM) transcribes the page, then an LLM grades the transcript. Reading mistakes turn into wrong grades in two ways:
- **misreads** cost students marks they earned;
- **silent corrections** ("correction inception") give students marks they did not earn, because the reader quietly fixed their mistake.

This repository contains:
1. **Evaluation:** a model-agnostic reproduction of EDU-CIRCUIT-HW, plus our grade-impact metric **TCE**.
2. **A verbatim reader:** Qwen2.5-VL-3B fine-tuned with QLoRA on FERMAT.
3. **Misread detection:** teacher-forced confidence and a learned selector.
4. **Correction tools:** forced-choice verification, a fine-tuned A/B checker, beam candidates, a blank-image guard and an answer-key guard.
5. **Advocate/Auditor agents** with a rule-based or learned Arbiter, plus the full ablation study.

---

## Headline results

| | Result | Source |
|---|---|---|
| SER of 6 VLMs (paper method) | 37.6–85.6% (paper: 37.62–85.43%) | `results/baseline_reference/reference_paper_metrics.json` |
| TCE, Gemini 2.5 Pro / our reader | 8.1% / 28.6% | `results/kaggle_run2/`, `results/pc_run/` |
| FERMAT silent correction, base → our QLoRA reader | 22.2% → 14.0% | `results/checker/CHECKER_RESULTS.md` §5 |
| Token error vs written, base → our reader | 34.9% → 8.5% | same |
| Misread detection AUROC (Qwen / Gemini) | 0.75 / 0.83 | `results/detect/*/detection_metrics.json` |
| Learned selector at 80% recall (Qwen) | 44% of numbers flagged (confidence alone: 64%) | `docs/DETECT_AND_CORRECT_PLAN.md` |
| A/B checker: true value picked on EDU misreads / correct numbers kept | 57.1% / 99.5% | `results/checker/CHECKER_RESULTS.md` §1 |
| Ablation TCE, no correction → agents → B2 self-correction (qwen) | 28.2 → 28.2 → 25.0 | `results/ablation/ABLATION_RESULTS.md` |

In short: detection works, but automatic correction does not yet lower grading errors. The report explains why.

---

## Repository layout

```
src/eval/        transcription (run_ocr.py), paper evaluation (evaluate_recognition.py),
                 maths diff (transcript_diff.py), grade impact / TCE (grade_impact.py), LLM backends (llm.py)
src/finetune/    FERMAT data + split, QLoRA reader training, FERMAT eval, A/B checker training
src/detect/      token scoring, misread labels, detection metrics, selector, forced-choice verification,
                 beam candidates, consistency, correction, learned arbiter, FERMAT/EDU silent-correction audits
src/agents/      claim ledger, verification tools, Arbiter, Advocate/Auditor, run_agents.py
src/analysis/    no-GPU analyses on the released data
scripts/         runners: run_detect.sh, run_checker.sh, run_ablation.sh, ablation_full.slurm, setup_ada.sh, ...
job.sh           SLURM job for the A/B checker pipeline
notebooks/       Kaggle notebooks (reader baselines, fine-tuning)
results/         metrics.json and summaries for every run (no student data)
docs/            report, plans (DETECT_AND_CORRECT_PLAN.md, CHECKER.md, ABLATION.md), ablation run notes
```

---

## 1. Requirements

- Linux and an NVIDIA GPU with at least 11 GB (tested on 2× RTX 2080 Ti on IIIT-H ADA, a T4 on Kaggle, and an RTX 5070 Ti).
- Python 3.11. Python 3.9 cannot install `transformers==5.20.2`.
- About 40 GB of disk for the models, data and Hugging Face cache.
- A Hugging Face account that has accepted the FERMAT terms (`ai4bharat/FERMAT` is gated).

```bash
python3.11 -m venv ~/lma-venv311 && source ~/lma-venv311/bin/activate
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt        # transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, ...
echo "HF_TOKEN=hf_..." > .env              # never commit .env
```

For CPU-only analyses, `pip install -r requirements.txt` is enough.

## 2. Data

| Data | How to get it | Where it goes |
|---|---|---|
| EDU-CIRCUIT-HW (transcripts, rubrics, expert transcripts, page images, released VLM transcripts in `Observationset_Final/v6_*`) | `bash download_edu_zip.sh`, or see https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL | `data/edu_text/edu_data/` |
| FERMAT | Downloaded automatically on first use (~5 GB; needs `HF_TOKEN`) | Hugging Face cache |
| Our reader transcripts (`qwen3b_ft`) and LoRA adapter (`reader_lora`) | From the team bundle, or regenerate them (step 4) | `results/Outputs/qwen3b_ft/`, `results/reader_lora/` |

Set the paths:

```bash
export LMA_DATA=$PWD/data
export LMA_QWEN_ROOT=results/Outputs/qwen3b_ft
export LMA_ADAPTER=results/reader_lora
export HF_HOME=/scratch/$USER/hf_cache     # on ADA: node-local, because /share1 is not mounted on compute nodes
```

Student transcripts, `labels.csv` and the `scores*.csv` files quote student work. **Do not commit them** (they are in `.gitignore`).

## 3. Quick check (CPU, no downloads beyond the dataset)

```bash
python -m pytest src/agents src/detect -q                 # unit tests
python src/eval/evaluate_recognition.py --reference       # recomputes the paper's SER/AEC table from released verdicts
```

## 4. Reproducing each experiment

All GPU stages are **resumable**: if a run is killed, submit the same command again.

| Experiment | Command | Output | Time (2× 2080 Ti) |
|---|---|---|---|
| Transcribe with our reader | `python src/eval/run_ocr.py --model qwen3b_ft --adapter $LMA_ADAPTER --repetition-penalty 1.05` | `results/Outputs/qwen3b_ft/` | ~2–3 h |
| Fine-tune the reader (QLoRA on FERMAT) | `python src/finetune/train_reader.py` (or the Kaggle notebook `notebooks/02_*`) | `results/reader_lora/` | ~10–20 h on a T4 |
| FERMAT silent-correction eval | `python src/finetune/eval_fermat.py` | `results/fermat/*/metrics.json` | ~1 h |
| Grade impact (TCE) | `python src/eval/grade_impact.py --target <transcript dir>` | `results/grade_impact/<name>/metrics.json` | ~10–30 min per transcriber |
| Detection + correction | `bash scripts/run_detect.sh` (`LMA_LIMIT=10` for a smoke test; `LMA_GRADE=1` adds TCE) | `results/detect/`, `results/correction*/` | several hours |
| A/B checker (train, eval, correct, Arbiter, agents) | `LMA_SMOKE=1 sbatch job.sh`, then `sbatch job.sh` | `results/checker/CHECKER_RESULTS.md` | ~6–10 h |
| **Ablation study** | `sbatch scripts/ablation_full.slurm` | `results/ablation/ABLATION_RESULTS.md`, `results/grade_impact/abl_*/` | ~3.5 h |

Details per pipeline: `docs/DETECT_AND_CORRECT_PLAN.md`, `docs/CHECKER.md`, `docs/ABLATION.md`, and `docs/ABLATION_RUN_2026-11-15.md`.

### Running on IIIT-H ADA

```bash
sbatch scripts/ablation_full.slurm      # -A research -p u22 -G 2 --time=3-00:00:00
squeue -u $USER                         # status
tail -f ablation_<jobid>.out            # progress
```

- Partition `u22` has a **1-hour default time limit**, so always pass `--time`. QoS `medium` allows up to 4 days and 4 GPUs.
- `sinteractive` sessions end after 6 h, so use `sbatch` for long runs. Closing your laptop does not stop a batch job.
- The job downloads Qwen2.5-7B-Instruct and Qwen2.5-VL-3B-Instruct into the node's `/scratch`, then sets `HF_HUB_OFFLINE=1`.

## 5. Models used

| Role | Model |
|---|---|
| Transcriber (ours) | Qwen2.5-VL-3B-Instruct + QLoRA (`reader_lora`) |
| Transcriber (reference) | Gemini 2.5 Pro and others; released transcripts only, no API calls |
| Grader, agents, B2 self-correction | Qwen2.5-7B-Instruct, 4-bit nf4 |
| Re-reader, A/B checker | Qwen2.5-VL-3B-Instruct + `reader_lora` / `checker_fe` |

## 6. Branches

| Branch | Contents |
|---|---|
| `main` | The final merged code and results (after merging the branches below) |
| `ab_checker` | Detection, correction, A/B checker, learned Arbiter, tool-calling agents |
| `ablation-results` | Ablation scripts and the 15 Nov ADA run |
| `agent_interaction`, `misread_detection`, `forced_choice` | Earlier stages; already contained in `ab_checker` |

## Credits and licences

Evaluation prompts and parsing come from EDU-CIRCUIT-HW (Apache-2.0). FERMAT is used under CC BY 4.0. Qwen models are used under their respective licences.
