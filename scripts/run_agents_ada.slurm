#!/bin/bash
# SLURM job: the interactive agent protocol on the ADA cluster, using BOTH GPUs at once.
#   GPU 0 -> the 7B agent models (hf-text, batched across exams)
#   GPU 1 -> the fine-tuned 3B faithful reader (crop_reread tools)
# Both stay resident for the whole run, and exams are pipelined so the two GPUs work simultaneously.
#
#   sbatch scripts/run_agents_ada.slurm
#
# Overrides (env vars):
#   LMA_TARGET_ROOT=$HOME/lma_data/transcripts/qwen3b_ft LMA_NAME=agents_qwen3b_ft sbatch scripts/run_agents_ada.slurm
#   LMA_LIMIT=3 sbatch scripts/run_agents_ada.slurm        # smoke test on 3 exams
#   LMA_CONCURRENCY=8 sbatch scripts/run_agents_ada.slurm  # more exams in flight (more 7B batching)
#
# If only ONE GPU is granted, the staged path still works: see the block at the bottom.
# Every phase is resumable: re-submit and finished exams are skipped.

#SBATCH --job-name=lma_agents
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm_agents_%j.out

set -euo pipefail

ROOT="${LMA_ROOT:-$HOME/Correction-Inception}"
DATA="${LMA_DATA:-$HOME/lma_data}"                    # made by scripts/setup_ada.sh
EDU="$DATA/edu_text/edu_data"
IMAGES="$DATA/images"
TARGET_ROOT="${LMA_TARGET_ROOT:-$ROOT/results/Outputs/qwen3b_ft}"   # transcripts to repair
MODEL_DIR="${LMA_MODEL_DIR:-qwen3b_ft}"                             # models/<dir> inside TARGET_ROOT
ADAPTER="${LMA_ADAPTER:-$ROOT/results/reader_lora}"                 # trained faithful reader
NAME="${LMA_NAME:-agents_${MODEL_DIR}}"
OUT="${LMA_OUT:-$ROOT/results/agents}"
LIMIT="${LMA_LIMIT:-}"
CONCURRENCY="${LMA_CONCURRENCY:-6}"

cd "$ROOT"
source "$ROOT/venv/bin/activate"

# Pin the text model to GPU 0 for the whole job (the auto phase and the grader both use it).
export HF_TEXT_DEVICES="${HF_TEXT_DEVICES:-0}"

COMMON=(--target-root "$TARGET_ROOT" --model-dir "$MODEL_DIR" --name "$NAME" --out "$OUT"
        --edu-data "$EDU" --images "$IMAGES"
        --llm hf-text:Qwen/Qwen2.5-7B-Instruct --reader hf:Qwen/Qwen2.5-VL-3B-Instruct
        --adapter "$ADAPTER" --max-claims 3 --tool-budget 4 --workers 8
        --agent-device 0 --reader-device 1 --concurrency "$CONCURRENCY")
if [ -n "$LIMIT" ]; then COMMON+=(--limit "$LIMIT"); fi

echo "=== phase auto (both models resident: agents cuda:0, reader cuda:1)"
python src/agents/run_agents.py auto "${COMMON[@]}"

echo "=== phase decide (arbiter -> variants + interaction/tool stats)"
python src/agents/run_agents.py decide "${COMMON[@]}"

# ---- score every variant with the same scorers used for the reader runs ----
echo "=== scoring variants (maths diff + grade impact)"
for V in none agents noverify advocate_only auditor_only; do
    VROOT="$OUT/$NAME/Outputs/$V"
    [ -d "$VROOT" ] || continue
    if [ -z "$LIMIT" ]; then
        python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$VROOT" \
            --target-model-dir "$MODEL_DIR" --name "${NAME}_${V}" --out "$OUT/$NAME/transcript_diff" || true
        python src/eval/grade_impact.py --data-root "$EDU" --target-root "$VROOT" \
            --target-model-dir "$MODEL_DIR" --grader hf-text:Qwen/Qwen2.5-7B-Instruct \
            --name "B1_${NAME}_${V}_qwen7b" --workers 8 || true
    fi
done

echo "=== ALL DONE -> $OUT/$NAME"
echo "    decide_summary.json      : interaction + tool + claim-precision stats"
echo "    interactive_ledger.jsonl : one line per exam, full claim history"
echo "    Outputs/<variant>/       : corrected transcripts, ready for any scorer"

# ---------------------------------------------------------------------------------------------
# Single-GPU fallback (e.g. --gres=gpu:1): unset HF_TEXT_DEVICES and run the phases in sequence
# (each phase keeps exactly one model in memory).
#   python src/agents/run_agents.py r1    "${COMMON[@]}"
#   python src/agents/run_agents.py tools --round 1 "${COMMON[@]}"
#   python src/agents/run_agents.py cross "${COMMON[@]}"
#   python src/agents/run_agents.py tools --round 2 "${COMMON[@]}"
#   python src/agents/run_agents.py rebut "${COMMON[@]}"
#   python src/agents/run_agents.py decide "${COMMON[@]}"
# ---------------------------------------------------------------------------------------------
