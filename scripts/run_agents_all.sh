#!/usr/bin/env bash
# Full agent-protocol run on ADA: ALL exams with expert transcripts (293), not just the smoke.
# Same phases as the 3-exam smoke, minus --limit, plus scoring of every variant.
#
#   bash scripts/run_agents_all.sh
#
# Resumable: the auto phase skips exams already listed in auto_done.jsonl, so re-running
# this script after a crash or a killed session continues where it stopped. The decide
# and scoring phases are idempotent and are simply re-run.
#
# Overrides (env vars):
#   LMA_NAME=agents_qwen3b_ft_full   run name (default below)
#   LMA_LIMIT=10                     only the first 10 exams (turns this back into a smoke)
#   LMA_CONCURRENCY=8                exams in flight (more 7B batching on the agent GPU)
#   LMA_THETA=0.85                   arbiter acceptance threshold
#   LMA_SELF_CORRECT=1               also run the B2 one-pass self-correction baseline
#   LMA_OUT / LMA_TARGET_ROOT / LMA_ADAPTER / LMA_DATA   path overrides
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/venv/bin/activate"

DATA="${LMA_DATA:-$HOME/Correction-Inception/lma_data}"
TARGET_ROOT="${LMA_TARGET_ROOT:-local_results/results/Outputs/qwen3b_ft}"
MODEL_DIR="${LMA_MODEL_DIR:-qwen3b_ft}"
ADAPTER="${LMA_ADAPTER:-local_results/results/reader_lora}"
NAME="${LMA_NAME:-agents_qwen3b_ft_full}"
OUT="${LMA_OUT:-results/agents}"
LIMIT="${LMA_LIMIT:-}"
CONCURRENCY="${LMA_CONCURRENCY:-6}"
THETA="${LMA_THETA:-0.85}"
export HF_TEXT_DEVICES="${HF_TEXT_DEVICES:-0}"   # pin the 7B agent model to GPU 0

mkdir -p "$OUT/$NAME"
LOG="$OUT/$NAME/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== root:        $ROOT"
echo "=== data:        $DATA"
echo "=== transcripts: $TARGET_ROOT (model dir: $MODEL_DIR)"
echo "=== adapter:     $ADAPTER"
echo "=== run name:    $NAME   out: $OUT"
echo "=== log:         $LOG"

# ---- sanity checks before burning GPU hours
[ -d "$TARGET_ROOT" ] || { echo "!!! target-root missing: $TARGET_ROOT"; exit 1; }
[ -d "$ADAPTER" ] || { echo "!!! adapter missing: $ADAPTER"; exit 1; }
[ -d "$DATA/edu_text/edu_data" ] || { echo "!!! edu-data missing: $DATA/edu_text/edu_data"; exit 1; }

N_EXAMS=$(python -c "import pandas as pd; print(len(pd.read_csv('src/eval/items_with_expert_transcript.csv', dtype=str)))")
DONE=$(wc -l < "$OUT/$NAME/auto_done.jsonl" 2>/dev/null || echo 0)
echo "=== exams in the item list: $N_EXAMS"
echo "=== already negotiated:     $DONE  ->  $((N_EXAMS - DONE)) to go"

COMMON=(--target-root "$TARGET_ROOT" --model-dir "$MODEL_DIR" --name "$NAME" --out "$OUT"
        --edu-data "$DATA/edu_text/edu_data" --images "$DATA/images"
        --llm hf-text:Qwen/Qwen2.5-7B-Instruct --reader hf:Qwen/Qwen2.5-VL-3B-Instruct
        --adapter "$ADAPTER"
        --max-claims 3 --tool-budget 4 --workers 8
        --agent-device 0 --reader-device 1 --concurrency "$CONCURRENCY" --theta "$THETA")
if [ -n "$LIMIT" ]; then COMMON+=(--limit "$LIMIT"); fi

echo "=== phase auto (agents cuda:0, reader cuda:1) — resumable"
python src/agents/run_agents.py auto "${COMMON[@]}"

echo "=== phase decide (arbiter -> variants + interaction/tool stats)"
python src/agents/run_agents.py decide "${COMMON[@]}"

# ---- score every variant with the same scorers used for the reader runs
echo "=== scoring variants (maths diff + grade impact)"
for V in none agents noverify advocate_only auditor_only; do
    VROOT="$OUT/$NAME/Outputs/$V"
    [ -d "$VROOT" ] || { echo "--- skip $V (no Outputs/$V)"; continue; }
    if [ -f "$OUT/$NAME/transcript_diff/${NAME}_${V}/metrics.json" ]; then
        echo "--- $V already scored, skipping"
        continue
    fi
    echo "--- $V: transcript_diff"
    python src/eval/transcript_diff.py --data-root "$DATA/edu_text/edu_data" \
        --target-root "$VROOT" --target-model-dir "$MODEL_DIR" \
        --name "${NAME}_${V}" --out "$OUT/$NAME/transcript_diff" || true
    echo "--- $V: grade_impact"
    python src/eval/grade_impact.py --data-root "$DATA/edu_text/edu_data" \
        --target-root "$VROOT" --target-model-dir "$MODEL_DIR" \
        --grader hf-text:Qwen/Qwen2.5-7B-Instruct \
        --name "B1_${NAME}_${V}_qwen7b" --workers 8 || true
done

if [ "${LMA_SELF_CORRECT:-0}" = "1" ]; then
    echo "=== phase selfcorrect (baseline B2, one-pass fix)"
    python src/agents/run_agents.py selfcorrect "${COMMON[@]}"
fi

echo "=== ALL DONE -> $OUT/$NAME"
echo "    decide_summary.json      : interaction + tool + claim-precision stats"
echo "    interactive_ledger.jsonl : one line per exam, full claim history"
echo "    Outputs/<variant>/       : corrected transcripts (scored above)"
echo "    transcript_diff/         : maths-diff metrics per variant"
echo "    results/grade_impact/    : TCE per variant"
