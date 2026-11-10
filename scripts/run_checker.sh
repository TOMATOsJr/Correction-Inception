#!/usr/bin/env bash
# Fine-tuned A/B checker on ADA: build pairs -> train -> measure -> use it in the correction pipeline.
# Why and what: docs/CHECKER.md
#
#   bash scripts/run_checker.sh                 # everything (or: sbatch scripts/checker_ada.slurm)
#   STAGES="eval" bash scripts/run_checker.sh   # only some stages
#
# Stages (each skips finished work, so re-running after a crash or time-out continues):
#   test     unit tests (CPU, seconds)
#   pairs    A/B pairs from FERMAT (downloads it once, needs HF_TOKEN in .env) and from EDU (labels.csv + transcripts)
#   train    checker_f (FERMAT pairs) and checker_fe (FERMAT + 3 EDU students), both starting from our reader's LoRA
#   eval     base model, reader LoRA, checker_f, checker_fe on held-out pairs -> results/checker/eval/summary.json
#   correct  the top-3 most doubtful numbers per page checked by the reader (old way) and by the best checker,
#            on students the checker never saw -> change / keep decisions -> transcript diff [+ grades: LMA_GRADE=1]
#
# Overrides (env vars):
#   STAGES="test pairs train eval correct"
#   LMA_SMOKE=1                  tiny run of every stage (~20 min) to catch path problems: results/checker_smoke
#   LMA_TRAIN_STUDENTS=3         EDU students used to train checker_fe (and left out of every evaluation)
#   LMA_MAX_EXAMPLES=6000        training examples per checker
#   LMA_CHECKER=auto             checker used in 'correct': auto (best edu/misread accuracy) | checker_f | checker_fe
#   LMA_CORRECT_BASELINE=1       0 = skip the reader-as-checker baseline in 'correct' (saves half the time)
#   LMA_GRADE=0                  1 = also grade none / corrected with the 7B grader (slow)
#   LMA_DATA / LMA_EDU / LMA_IMAGES / LMA_QWEN_ROOT / LMA_ADAPTER / LMA_DETECT_DIR / LMA_OUT / HF_HOME   paths
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${LMA_VENV:-$ROOT/../venv}/bin/activate"

DATA="${LMA_DATA:-$HOME/Correction-Inception/lma_data}"
EDU="${LMA_EDU:-$DATA/edu_text/edu_data}"
IMAGES="${LMA_IMAGES:-$EDU}"
PROMPT=src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt
QWEN_ROOT="${LMA_QWEN_ROOT:-local_results/results/Outputs/qwen3b_ft}"
ADAPTER="${LMA_ADAPTER:-local_results/results/reader_lora}"
SMOKE="${LMA_SMOKE:-0}"
OUT="${LMA_OUT:-results/checker$([ "$SMOKE" = 1 ] && echo _smoke)}"
# labels.csv + scores.csv of the Qwen transcripts (made by run_detect.sh; made here if missing). The smoke run keeps
# its own copy, so its 4-page scores never stand in for the real ones.
DETECT_DIR="${LMA_DETECT_DIR:-$([ "$SMOKE" = 1 ] && echo "$OUT/detect" || echo results/detect/qwen3b_ft_ftreader)}"
STAGES="${STAGES:-test pairs train eval correct}"
TRAIN_STUDENTS="${LMA_TRAIN_STUDENTS:-3}"
MAX_EXAMPLES="${LMA_MAX_EXAMPLES:-6000}"
PIX=602112   # 768*28*28, the resolution of the earlier local correction run; the checker is trained at the same
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export PYTHONUNBUFFERED=1
if [ "$SMOKE" = 1 ]; then
  MAX_EXAMPLES=40; PER_KIND=10; LIMIT="--limit 4"; EPOCHS=1
else
  PER_KIND=400; LIMIT=""; EPOCHS=1
fi

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== stages: $STAGES | out: $OUT | HF_HOME: $HF_HOME"

NGPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
has() { [[ " $STAGES " == *" $1 "* ]]; }
need_gpu() {
  if [ "$NGPU" -lt 1 ]; then
    echo "no GPU visible (CUDA sees 0): stopping instead of running on the CPU. Check nvidia-smi / --gres." >&2
    exit 1
  fi
}
# run two commands on two GPUs at once if there are two, else one after the other
pair_run() {  # "cmd0" "cmd1" (each gets --device appended)
  if [ "$NGPU" -ge 2 ]; then
    eval "$1 --device 0" & local p0=$!
    eval "$2 --device 1" & local p1=$!
    local rc=0
    wait $p0 || rc=1
    wait $p1 || rc=1   # let both finish before stopping on a failure
    [ $rc = 0 ]
  else
    eval "$1 --device 0"; eval "$2 --device 0"
  fi
}

# ---------------------------------------------------------------- test
if has test; then
  python src/detect/test_detect.py
  python src/finetune/test_checker.py
fi

# ---------------------------------------------------------------- pairs
PAIRS="$OUT/pairs"
if has pairs; then
  if [ ! -f "$PAIRS/pairs_fermat_train.jsonl" ]; then
    echo "=== FERMAT pairs"
    python src/finetune/ab_data.py fermat --out "$PAIRS"
  fi
  if [ ! -f "$DETECT_DIR/labels.csv" ]; then
    echo "=== labels for $QWEN_ROOT (no $DETECT_DIR/labels.csv yet)"
    python src/detect/labels.py --edu-data "$EDU" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft --out "$DETECT_DIR"
  fi
  if [ ! -f "$PAIRS/pairs_edu_train.jsonl" ]; then
    echo "=== EDU pairs"
    python src/finetune/ab_data.py edu --labels "$DETECT_DIR/labels.csv" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft \
      --images "$IMAGES" --train-students "$TRAIN_STUDENTS" --out "$PAIRS"
  fi
  cat "$PAIRS/pairs_summary.json"
fi
TRAIN_LIST=$(python -c "import json; print(','.join(json.load(open('$PAIRS/pairs_summary.json'))['edu']['train_students']))" 2>/dev/null || echo "")

# ---------------------------------------------------------------- train
train_cmd() {  # tag sources
  echo "python src/finetune/train_checker.py --pairs $PAIRS --sources $2 --out $OUT/$1 --init-adapter $ADAPTER \
    --max-examples $MAX_EXAMPLES --epochs $EPOCHS --max-pixels $PIX"
}
if has train; then
  need_gpu
  todo=()
  for spec in checker_f:fermat checker_fe:fermat,edu; do
    tag="${spec%%:*}"
    # the top-level adapter is only saved when training finishes (checkpoints live in checkpoint-*/)
    [ -f "$OUT/$tag/adapter_config.json" ] && { echo "=== $tag already trained"; continue; }
    todo+=("$(train_cmd "$tag" "${spec#*:}")")
  done
  if [ "${#todo[@]}" = 2 ]; then pair_run "${todo[0]}" "${todo[1]}"
  elif [ "${#todo[@]}" = 1 ]; then eval "${todo[0]} --device 0"; fi
fi

# ---------------------------------------------------------------- eval
eval_cmd() {  # tag [adapter]
  local ad=""; [ -n "${2:-}" ] && ad="--adapter $2"
  echo "python src/finetune/eval_checker.py --pairs $PAIRS --tag $1 $ad --max-pixels $PIX --per-kind $PER_KIND --out $OUT/eval"
}
if has eval; then
  need_gpu
  todo=()
  for spec in base: reader:$ADAPTER checker_f:$OUT/checker_f checker_fe:$OUT/checker_fe; do
    tag="${spec%%:*}"; ad="${spec#*:}"
    [ -f "$OUT/eval/$tag/checker_metrics.json" ] && { echo "=== eval $tag done"; continue; }
    if [ -n "$ad" ] && [ ! -f "$ad/adapter_config.json" ]; then echo "=== eval $tag: no adapter at $ad, skipped"; continue; fi
    todo+=("$(eval_cmd "$tag" "$ad")")
  done
  while [ "${#todo[@]}" -ge 2 ]; do pair_run "${todo[0]}" "${todo[1]}"; todo=("${todo[@]:2}"); done
  [ "${#todo[@]}" = 1 ] && eval "${todo[0]} --device 0"
  python src/finetune/checker_table.py --eval "$OUT/eval"
fi

# ---------------------------------------------------------------- correct
if has correct; then
  need_gpu
  if [ ! -f "$DETECT_DIR/scores.csv" ] && ! ls "$DETECT_DIR"/scores*.csv >/dev/null 2>&1; then
    echo "=== token scores for $QWEN_ROOT (needed to pick the doubtful numbers)"
    python src/detect/score_tokens.py --edu-data "$EDU" --images "$IMAGES" --prompt-file "$PROMPT" \
      --target-root "$QWEN_ROOT" --model-dir qwen3b_ft --out "$DETECT_DIR" --reader hf:Qwen/Qwen2.5-VL-3B-Instruct \
      --adapter "$ADAPTER" --max-pixels $PIX --device 0 $LIMIT
  fi
  CHECKER="${LMA_CHECKER:-auto}"
  if [ "$CHECKER" = auto ]; then
    CHECKER=$(python src/finetune/checker_table.py --eval "$OUT/eval" --best)
  fi
  echo "=== correction with $CHECKER; students left out (checker training): ${TRAIN_LIST:-none}"
  runs=("$CHECKER:$OUT/$CHECKER")
  [ "${LMA_CORRECT_BASELINE:-1}" = 1 ] && runs=("reader:$ADAPTER" "${runs[@]}")
  for spec in "${runs[@]}"; do
    tag="${spec%%:*}"; ad="${spec#*:}"
    dir="$OUT/correct_$tag"
    mkdir -p "$dir"
    for src in "$DETECT_DIR"/labels.csv "$DETECT_DIR"/scores*.csv; do  # same numbers checked in every run
      [ -e "$dir/$(basename "$src")" ] || ln -s "$(realpath "$src")" "$dir/"
    done
    vcmd="python src/detect/verify_choice.py --dir $dir --edu-data $EDU --images $IMAGES --target-root $QWEN_ROOT \
      --model-dir qwen3b_ft --select topk --k 3 --reader hf:Qwen/Qwen2.5-VL-3B-Instruct --adapter $ad \
      --max-pixels $PIX --exclude-students '$TRAIN_LIST' $LIMIT"
    if [ "$NGPU" -ge 2 ]; then
      pair_run "$vcmd --shard 0/2" "$vcmd --shard 1/2"
    else
      eval "$vcmd --device 0"
    fi
    python src/detect/correct.py --dir "$dir" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft
    for v in none corrected; do
      python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir qwen3b_ft \
        --name "${tag}_$v" --out "$dir/transcript_diff" || true
      if [ "${LMA_GRADE:-0}" = 1 ]; then
        python src/eval/grade_impact.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir qwen3b_ft \
          --grader hf-text:Qwen/Qwen2.5-7B-Instruct --name "checker_${tag}_${v}_qwen7b" --workers 8 || true
      fi
    done
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi
echo "=== done. Results: $OUT/CHECKER_RESULTS.md"
