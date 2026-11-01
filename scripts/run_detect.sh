#!/usr/bin/env bash
# Misread detection and correction on ADA. For each transcriber x reader:
#   labels (CPU) -> teacher-forced token scores (GPU) -> detection metrics (CPU) -> learned selector (CPU)
#   -> forced-choice verification of the selected numbers (GPU) -> change / keep / TA + corrected transcripts (CPU)
#   -> maths diff of the corrected transcripts (CPU) [-> grading impact with the 7B grader (GPU), LMA_GRADE=1]
# Plan and metrics: docs/DETECT_AND_CORRECT_PLAN.md
#
#   bash scripts/run_detect.sh
#
# Transcribers: our fine-tuned 3B (qwen3b_ft) and the released Gemini 2.5 Pro transcripts.
# Readers: fine-tuned (adapter) and the untuned base model (ablation: does fine-tuning help detection?).
# Resumable: scoring and verification skip pages already done; the CPU steps are simply re-run.
#
# Overrides (env vars):
#   LMA_LIMIT=10                     only the first 10 pages per run (smoke)
#   LMA_READER_DEVICE=1              GPU for the reader (scoring and verification; one GPU is enough)
#   LMA_TARGET_RECALL=0.8            share of misreads the selector must flag
#   LMA_SELECT=auto                  numbers to verify: selector | topk | auto (selector when it could be trained)
#   LMA_VERIFY_K=3                   topk only: numbers per page; 0 = stop after detection
#   LMA_MAX_FALSE_CHANGE=0.01        correct numbers the corrector may change (sets tau_high on other students)
#   LMA_OTHER=0                      1 = use Gemini 2.5's transcript as a candidate source for the Qwen runs.
#                                    Off by default: the expert transcripts are corrected Gemini transcripts, so this
#                                    flatters the evaluation.
#   LMA_GRADE=0                      1 = also grade none / corrected / corrected_ta with the 7B grader (slow)
#   LMA_OUT / LMA_DATA / LMA_EDU / LMA_IMAGES / LMA_ADAPTER / LMA_QWEN_ROOT   path overrides
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/../venv/bin/activate"

DATA="${LMA_DATA:-$HOME/Correction-Inception/lma_data}"
EDU="${LMA_EDU:-$DATA/edu_text/edu_data}"   # same layout as setup_ada.sh
IMAGES="${LMA_IMAGES:-$EDU}"   # images live in the dataset itself ($DATA/images is only a symlink to it)
PROMPT=src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt   # the paper's v6 prompt (external/ is not in git)
QWEN_ROOT="${LMA_QWEN_ROOT:-local_results/results/Outputs/qwen3b_ft}"
GEMINI_ROOT="$EDU/Observationset_Final/v6_Gemini_2p5"
ADAPTER="${LMA_ADAPTER:-local_results/results/reader_lora}"
OUT="${LMA_OUT:-results/detect}"
LIMIT="${LMA_LIMIT:+--limit $LMA_LIMIT}"
READER_DEVICE="${LMA_READER_DEVICE:-1}"
TARGET_RECALL="${LMA_TARGET_RECALL:-0.8}"
SELECT="${LMA_SELECT:-auto}"
VERIFY_K="${LMA_VERIFY_K:-3}"
MAX_FALSE_CHANGE="${LMA_MAX_FALSE_CHANGE:-0.01}"
export HF_HOME="${HF_HOME-/scratch/jayanth.raju/hf_cache}"

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

run() {  # name target_root model_dir reader_tag [adapter args...]
  local name="$1" root="$2" mdir="$3" tag="$4"; shift 4
  local dir="$OUT/${name}_${tag}"
  local reader=(--reader hf:Qwen/Qwen2.5-VL-3B-Instruct --device "$READER_DEVICE" "$@")
  local other=()
  if [ "${LMA_OTHER:-0}" = 1 ] && [ "$mdir" != gemini-2.5-pro ]; then
    other=(--other-root "$GEMINI_ROOT" --other-model-dir gemini-2.5-pro)
  fi
  echo "=== $dir: transcripts $root ($mdir), reader $tag"
  python src/detect/labels.py --edu-data "$EDU" --target-root "$root" --model-dir "$mdir" --out "$dir"
  python src/detect/score_tokens.py --edu-data "$EDU" --images "$IMAGES" --prompt-file "$PROMPT" --target-root "$root" \
    --model-dir "$mdir" --out "$dir" "${reader[@]}" $LIMIT
  python src/detect/evaluate_detection.py --dir "$dir"
  python src/detect/selector.py --dir "$dir" --target-recall "$TARGET_RECALL"
  [ "$VERIFY_K" = 0 ] && return
  python src/detect/verify_choice.py --dir "$dir" --edu-data "$EDU" --images "$IMAGES" --target-root "$root" \
    --model-dir "$mdir" --select "$SELECT" --k "$VERIFY_K" ${other[@]+"${other[@]}"} "${reader[@]}" $LIMIT
  python src/detect/correct.py --dir "$dir" --target-root "$root" --model-dir "$mdir" --max-false-change "$MAX_FALSE_CHANGE"
  for v in none corrected corrected_ta; do
    python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir "$mdir" \
      --name "${name}_${tag}_$v" --out "$dir/transcript_diff" || true
    if [ "${LMA_GRADE:-0}" = 1 ]; then
      python src/eval/grade_impact.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir "$mdir" \
        --grader hf-text:Qwen/Qwen2.5-7B-Instruct --name "detect_${name}_${tag}_${v}_qwen7b" --workers 8 || true
    fi
  done
}

python src/detect/test_detect.py
# the reader silently falls back to CPU without a GPU (~35 s/page): stop instead
python -c "
import sys, torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
d = $READER_DEVICE
if d >= n:
    sys.exit(f'GPU {d} not available (CUDA sees {n} GPU(s)); check nvidia-smi / CUDA_VISIBLE_DEVICES, or set LMA_READER_DEVICE=0')
print(f'reader on GPU {d}: {torch.cuda.get_device_name(d)}')
"
run qwen3b_ft "$QWEN_ROOT" qwen3b_ft ftreader --adapter "$ADAPTER"
# run gemini25 "$GEMINI_ROOT" gemini-2.5-pro ftreader --adapter "$ADAPTER"
run qwen3b_ft "$QWEN_ROOT" qwen3b_ft basereader
# run gemini25 "$GEMINI_ROOT" gemini-2.5-pro basereader
echo "=== done: $OUT/*/{detection,selector,verify,correct}_metrics.json, $OUT/*/transcript_diff/"
