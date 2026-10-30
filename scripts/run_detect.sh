#!/usr/bin/env bash
# Misread detection on ADA: does our reader's token confidence find the numbers a transcriber misread?
# For each transcriber x reader: labels (CPU) -> teacher-forced token scores (GPU) -> detection metrics (CPU)
# -> forced-choice verification of the flagged numbers (GPU).
#
#   bash scripts/run_detect.sh
#
# Transcribers: our fine-tuned 3B (qwen3b_ft) and the released Gemini 2.5 Pro transcripts.
# Readers: fine-tuned (adapter) and the untuned base model (ablation: does fine-tuning help detection?).
# Resumable: scoring and verification skip pages already done; labels and metrics are simply re-run.
#
# Overrides (env vars):
#   LMA_LIMIT=10                     only the first 10 pages per run (smoke)
#   LMA_READER_DEVICE=1              GPU for the reader (scoring and verification; one GPU is enough)
#   LMA_VERIFY_K=3                   forced-choice check of the top-k numbers per page (0 = skip)
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
ADAPTER="${LMA_ADAPTER:-local_results/results/reader_lora}"
OUT="${LMA_OUT:-results/detect}"
LIMIT="${LMA_LIMIT:+--limit $LMA_LIMIT}"
READER_DEVICE="${LMA_READER_DEVICE:-1}"
VERIFY_K="${LMA_VERIFY_K:-3}"
export HF_HOME="${HF_HOME-/scratch/jayanth.raju/hf_cache}"

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

run() {  # name target_root model_dir reader_tag [adapter args...]
  local name="$1" root="$2" mdir="$3" tag="$4"; shift 4
  local dir="$OUT/${name}_${tag}"
  echo "=== $dir: transcripts $root ($mdir), reader $tag"
  python src/detect/labels.py --edu-data "$EDU" --target-root "$root" --model-dir "$mdir" --out "$dir"
  python src/detect/score_tokens.py --edu-data "$EDU" --images "$IMAGES" --prompt-file "$PROMPT" --target-root "$root" --model-dir "$mdir" --out "$dir" \
    --reader hf:Qwen/Qwen2.5-VL-3B-Instruct --device "$READER_DEVICE" "$@" $LIMIT
  python src/detect/evaluate_detection.py --dir "$dir"
  if [ "$VERIFY_K" != 0 ]; then  # forced-choice A/B on the flagged numbers (same reader)
    python src/detect/verify_choice.py --dir "$dir" --edu-data "$EDU" --images "$IMAGES" --target-root "$root" \
      --model-dir "$mdir" --reader hf:Qwen/Qwen2.5-VL-3B-Instruct --device "$READER_DEVICE" --k "$VERIFY_K" "$@" $LIMIT
  fi
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
# run gemini25 "$EDU/Observationset_Final/v6_Gemini_2p5" gemini-2.5-pro ftreader --adapter "$ADAPTER"
run qwen3b_ft "$QWEN_ROOT" qwen3b_ft basereader
# run gemini25 "$EDU/Observationset_Final/v6_Gemini_2p5" gemini-2.5-pro basereader
echo "=== done: $OUT/*/detection_metrics.json, $OUT/*/verify_metrics.json"
