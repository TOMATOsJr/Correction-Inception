#!/usr/bin/env bash
# Fine-tuned A/B checker on ADA: build pairs -> train -> measure -> use it in the correction pipeline.
# Why and what: docs/CHECKER.md
#
#   bash scripts/run_checker.sh                 # everything (on ADA: sbatch job.sh)
#   STAGES="eval" bash scripts/run_checker.sh   # only some stages
#
# Stages (each skips finished work, so re-running after a crash or time-out continues):
#   test     unit tests (CPU, seconds)
#   pairs    A/B pairs from FERMAT (downloads it once, needs HF_TOKEN in .env) and from EDU (labels.csv + transcripts)
#   train    checker_f (FERMAT pairs) and checker_fe (FERMAT + 3 EDU students), both starting from our reader's LoRA
#   eval     base model, reader LoRA, checker_f, checker_fe on held-out pairs -> results/checker/eval/summary.json
#   correct  the top-3 most doubtful numbers per page checked by the reader (old way) and by the best checker,
#            on students the checker never saw -> change / keep decisions -> transcript diff [+ grades: LMA_GRADE=1]
#   scale    (not in the default list: STAGES=scale) the trained checker on the K most doubtful numbers per page with
#            the reader's top-k whole-number beam candidates -> decisions -> what-if sweep -> transcript diff
#   arbiter  (STAGES=arbiter, CPU) the learned Arbiter over all tool evidence for every correct_* run
#   clean_eval (STAGES=clean_eval) every model re-measured on FERMAT pairs without the red error markers
#   audit    (STAGES=audit) the Auditor's silent-correction detectors on FERMAT, exact labels [LMA_AUDIT_FULL=1: + fixes]
#   consistency (CPU) B1: challenger / written value elsewhere on the page, decision rules on held-out students
#   edu_silent  (CPU) A2: EDU silent corrections via the answer key, our reader and Gemini
#   edu_silent_blank (GPU) A2: blank-image scores of the EDU transcripts + the blank-image signal on them
#   audit_fix   (GPU) A1: Auditor fixes on FERMAT, numbers chosen by the blank-image signal
#   agents      (GPU) C1: Advocate + Auditor as tool-calling LLM agents over the flagged numbers, vs no agents
#
# Overrides (env vars):
#   STAGES="test pairs train eval correct"
#   LMA_SMOKE=1                  tiny run of every stage (~20 min) to catch path problems: results/checker_smoke
#   LMA_TRAIN_STUDENTS=3         EDU students used to train checker_fe (and left out of every evaluation)
#   LMA_MAX_EXAMPLES=6000        training examples per checker
#   LMA_CHECKER=auto             checker used in 'correct': auto (best edu/misread accuracy) | checker_f | checker_fe
#   LMA_CORRECT_BASELINE=1       0 = skip the reader-as-checker baseline in 'correct' (saves half the time)
#   LMA_GRADE=0                  1 = also grade none / corrected with the 7B grader (slow)
#   LMA_K=10 LMA_CANDS=4         scale: numbers checked per page, candidates per number (beams first)
#   LMA_SCALE_LIMIT=5            scale: quick test on 5 pages per GPU shard, in correct_<tag>_try (delete it after)
#   LMA_LIMIT=3                  quick check of the GPU stages on 3 pages per shard, in the real folders (resumable)
#   LMA_DATA (default: repo root; everything else is found inside it) / LMA_EDU / LMA_IMAGES / LMA_QWEN_ROOT /
#   LMA_ADAPTER / LMA_DETECT_DIR / LMA_OUT   path overrides
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -n "${VIRTUAL_ENV:-}" ] || source "${LMA_VENV:-$ROOT/../venv}/bin/activate"

# Paths are found by contents under LMA_DATA (default: the repo root, as job.sh / run_correction.py --data .)
DATA="${LMA_DATA:-$ROOT}"
eval "$(python - "$DATA" <<'PY'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from run_correction import resolve_data
p = resolve_data(Path(sys.argv[1]))
for k, v in [("FOUND_EDU", p["edu"]), ("FOUND_IMAGES", p["images"]), ("FOUND_QWEN", p["qwen"]), ("FOUND_LORA", p["lora"])]:
    print(f"{k}='{v or ''}'")
PY
)"
EDU="${LMA_EDU:-$FOUND_EDU}"
IMAGES="${LMA_IMAGES:-$FOUND_IMAGES}"
QWEN_ROOT="${LMA_QWEN_ROOT:-$FOUND_QWEN}"
ADAPTER="${LMA_ADAPTER:-$FOUND_LORA}"
PROMPT=src/eval/prompts/Initial_prompt_wo_problem_statement_v6.txt
SMOKE="${LMA_SMOKE:-0}"
# (not $([ .. ] && echo ..): when the test is false that assignment fails and set -e exits without a word)
if [ "$SMOKE" = 1 ]; then OUT="${LMA_OUT:-results/checker_smoke}"; else OUT="${LMA_OUT:-results/checker}"; fi
# labels.csv + scores.csv of the Qwen transcripts: the 600k-pixel scores of the last correction run (job.sh) when
# present, so the same numbers are checked as before; made here if missing. The smoke run keeps its own copy.
if [ -n "${LMA_DETECT_DIR:-}" ]; then DETECT_DIR="$LMA_DETECT_DIR"
elif [ "$SMOKE" = 1 ]; then DETECT_DIR="$OUT/detect"
elif [ -f results/correction_600k/qwen_ftreader/scores.csv ]; then DETECT_DIR=results/correction_600k/qwen_ftreader
else DETECT_DIR=results/detect/qwen3b_ft_ftreader; fi
for v in EDU IMAGES QWEN_ROOT ADAPTER; do
  [ -n "${!v}" ] && [ -e "${!v}" ] || { echo "missing $v (looked under $DATA); set LMA_$v" >&2; exit 1; }
done
STAGES="${STAGES:-test pairs train eval correct}"
TRAIN_STUDENTS="${LMA_TRAIN_STUDENTS:-3}"
MAX_EXAMPLES="${LMA_MAX_EXAMPLES:-6000}"
PIX=602112   # 768*28*28, the resolution of the earlier local correction run; the checker is trained at the same
# HF_HOME: left as is (the default ~/.cache/huggingface already holds the models from earlier jobs)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
if [ "$SMOKE" = 1 ]; then
  MAX_EXAMPLES=40; PER_KIND=10; LIMIT="--limit 4"; EPOCHS=1
else
  PER_KIND=400; LIMIT=""; EPOCHS=1
fi
# LMA_LIMIT=N: the GPU stages on N pages / samples per GPU shard, in the REAL output folders (unlike LMA_SMOKE). A quick
# check before a long run; every stage resumes, so the full run then continues from these pages.
if [ -n "${LMA_LIMIT:-}" ]; then LIMIT="--limit $LMA_LIMIT"; fi

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== stages: $STAGES | out: $OUT | HF_HOME: ${HF_HOME:-default}"
echo "=== edu: $EDU | images: $IMAGES | transcripts: $QWEN_ROOT | reader LoRA: $ADAPTER | scores: $DETECT_DIR"

NGPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
has() { [[ " $STAGES " == *" $1 "* ]]; }
need_gpu() {
  if [ "$NGPU" -lt 1 ]; then
    echo "no GPU visible (CUDA sees 0): stopping instead of running on the CPU. Check nvidia-smi / --gres." >&2
    exit 1
  fi
}
# run two commands on two GPUs at once if there are two, else one after the other. Each process is pinned with
# CUDA_VISIBLE_DEVICES and sees its GPU as device 0 (the Trainer ignored --device 1 and put both on GPU 0).
pair_run() {  # "cmd0" "cmd1" (each gets --device 0 appended)
  if [ "$NGPU" -ge 2 ]; then
    eval "CUDA_VISIBLE_DEVICES=0 $1 --device 0" & local p0=$!
    eval "CUDA_VISIBLE_DEVICES=1 $2 --device 0" & local p1=$!
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
# ---------------------------------------------------------------- scale (STAGES=scale; no training)
# The trained checker on more numbers and better options: the K most doubtful numbers per page (not 3), with the
# reader's top-k whole-number beam candidates first (they can differ in several digits, the decimal point or the
# length), then the old digit candidates. Then change / keep decisions, the what-if sweep and the transcript diff.
if has scale; then
  need_gpu
  K="${LMA_K:-10}"; CANDS="${LMA_CANDS:-4}"
  CHECKER="${LMA_CHECKER:-auto}"
  [ "$CHECKER" = auto ] && CHECKER=$(python src/finetune/checker_table.py --eval "$OUT/eval" --best)
  tag="${CHECKER}_k${K}_beams"
  SLIMIT="$LIMIT"
  if [ -n "${LMA_SCALE_LIMIT:-}" ]; then SLIMIT="--limit $LMA_SCALE_LIMIT"; tag="${tag}_try"; fi  # quick test
  dir="$OUT/correct_$tag"
  mkdir -p "$dir"
  for src in "$DETECT_DIR"/labels.csv "$DETECT_DIR"/scores*.csv; do
    [ -e "$dir/$(basename "$src")" ] || ln -s "$(realpath "$src")" "$dir/"
  done
  echo "=== scale: $CHECKER on the $K most doubtful numbers per page, beam candidates; left out: ${TRAIN_LIST:-none}"
  common="--dir $dir --edu-data $EDU --images $IMAGES --target-root $QWEN_ROOT --model-dir qwen3b_ft --select topk \
    --k $K --reader hf:Qwen/Qwen2.5-VL-3B-Instruct --max-pixels $PIX --exclude-students '$TRAIN_LIST' $SLIMIT"
  bcmd="python src/detect/beam_candidates.py $common --adapter $ADAPTER --prompt-file $PROMPT --beams 6 --keep 4"
  vcmd="python src/detect/verify_choice.py $common --adapter $OUT/$CHECKER --beams --max-cands $CANDS"
  for cmd in "$bcmd" "$vcmd"; do
    if [ "$NGPU" -ge 2 ]; then pair_run "$cmd --shard 0/2" "$cmd --shard 1/2"; else eval "$cmd --device 0"; fi
  done
  python src/detect/correct.py --dir "$dir" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft
  python src/detect/sweep_correct.py --dir "$dir" || true
  for v in none corrected; do
    python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir qwen3b_ft \
      --name "${tag}_$v" --out "$dir/transcript_diff" || true
    if [ "${LMA_GRADE:-0}" = 1 ]; then
      python src/eval/grade_impact.py --data-root "$EDU" --target-root "$dir/Outputs/$v" --target-model-dir qwen3b_ft \
        --grader hf-text:Qwen/Qwen2.5-7B-Instruct --name "checker_${tag}_${v}_qwen7b" --workers 8 || true
    fi
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi
# ---------------------------------------------------------------- arbiter (STAGES=arbiter; CPU)
# The learned Arbiter (src/detect/arbiter.py) on every correction run that has verify*.csv: all tool evidence combined,
# student-held-out, threshold = most net fixes on the training students. Writes <run>/arbiter/ + transcript diff.
if has arbiter; then
  for dir in "$OUT"/correct_*; do
    ls "$dir"/verify*.csv >/dev/null 2>&1 || continue
    tag="${dir##*/correct_}"
    echo "=== arbiter: $tag"
    python src/detect/arbiter.py --dir "$dir" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft
    for v in none corrected; do
      python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$dir/arbiter/Outputs/$v" \
        --target-model-dir qwen3b_ft --name "${tag}_arbiter_$v" --out "$dir/arbiter/transcript_diff" || true
      if [ "${LMA_GRADE:-0}" = 1 ]; then
        python src/eval/grade_impact.py --data-root "$EDU" --target-root "$dir/arbiter/Outputs/$v" \
          --target-model-dir qwen3b_ft --grader hf-text:Qwen/Qwen2.5-7B-Instruct \
          --name "checker_${tag}_arbiter_${v}_qwen7b" --workers 8 || true
      fi
    done
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi

# ---------------------------------------------------------------- clean_eval (STAGES=clean_eval; GPU ~40 min)
# FERMAT's pert_a marks every planted error in red; the first pairs kept the marker in the shown line (46% of the
# silent test pairs had it right at [?]). Rebuild the FERMAT pairs without markers and re-measure every model.
if has clean_eval; then
  need_gpu
  CPAIRS="$OUT/pairs_clean"
  [ -f "$CPAIRS/pairs_fermat_test.jsonl" ] || python src/finetune/ab_data.py fermat --out "$CPAIRS"
  todo=()
  for spec in base: reader:$ADAPTER checker_f:$OUT/checker_f checker_fe:$OUT/checker_fe; do
    tag="${spec%%:*}"; ad="${spec#*:}"
    [ -f "$OUT/eval_clean/$tag/checker_metrics.json" ] && continue
    [ -n "$ad" ] && [ ! -f "$ad/adapter_config.json" ] && continue
    a2=""; [ -n "$ad" ] && a2="--adapter $ad"
    todo+=("python src/finetune/eval_checker.py --pairs $CPAIRS --sources fermat --tag $tag $a2 --max-pixels $PIX \
      --per-kind $PER_KIND --out $OUT/eval_clean")
  done
  while [ "${#todo[@]}" -ge 2 ]; do pair_run "${todo[0]}" "${todo[1]}"; todo=("${todo[@]:2}"); done
  [ "${#todo[@]}" = 1 ] && eval "${todo[0]} --device 0"
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi

# ---------------------------------------------------------------- audit (STAGES=audit; GPU ~20-30 min)
# The Auditor on FERMAT (exact labels): label the numbers of our base and fine-tuned readers' transcripts (ok / silent
# correction / misread), score them with the page and with a blank image, and measure the silent-correction detectors.
# LMA_AUDIT_FULL=1 also runs beams -> checker -> Arbiter on the fine-tuned reader's flagged numbers (few positives).
if has audit; then
  need_gpu
  for m in qwen3b_ft qwen3b_base; do
    pred="results/pc_run/results/fermat/$m/predictions.jsonl"
    [ -f "$pred" ] || { echo "=== audit: no $pred, skipped"; continue; }
    ad="$OUT/audit_fermat_$m"
    echo "=== audit: $m"
    [ -f "$ad/labels.csv" ] || python src/detect/fermat_audit.py label --pred "$pred" --out "$ad"
    acmd="python src/detect/fermat_audit.py score --out $ad --adapter $ADAPTER --max-pixels $PIX $LIMIT"
    if [ "$NGPU" -ge 2 ]; then pair_run "$acmd --shard 0/2" "$acmd --shard 1/2"; else eval "$acmd --device 0"; fi
    python src/detect/fermat_audit.py detect --out "$ad" --k 4
    if [ "${LMA_AUDIT_FULL:-0}" = 1 ] && [ "$m" = qwen3b_ft ]; then
      CK="${LMA_CHECKER:-checker_fe}"
      for sub in "beams --adapter $ADAPTER" "verify --adapter $OUT/$CK"; do
        c="python src/detect/fermat_audit.py $sub --out $ad --max-pixels $PIX $LIMIT"
        if [ "$NGPU" -ge 2 ]; then pair_run "$c --shard 0/2" "$c --shard 1/2"; else eval "$c --device 0"; fi
      done
      python src/detect/fermat_audit.py decide --out "$ad"
    fi
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi
# ---------------------------------------------------------------- consistency (STAGES=consistency; CPU, minutes)
# B1: does the challenger / the written value appear elsewhere on the page (the student's own work)? Main rule fixed in
# advance: change only when the challenger does and the written value does not. Every correct_* run with verify*.csv.
if has consistency; then
  for dir in "$OUT"/correct_*; do
    ls "$dir"/verify*.csv >/dev/null 2>&1 || continue
    tag="${dir##*/correct_}"
    echo "=== consistency: $tag"
    python src/detect/consistency.py --dir "$dir" --target-root "$QWEN_ROOT" --model-dir qwen3b_ft
    for v in none corrected; do
      python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$dir/consistency/Outputs/$v" \
        --target-model-dir qwen3b_ft --name "${tag}_consistency_$v" --out "$dir/consistency/transcript_diff" || true
    done
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi

# ---------------------------------------------------------------- edu_silent (STAGES=edu_silent; CPU, minutes)
# A2: silent corrections in the real exams, through the rubric's answer key, for our reader and Gemini 2.5 Pro.
GEMINI_ROOT="$EDU/Observationset_Final/v6_Gemini_2p5"
if has edu_silent; then
  python src/detect/edu_silent.py analyse --labels "$DETECT_DIR/labels.csv" --edu-data "$EDU" --name qwen3b_ft \
    --out "$OUT/edu_silent"
  if [ -d "$GEMINI_ROOT" ]; then
    gl="$OUT/edu_silent/gemini25"
    [ -f "$gl/labels.csv" ] || python src/detect/labels.py --edu-data "$EDU" --target-root "$GEMINI_ROOT" \
      --model-dir gemini-2.5-pro --out "$gl"
    python src/detect/edu_silent.py analyse --labels "$gl/labels.csv" --edu-data "$EDU" --name gemini25 --out "$OUT/edu_silent"
  fi
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi

# ---------------------------------------------------------------- edu_silent_blank (STAGES=edu_silent_blank; GPU ~30 min)
# A2, optional: score our reader's EDU transcripts with a blank image, then test the blank-image signal on them.
if has edu_silent_blank; then
  need_gpu
  bo="$OUT/edu_silent/qwen3b_ft"
  bcmd="python src/detect/edu_silent.py blank --edu-data $EDU --images $IMAGES --target-root $QWEN_ROOT \
    --model-dir qwen3b_ft --adapter $ADAPTER --prompt-file $PROMPT --max-pixels $PIX --out $bo $LIMIT"
  if [ "$NGPU" -ge 2 ]; then pair_run "$bcmd --shard 0/2" "$bcmd --shard 1/2"; else eval "$bcmd --device 0"; fi
  python src/detect/edu_silent.py analyse --labels "$DETECT_DIR/labels.csv" --edu-data "$EDU" --name qwen3b_ft \
    --out "$OUT/edu_silent" --blank --scores "$DETECT_DIR"
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi

# ---------------------------------------------------------------- audit_fix (STAGES=audit_fix; GPU ~1-2 h)
# A1: the Auditor's fix step on FERMAT, checking the 4 numbers per answer the blank-image signal finds most suspicious
# (it separated silent corrections best), then beams -> checker -> decision (simple rule and the Arbiter).
if has audit_fix; then
  need_gpu
  src="$OUT/audit_fermat_qwen3b_ft"
  ad="$OUT/audit_fix_qwen3b_ft"
  mkdir -p "$ad"
  pred="results/pc_run/results/fermat/qwen3b_ft/predictions.jsonl"
  [ -f "$src/labels.csv" ] || python src/detect/fermat_audit.py label --pred "$pred" --out "$src"
  if ! ls "$src"/scores*.csv >/dev/null 2>&1; then
    acmd="python src/detect/fermat_audit.py score --out $src --adapter $ADAPTER --max-pixels $PIX"
    if [ "$NGPU" -ge 2 ]; then pair_run "$acmd --shard 0/2" "$acmd --shard 1/2"; else eval "$acmd --device 0"; fi
  fi
  for f in "$src"/labels.csv "$src"/labels_summary.json "$src"/samples.jsonl "$src"/scores*.csv; do
    [ -e "$ad/$(basename "$f")" ] || ln -s "$(realpath "$f")" "$ad/"
  done
  python src/detect/fermat_audit.py detect --out "$ad" --k "${LMA_AUDIT_K:-4}" --select blank
  CK="${LMA_CHECKER:-checker_fe}"
  for sub in "beams --adapter $ADAPTER" "verify --adapter $OUT/$CK"; do
    c="python src/detect/fermat_audit.py $sub --out $ad --max-pixels $PIX $LIMIT"
    if [ "$NGPU" -ge 2 ]; then pair_run "$c --shard 0/2" "$c --shard 1/2"; else eval "$c --device 0"; fi
  done
  python src/detect/fermat_audit.py decide --out "$ad"
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi
# ---------------------------------------------------------------- agents (STAGES=agents; GPU, a few hours)
# C1: Advocate and Auditor as tool-calling LLM agents (Qwen2.5-7B, 4-bit) over the ledger of flagged numbers of one
# correction run; the tools answer from that run's evidence; a fixed rule decides. Compared with the same tools
# without agents on the same unseen students. LMA_AGENTS_RUN=<correct_* folder name>, LMA_AGENTS_LIMIT=5 = quick test.
if has agents; then
  need_gpu
  run="$OUT/${LMA_AGENTS_RUN:-correct_checker_fe_k10_beams}"
  [ -d "$run" ] || { echo "no $run: run STAGES=scale first" >&2; exit 1; }
  alim=""; [ -n "${LMA_AGENTS_LIMIT:-${LMA_LIMIT:-}}" ] && alim="--limit ${LMA_AGENTS_LIMIT:-$LMA_LIMIT}"
  python src/agents/ledger_agents.py --dir "$run" --target-root "$QWEN_ROOT" --edu-data "$EDU" \
    --llm hf-text:Qwen/Qwen2.5-7B-Instruct --workers "${LMA_AGENTS_WORKERS:-16}" $alim
  tag="${run##*/correct_}"
  for v in none corrected; do
    python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$run/agents/Outputs/$v" \
      --target-model-dir qwen3b_ft --name "${tag}_agents_$v" --out "$run/agents/transcript_diff" || true
  done
  python src/finetune/checker_table.py --eval "$OUT/eval" --correct "$OUT"
fi
echo "=== done. Results: $OUT/CHECKER_RESULTS.md"
