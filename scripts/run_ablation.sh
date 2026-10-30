#!/usr/bin/env bash
# Ablations #2 #3 #6 #7 #9 #10 #16 on a SUBSET of exams (default 80) on ADA (2 GPUs). See docs/ABLATION.md.
#
#   bash scripts/run_ablation.sh                         # every stage, in order
#   STAGES="verify ablate" bash scripts/run_ablation.sh  # only some stages
#
# Stages (each is resumable: re-running skips finished work)
#   subset   pick N exams usable by both transcribers            CPU
#   agents   per transcriber: interactive (both agents), solo_advocate, solo_auditor (auto: 7B on GPU0,
#            reader on GPU1), oneshot claims and selfcorrect (7B on both GPUs)
#   verify   re-read every claim with 4 reader configs {ft,base} x {masked,unmasked}   GPU0 + GPU1 in parallel
#   ablate   decide (interaction stats) + Arbiter for every evidence mode x theta x reader   CPU
#   score    transcript_diff (CPU) + grade_impact (7B grader on both GPUs) for every variant
#   collect  results/ablation/ABLATION_RESULTS.md                                    CPU
#
# Overrides (env vars):
#   LMA_N=80                      exams in the subset (100 is fine; 293 = everything)
#   LMA_TRANSCRIBERS="qwen3b_ft gemini25"
#   LMA_CONCURRENCY=6 LMA_THETA=0.85 LMA_REREAD_MAX_PIXELS=1003520
#   LMA_DATA / LMA_QWEN_ROOT / LMA_ADAPTER / LMA_OUT / LMA_VENV   paths
#   LMA_LLM / LMA_READER / LMA_GRADER   models ("mock" for an offline plumbing test, see docs/ABLATION.md)
#   LMA_GI_PREFIX=abl             prefix of the results/grade_impact/<prefix>_* folders (set it for dry runs)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${LMA_VENV:-$ROOT/../venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

DATA="${LMA_DATA:-$HOME/Correction-Inception/lma_data}"
EDU="${LMA_EDU:-$DATA/edu_text/edu_data}"
IMAGES="${LMA_IMAGES:-$EDU}"   # page images live in the dataset itself
QWEN_ROOT="${LMA_QWEN_ROOT:-local_results/results/Outputs/qwen3b_ft}"
GEMINI_ROOT="${LMA_GEMINI_ROOT:-$EDU/Observationset_Final/v6_Gemini_2p5}"
ADAPTER="${LMA_ADAPTER:-local_results/results/reader_lora}"
OUT="${LMA_OUT:-results/ablation}"
N="${LMA_N:-80}"
TRANSCRIBERS="${LMA_TRANSCRIBERS:-qwen3b_ft gemini25}"
CONCURRENCY="${LMA_CONCURRENCY:-6}"
THETA="${LMA_THETA:-0.85}"
MAX_PIXELS="${LMA_REREAD_MAX_PIXELS:-1003520}"   # 1280*28*28; lower it (e.g. 602112) if re-reads hit CUDA OOM
LLM="${LMA_LLM:-hf-text:Qwen/Qwen2.5-7B-Instruct}"
READER="${LMA_READER:-hf:Qwen/Qwen2.5-VL-3B-Instruct}"
GRADER="${LMA_GRADER:-hf-text:Qwen/Qwen2.5-7B-Instruct}"
STAGES="${STAGES:-subset agents verify ablate score collect}"
GI_PREFIX="${LMA_GI_PREFIX:-abl}"   # grade_impact always writes results/grade_impact/<name>: use another prefix for dry runs
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
SUBSET="$OUT/subset_items.csv"

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date)  stages: $STAGES"
echo "=== data: $EDU   qwen: $QWEN_ROOT   gemini: $GEMINI_ROOT   adapter: $ADAPTER"
echo "=== out: $OUT   N=$N   transcribers: $TRANSCRIBERS   log: $LOG"

[ -d "$EDU/Screenshot_output_anon" ] || { echo "!!! dataset missing: $EDU"; exit 1; }
[ "$READER" = mock ] || [ -d "$ADAPTER" ] || { echo "!!! adapter missing: $ADAPTER"; exit 1; }

target_of() {   # transcriber -> "ROOT MODEL_DIR"
    case "$1" in
        qwen3b_ft) echo "$QWEN_ROOT qwen3b_ft" ;;
        gemini25)  echo "$GEMINI_ROOT gemini-2.5-pro" ;;
        *) echo "!!! unknown transcriber $1" >&2; exit 1 ;;
    esac
}

ra() {   # ra TRANSCRIBER RUN PHASE [extra args...]  -> src/agents/run_agents.py with the shared arguments
    local T="$1" RUN="$2" PH="$3"; shift 3
    local TROOT MDIR; read -r TROOT MDIR <<< "$(target_of "$T")"
    python src/agents/run_agents.py "$PH" --target-root "$TROOT" --model-dir "$MDIR" --name "$RUN" --out "$OUT/$T" \
        --edu-data "$EDU" --images "$IMAGES" --items "$SUBSET" --llm "$LLM" --reader "$READER" --theta "$THETA" \
        --max-claims 3 --tool-budget 4 --workers 8 --reread-max-pixels "$MAX_PIXELS" "$@"
}

has() { [[ " $STAGES " == *" $1 "* ]]; }

# ------------------------------------------------------------------ subset
if has subset; then
    if [ -f "$SUBSET" ]; then
        echo "=== subset exists ($(($(wc -l < "$SUBSET") - 1)) exams): $SUBSET  (delete it to re-sample)"
    else
        TARGETS=()
        for T in $TRANSCRIBERS; do read -r TROOT MDIR <<< "$(target_of "$T")"; TARGETS+=(--target "$TROOT:$MDIR"); done
        python scripts/make_ablation_subset.py --edu-data "$EDU" --n "$N" --out "$SUBSET" "${TARGETS[@]}"
    fi
fi
[ -f "$SUBSET" ] || { echo "!!! no subset file: run the subset stage first"; exit 1; }

# ------------------------------------------------------------------ agents (#9 #10 #16)
if has agents; then
    for T in $TRANSCRIBERS; do
        AUTO=(--adapter "$ADAPTER" --agent-device 0 --reader-device 1 --concurrency "$CONCURRENCY")
        echo "=== [$T] interactive: both agents, 3 rounds"
        HF_TEXT_DEVICES=0 ra "$T" interactive auto "${AUTO[@]}"
        echo "=== [$T] solo_advocate / solo_auditor: one agent alone (#10)"
        HF_TEXT_DEVICES=0 ra "$T" solo_advocate auto --agents advocate "${AUTO[@]}"
        HF_TEXT_DEVICES=0 ra "$T" solo_auditor auto --agents auditor "${AUTO[@]}"
        echo "=== [$T] oneshot: claims without tools or interaction (#9)"
        HF_TEXT_DEVICES=0,1 ra "$T" oneshot claims
        echo "=== [$T] selfcorrect: one-pass LLM repair, baseline B2 (#9)"
        HF_TEXT_DEVICES=0,1 HF_MAX_NEW_TOKENS=3072 ra "$T" selfcorrect selfcorrect
    done
fi

# ------------------------------------------------------------------ verify (#2 #6 #7 evidence)
verify() {   # verify TRANSCRIBER RUN TAG   (runs on whatever GPU CUDA_VISIBLE_DEVICES exposes)
    local args=()
    case "$3" in
        ft_mask)     args=(--adapter "$ADAPTER") ;;
        base_mask)   args=() ;;
        ft_nomask)   args=(--adapter "$ADAPTER" --no-mask) ;;
        base_nomask) args=(--no-mask) ;;
    esac
    ra "$1" "$2" verify --verify-tag "$3" "${args[@]}"
}
if has verify; then
    (   # GPU 0: the main reader config for every run
        export CUDA_VISIBLE_DEVICES=0
        for T in $TRANSCRIBERS; do for RUN in interactive solo_advocate solo_auditor oneshot; do
            verify "$T" "$RUN" ft_mask
        done; done
    ) & P0=$!
    (   # GPU 1: the reader ablations, on the interactive run only
        export CUDA_VISIBLE_DEVICES=1
        for T in $TRANSCRIBERS; do for TAG in base_mask ft_nomask base_nomask; do
            verify "$T" interactive "$TAG"
        done; done
    ) & P1=$!
    FAIL=0; wait $P0 || FAIL=1; wait $P1 || FAIL=1
    [ $FAIL = 0 ] || { echo "!!! a verify job failed (see log); re-run STAGES=verify to resume"; exit 1; }
fi

# ------------------------------------------------------------------ ablate (#2 #3 #6 #7 #10)
if has ablate; then
    for T in $TRANSCRIBERS; do
        for RUN in interactive solo_advocate solo_auditor; do
            rm -f "$OUT/$T/$RUN/interactive_ledger.jsonl"   # decide appends to it
            echo "=== [$T] $RUN: decide (interaction + agent-tool stats, Outputs/agents = as-built system)"
            ra "$T" "$RUN" decide --adapter "$ADAPTER" > "$OUT/$T/$RUN/decide_stdout.json"
        done
        for RUN in interactive solo_advocate solo_auditor oneshot; do
            echo "=== [$T] $RUN: ablate"
            ra "$T" "$RUN" ablate --main-tag ft_mask --write-thetas 0.5,0.7,0.95
        done
    done
fi

# ------------------------------------------------------------------ score
score() {   # score TRANSCRIBER RUN VARIANT_DIR LABEL
    local T="$1" RUN="$2" VDIR="$3" V="$4" TROOT MDIR
    read -r TROOT MDIR <<< "$(target_of "$T")"
    local NAME="${GI_PREFIX}_${T}_${RUN}__${V}"
    [ -d "$VDIR" ] || { echo "--- skip $NAME (no $VDIR)"; return; }
    if [ ! -f "$OUT/$T/$RUN/transcript_diff/$NAME/metrics.json" ]; then
        python src/eval/transcript_diff.py --data-root "$EDU" --target-root "$VDIR" --target-model-dir "$MDIR" \
            --name "$NAME" --out "$OUT/$T/$RUN/transcript_diff" > /dev/null || echo "!!! transcript_diff failed: $NAME"
    fi
    if [ ! -f "results/grade_impact/$NAME/metrics.json" ]; then
        echo "--- grade_impact $NAME"
        HF_TEXT_DEVICES=0,1 python src/eval/grade_impact.py --data-root "$EDU" --target-root "$VDIR" \
            --target-model-dir "$MDIR" --grader "$GRADER" --name "$NAME" --workers 8 || echo "!!! grade_impact failed: $NAME"
    fi
}
if has score; then
    for T in $TRANSCRIBERS; do
        D="$OUT/$T"
        # interactive: every variant (unchanged exams hit the grading cache, so this is cheap)
        for VDIR in "$D"/interactive/variants/*/; do score "$T" interactive "${VDIR%/}" "$(basename "$VDIR")"; done
        score "$T" interactive "$D/interactive/Outputs/agents" agenttools
        for RUN in solo_advocate solo_auditor; do score "$T" "$RUN" "$D/$RUN/variants/full__ft_mask" full__ft_mask; done
        score "$T" oneshot "$D/oneshot/variants/full__ft_mask" full__ft_mask
        score "$T" oneshot "$D/oneshot/variants/noverify" noverify
        score "$T" selfcorrect "$D/selfcorrect/Outputs/selfcorrect" selfcorrect
    done
fi

# ------------------------------------------------------------------ collect
if has collect; then
    python scripts/collect_ablation.py --out "$OUT" --prefix "$GI_PREFIX" --transcribers $TRANSCRIBERS
fi
echo "=== done $(date)"
