#!/usr/bin/env bash
# Parallel download of EDU-CIRCUIT-HW_v1.zip (~9 GB).
#   bash scripts/download_edu_zip.sh [OUT_FILE] [N_CHUNKS]
#
# Strategy:
#   1. aria2c with 16 connections if available (best; apt-get install aria2)
#   2. N parallel curl range-request chunks (each chunk independently resumable)
#   3. plain single-stream resumable curl if the server ignores ranges
#
# Safe to re-run: it resumes instead of restarting.
set -euo pipefail

ZIP_URL="https://repository.gatech.edu/server/api/core/bitstreams/09496897-10b3-4675-bcf5-73b5b9802cd2/content"

OUT="${1:-${DATA_DIR:-$HOME/Desktop/Correction-Inception/lma_data}/EDU-CIRCUIT-HW_v1.zip}"
N="${2:-8}"
TMP="${OUT}.parts"

mkdir -p "$(dirname "$OUT")" "$TMP"

fail() { echo "!!! $*" >&2; exit 1; }

# ---------- 1) aria2c: 16 connections, resumable ----------
if command -v aria2c >/dev/null 2>&1; then
    echo "=== parallel download with aria2c (16 connections)"
    aria2c -x 16 -s 16 -k 1M --file-allocation=none -c \
        --console-log-level=warn --summary-interval=0 \
        -d "$(dirname "$OUT")" -o "$(basename "$OUT")" "$ZIP_URL"
    exit 0
fi

echo "=== aria2c not found; falling back to parallel curl chunks"
echo "    (tip: 'sudo apt-get install -y aria2' is faster on a 9 GB file)"

# ---------- determine size + range support via a 1-byte ranged HEAD ----------
HEAD_OUT="$(curl -sIL -r 0-0 "$ZIP_URL" | tr -d '\r')"
SIZE="$(awk 'tolower($1)=="content-range:"{split($2,a,"/"); n=a[2]}
             tolower($1)=="content-length:"{cl=$2}
             END{ if (n != "" && n > 0) print n; else print cl }' <<<"$HEAD_OUT")"

[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] 2>/dev/null || fail "could not determine file size"

if ! grep -qi '^content-range:' <<<"$HEAD_OUT"; then
    echo "=== server does not support ranges; single-stream resumable download"
    curl -L -C - -o "$OUT" "$ZIP_URL"
    exit 0
fi

echo "=== total size: $SIZE bytes, splitting into $N chunks"

# ---------- 2) N parallel resumable chunk downloads ----------
CHUNK=$(( (SIZE + N - 1) / N ))   # ceil; last chunk may be smaller
pids=()

for i in $(seq 0 $((N - 1))); do
    start=$(( i * CHUNK ))
    [ "$start" -ge "$SIZE" ] && break
    end=$(( start + CHUNK - 1 ))
    [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
    want=$(( end - start + 1 ))

    part="$TMP/part_$(printf '%03d' "$i")"
    have=0
    [ -f "$part" ] && have=$(stat -c%s "$part")
    [ "$have" -ge "$want" ] && { echo "chunk $i: already complete"; continue; }

    echo "chunk $i: bytes $((start + have))-$end  ($(( want - have )) to go)"
    (
        for _ in 1 2 3; do
            have=0
            [ -f "$part" ] && have=$(stat -c%s "$part")
            [ "$have" -ge "$want" ] && exit 0
            if curl -sL --fail -r "$((start + have))-$end" "$ZIP_URL" >> "$part"; then
                exit 0
            fi
            sleep 2
        done
        exit 1
    ) &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
[ "$rc" -eq 0 ] || fail "one or more chunk downloads failed; re-run to resume"

# ---------- verify + concatenate ----------
TOTAL=0
for part in "$TMP"/part_*; do
    TOTAL=$(( TOTAL + $(stat -c%s "$part") ))
done
[ "$TOTAL" -eq "$SIZE" ] || fail "size mismatch after download ($TOTAL != $SIZE); re-run to resume"

echo "=== assembling $OUT"
cat "$TMP"/part_* > "$OUT"
rm -rf "$TMP"
echo "=== done: $OUT ($SIZE bytes)"