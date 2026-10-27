#!/usr/bin/env bash
# One-time setup on a Linux GPU cluster (e.g. ADA: 2080 Ti, 11 GB each).
#   bash scripts/setup_ada.sh [DATA_DIR] [CUDA_TAG]
#   DATA_DIR   where the dataset lives (default: $HOME/lma_data)
#   CUDA_TAG   PyTorch wheel tag: cu121 (default, safest for Turing) | cu124 | cu128
#
# Layout created (what scripts/run_all.py --data expects):
#   $DATA/edu_text/edu_data  <- extracted EDU-CIRCUIT-HW zip (Screenshot_output_anon, ...)
#   $DATA/edu_text/edu_repo  <- the paper's GitHub repo
#   $DATA/images             -> symlink to edu_data
#
# HF_TOKEN: put "HF_TOKEN=hf_..." in .env in the project root yourself (never paste
# it into chat). The account must have accepted FERMAT's terms.
set -euo pipefail

DATA="${1:-$HOME/lma_data}"
CUDA="${2:-cu121}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
ZIP_URL="https://repository.gatech.edu/server/api/core/bitstreams/09496897-10b3-4675-bcf5-73b5b9802cd2/content"

echo "=== project root: $ROOT"
echo "=== data dir:     $DATA"

# 1) venv with the pinned GPU stack (PyTorch FIRST, then the pinned list)
if [ ! -x "$PY" ]; then
    echo "=== creating venv (python3 must be 3.10-3.12)"
    python3 -m venv "$ROOT/venv"
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
"$PY" -m pip install -q -r "$ROOT/requirements-gpu.txt"
"$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, \
'cuda avail:', torch.cuda.is_available())"

# 2) dataset zip (resumable: re-run this script and it continues)
mkdir -p "$DATA/edu_text"
if [ ! -f "$DATA/edu_text/edu_data/Screenshot_output_anon/set_splitting/obsetf_involved_data.csv" ]; then
    echo "=== downloading EDU-CIRCUIT-HW_v1.zip (~9 GB, resumes on re-run)"
    curl -L -C - -o "$DATA/EDU-CIRCUIT-HW_v1.zip" "$ZIP_URL"
    echo "=== extracting"
    mkdir -p "$DATA/edu_text/edu_data"
    unzip -q -o "$DATA/EDU-CIRCUIT-HW_v1.zip" -d "$DATA/edu_text/edu_data"
    # the zip may nest everything in a single top folder; flatten if so
    if [ ! -d "$DATA/edu_text/edu_data/Screenshot_output_anon" ]; then
        mv "$DATA/edu_text/edu_data"/*/* "$DATA/edu_text/edu_data/" 2>/dev/null || true
    fi
    rm -f "$DATA/EDU-CIRCUIT-HW_v1.zip"   # 9 GB free again
fi

# 3) the paper's repo (edu_repo)
if [ ! -d "$DATA/edu_text/edu_repo/.git" ]; then
    git clone --depth 1 https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL "$DATA/edu_text/edu_repo"
fi

# 4) images path = the same extracted data
[ -e "$DATA/images" ] || ln -s "$DATA/edu_text/edu_data" "$DATA/images"

# 5) HF token present? (checked, never printed)
if [ -f "$ROOT/.env" ] && grep -q "^HF_TOKEN=" "$ROOT/.env"; then
    echo "=== .env with HF_TOKEN found"
else
    echo "!!! no HF_TOKEN in $ROOT/.env — FERMAT is gated."
    echo "!!! Create .env yourself with one line:  HF_TOKEN=hf_..."
    echo "!!! (account must have accepted the terms at huggingface.co/datasets/ai4bharat/FERMAT)"
    echo "!!! Setup will finish; the preflight will fail until the token exists."
fi

# 6) preflight (downloads models ~23 GB + FERMAT ~5 GB; needs the token)
echo "=== preflight"
"$PY" "$ROOT/scripts/run_all.py" --data "$DATA" --preflight && echo "SETUP OK" || echo "SETUP FINISHED WITH FAILURES (see above)"
