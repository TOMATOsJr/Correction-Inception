#!/bin/bash
#SBATCH -A research
#SBATCH -J "correction"
#SBATCH -c 12
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH -o "output.txt"
#SBATCH -e "output.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prabhavsai.kollipara@research.iiit.ac.in
#SBATCH -w gnode057

cd "/home2/prabhavsai.kollipara/Correction-Inception/"

source /home2/prabhavsai.kollipara/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

CUDA_VISIBLE_DEVICES=0 python3 scripts/run_correction.py --data . --experiments topk3 \
    --max-pixels 600000 --out results/correction_600k > gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_correction.py --data . --experiments crop \
    --max-pixels 600000 --out results/correction_600k_crop > gpu1.log 2>&1 &
wait

echo "Time at exit: $(date)"
