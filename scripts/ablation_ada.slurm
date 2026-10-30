#!/bin/bash
# SLURM job for the ablations (scripts/run_ablation.sh) on ADA. Submit from the repo root:
#   sbatch scripts/ablation_ada.slurm
#   LMA_N=100 sbatch scripts/ablation_ada.slurm                 # bigger subset
#   STAGES="verify ablate score collect" sbatch scripts/ablation_ada.slurm   # resume from a stage
# Every stage is resumable: if the job hits the time limit, submit it again.
#SBATCH -A research
#SBATCH -J ablation
#SBATCH -c 18
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH --time=2-00:00:00
#SBATCH -o ablation_%j.out
#SBATCH -e ablation_%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "Start: $(date)  node: $(hostname)  dir: $PWD"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
bash scripts/run_ablation.sh
echo "End: $(date)"
