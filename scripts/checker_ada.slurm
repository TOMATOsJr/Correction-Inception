#!/bin/bash
# SLURM job for the A/B checker (scripts/run_checker.sh) on ADA. Submit from the repo root:
#   LMA_SMOKE=1 sbatch scripts/checker_ada.slurm        # first: ~20-30 min, every stage on a few examples
#   sbatch scripts/checker_ada.slurm                    # the real run
#   STAGES="eval correct" sbatch scripts/checker_ada.slurm   # resume from a stage
# Every stage is resumable: if the job hits the time limit, submit it again.
#SBATCH -A research
#SBATCH -J checker
#SBATCH -c 18
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH --time=2-00:00:00
#SBATCH -o checker_%j.out
#SBATCH -e checker_%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "Start: $(date)  node: $(hostname)  dir: $PWD"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
bash scripts/run_checker.sh
echo "End: $(date)"
