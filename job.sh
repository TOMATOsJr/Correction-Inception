#!/bin/bash
#SBATCH -A research
#SBATCH -J "checker"
#SBATCH -c 18
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH -o "output.txt"
#SBATCH -e "output.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prabhavsai.kollipara@research.iiit.ac.in
#SBATCH -w gnode080

# A/B checker: build pairs -> fine-tune -> measure -> correction with the checker (docs/CHECKER.md).
#   LMA_SMOKE=1 sbatch job.sh               smoke test first (~20-30 min): results/checker_smoke
#   sbatch job.sh                           the real run: results/checker/CHECKER_RESULTS.md
#   STAGES="eval correct" sbatch job.sh     only some stages (every stage resumes, so re-submitting continues)
#   LMA_GRADE=1 STAGES=correct sbatch job.sh   + grading, once the correction table shows a net gain
#   STAGES=scale sbatch job.sh              trained checker on 10 numbers/page + beam candidates (no training, ~5-6 h)
#   LMA_SCALE_LIMIT=5 STAGES=scale sbatch job.sh   quick test of that first (~20 min)
#   STAGES="arbiter clean_eval audit" sbatch job.sh   learned Arbiter (CPU) + FERMAT re-check without red markers
#                                           + the Auditor's silent-correction detectors on FERMAT (~1-1.5 h)
# run_checker.sh uses both GPUs itself (two trainings / evaluations / verify shards at once): no CUDA_VISIBLE_DEVICES.

cd "/home2/prabhavsai.kollipara/Correction-Inception/"

source /home2/prabhavsai.kollipara/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_TOKEN

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

LMA_DATA=. bash scripts/run_checker.sh

echo "Time at exit: $(date)"
