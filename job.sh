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
#
# AFTER THE SMOKE: the full runs as two jobs on two nodes (every stage resumes: re-submit the same line to continue)
#   job A (~4-6 h): A2 blank image on the EDU transcripts, A1 Auditor fix on FERMAT, C1 agents on the 10-per-page run
#     STAGES="edu_silent_blank audit_fix agents" sbatch -J ckA -w gnodeXX -o out_A.txt -e out_A.txt job.sh
#   job B (~12-16 h): B2 the checker on 20 numbers per page, then consistency + Arbiter on every run, agents on 20/page
#     STAGES="scale consistency arbiter agents" LMA_K=20 LMA_CHECKER=checker_fe \
#       LMA_AGENTS_RUN=correct_checker_fe_k20_beams sbatch -J ckB -w gnodeYY -o out_B.txt -e out_B.txt job.sh
#   the agents stage keeps the 7B in the node's /scratch (downloaded once per node, ~15 GB)
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
