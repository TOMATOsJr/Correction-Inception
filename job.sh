#!/bin/bash
#SBATCH -A research
#SBATCH -J "agent_interaction"
#SBATCH -c 18
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH -o "output.txt"
#SBATCH -e "output.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayanth.raju@research.iiit.ac.in
#SBATCH -w gnode079

cd "/home2/jayanth.raju/Correction-Inception/"

echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"

./scripts/run_agents_all.sh

echo "Time at exit: $(date)"
