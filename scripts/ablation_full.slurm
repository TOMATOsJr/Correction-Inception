#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -J ablation
#SBATCH -c 18
#SBATCH --mem-per-cpu=2048
#SBATCH -G 2
#SBATCH --time=3-00:00:00
#SBATCH --exclude=gnode069
#SBATCH -o ablation_%j.out
set -e
cd ~/Correction-Inception
echo "Start: $(date)  node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
source ~/lma-venv311/bin/activate
export LMA_VENV=~/lma-venv311 LMA_DATA=~/Correction-Inception/data LMA_QWEN_ROOT=results/Outputs/qwen3b_ft LMA_ADAPTER=results/reader_lora
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen2.5-VL-3B-Instruct']]"
export HF_HUB_OFFLINE=1
bash scripts/run_ablation.sh
echo "End: $(date)"
