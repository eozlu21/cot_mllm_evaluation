#!/bin/bash
#SBATCH --job-name=cot-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu            # adjust to your cluster’s GPU queue
#SBATCH --gres=gpu:a100:1           # 1× A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# 1  Activate pre‑created micromamba env
source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate mllm_verifier_env

# 2  Move to project root (edit the path!)
cd cot_mllm_evaluation

# 3  Kick off the evaluation – arguments may be overridden by sbatch --export
srun python -m cot_mllm_evaluation \
    --dataset jmhessel/newyorker_caption_contest \
    --mllm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --judge_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --fewshot 1