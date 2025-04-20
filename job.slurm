#!/bin/bash
#SBATCH -A users                    # account name
#SBATCH -p mid                      # partition / queue
#SBATCH --qos=users                 # quality of service
#SBATCH --gres=gpu:a100:1           # request 1 A100 GPU
#SBATCH -c 4                        # request 4 CPU cores
#SBATCH --mem=128G                  # request 128 GB RAM
#SBATCH --time=1-00:00:00           # max runtime (1 day)
#SBATCH -o results/results_%j.txt   # STDOUT log
#SBATCH -e errors/errors_%j.txt     # STDERR log
#SBATCH --mail-type=ALL             # email on start,end,fail
#SBATCH --mail-user=eozlu21@ku.edu.tr  # change to your e‑mail

# 1  Activate pre‑created micromamba env
source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate mllm_verifier_env

# 2  Move to project root (edit the path!)
cd cot_mllm_evaluation

# 3  Kick off the evaluation – arguments may be overridden by sbatch --export
srun python main.py \
    --dataset jmhessel/newyorker_caption_contest \
    --mllm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --judge_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --fewshot 1