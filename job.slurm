#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1                    # account name
#SBATCH --partition=ai                      # partition / queue
#SBATCH --gres=gpu:rtx_a6000:2         # request 1 A100 GPU
#SBATCH -c 4                        # request 4 CPU cores
#SBATCH --mem=128G                  # request 128 GB RAM
#SBATCH --time=1-00:00:00           # max runtime (1 day)
#SBATCH -o results/results_%j.txt   # STDOUT log
#SBATCH -e errors/errors_%j.txt     # STDERR log


# 3  Kick off the evaluation – arguments may be overridden by sbatch --export
srun python3 main.py --explanation_type uncanny

