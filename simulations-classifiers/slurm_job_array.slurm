#!/bin/bash
#SBATCH --job-name=multi_gpu_tasks
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gpus=a30:1
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-19%8

CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" commands.txt)
echo "Running task $SLURM_ARRAY_TASK_ID: $CMD"
eval $CMD
