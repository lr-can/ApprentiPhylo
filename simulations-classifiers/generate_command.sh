find config -type f -name "*.json" | sort | while read file; do
    echo "uv run python src/classifiers/pipeline.py --config $file"
done > commands.txt

N=$(wc -l < commands.txt)
cat > slurm_job_array.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=multi_gpu_tasks
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gpus=a30:1
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-$((N-1))%8

CMD=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" commands.txt)
echo "Running task \$SLURM_ARRAY_TASK_ID: \$CMD"
eval \$CMD
EOF