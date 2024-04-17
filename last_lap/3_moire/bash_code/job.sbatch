#!/bin/bash

#SBATCH --job-name=moire

# #SBATCH --partition=shared-cpu
# #SBATCH --time=12:00:00
#SBATCH --partition=debug-cpu
#SBATCH --time=00:05:00

#SBATCH --array=0-134
#SBATCH --output=bash_code/out.out
#SBATCH --error=bash_code/out.err

srun bash_code/qjob.qsub $SLURM_ARRAY_TASK_ID
