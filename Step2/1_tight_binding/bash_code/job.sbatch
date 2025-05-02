#!/bin/bash

#SBATCH --job-name=WSe2_tb

#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00

# #SBATCH --partition=debug-cpu
# #SBATCH --time=00:15:00


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-500

#SBATCH --output=out.out
#SBATCH --error=out.err

srun $HOME/1_tight_binding/qjob.qsub $SLURM_ARRAY_TASK_ID
