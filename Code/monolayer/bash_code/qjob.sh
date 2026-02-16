# Here we call the function for each array evaluation.

output_file=Scratch/${SGE_TASK_ID}.out
error_file=Scratch/${SGE_TASK_ID}.err

python3 monolayer.py $SGE_TASK_ID >$output_file 2>$error_file
