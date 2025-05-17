# Here we call the qsub function and we give all the arguments
# It is a job array so we give the name of the job and its out/err
# The main important part is the argument of t which gives the index
# of the job array (starts from 1)
name=tb_WSe2
output=outWSe2.out
error=outWSe2.err

qsub -N $name -o $output -e $error -t 1-240 -q rademaker qjob.sh


