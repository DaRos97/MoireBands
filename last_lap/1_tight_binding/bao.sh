#!/bin/bash

# rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/pars*.npy
#scp rossid@login1.yggdrasil.hpc.unige.ch:~/1_tight_binding/results/*.npy /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/

rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/*.npy
scp rossid@login2.baobab.hpc.unige.ch:~/1_tight_binding/results/temp/*.npy /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/
