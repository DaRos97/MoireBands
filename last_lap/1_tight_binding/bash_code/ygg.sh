#!/bin/bash

#rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/pars*.npy
#rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/pars*.npy
#
#scp rossid@login1.yggdrasil.hpc.unige.ch:~/1_tight_binding/results/*.npy /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/
scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/1_tight_binding/results/temp* /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/
