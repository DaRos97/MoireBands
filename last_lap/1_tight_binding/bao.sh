#!/bin/bash

# rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/pars*.npy
#scp rossid@login1.yggdrasil.hpc.unige.ch:~/1_tight_binding/results/*.npy /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/

rm /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/*WS2*.npy
scp rossid@login1.yggdrasil.hpc.unige.ch:~/1_tight_binding/results/temp/*WS2*.npy /home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/results/temp/
