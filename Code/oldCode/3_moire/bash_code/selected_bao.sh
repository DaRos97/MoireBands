#!/bin/bash

for Vg in 0.0300 0.0350
do
    for pg in 3.1416
    do
        for Vk in 0.0150
        do
            for pk in 2.0944 2.7925 3.1416
            do
                fn1=bands_fit_4_${Vg}_${pg}_${Vk}_${pk}_5_79.8_C3.png
                fn2=fit_Gauss_0.0100_0.0100_4_${Vg}_${pg}_${Vk}_${pk}_5_79.8_C3.png
                scp rossid@login2.baobab.hpc.unige.ch:~/3_moire/results/figures/bands/$fn1 /home/dario/Desktop/git/MoireBands/last_lap/3_moire/results/figures/selected/
                scp rossid@login2.baobab.hpc.unige.ch:~/3_moire/results/figures/spread/$fn2 /home/dario/Desktop/git/MoireBands/last_lap/3_moire/results/figures/selected/
            done
        done
    done
done
