#!/bin/bash

for N in 5
do
    for V in 0.01 0.0125 0.015 0.0175 0.02 0.0225 0.025 0.0275 0.03 0.0325 0.033 0.035 0.0375 0.04
    do
        for phi in 2.9 2.95 3.0 3.05 3.1 3.14 #0 0.3 0.6 1.2 1.5 1.8 2.1 2.4 2.7 3.14
        do
            for e_ in 0.03 0.04 0.05 0.06
            do
                for k_ in 0.02
                do
                    echo "Computing N=$N, V=$V, phi=$phi, e_=$e_, k_=$k_"
                    python G3_fit_Moire.py $N $V $phi $e_ $k_
                done
            done
        done
    done
done

