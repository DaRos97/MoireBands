#!/bin/bash

for N in 5
do
    for V in 0.005 0.01 0.015 0.02 0.025 0.03
    do
        for phi in 0 0.3 0.6 1.2 1.5 1.8 2.1 2.4 2.7 3.14
        do
            for e_ in 0.02 0.03 0.04 0.05 0.06
            do
                for k_ in 0.005 0.01 0.015 0.02
                do
                    echo "Computing N=$N, V=$V, phi=$phi, e_=$e_, k_=$k_"
                    #python G3_fit_Moire.py $N $V $phi $e_ $k_
                    python K3_fit_Moire.py $N $V $phi $e_ $k_
                done
            done
        done
    done
done

