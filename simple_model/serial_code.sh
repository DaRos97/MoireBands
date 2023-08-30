#!/bin/bash

for V in {2..3}
do
    for phase in {1..10}
    do
        echo "Computing $V and $phase"
        python interlayer_2.py -N 3 --pts 200 -V $V -p $phase
    done
done
