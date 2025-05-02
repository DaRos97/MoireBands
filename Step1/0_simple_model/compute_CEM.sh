#!/bin/bash

for i in {0..35}
do
    echo "Computing i=$i"
    python CEM.py $i
done

