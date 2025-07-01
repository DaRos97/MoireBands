#!/bin/bash

for ((i=0; i<=77; i++)); do
    echo "Index $i"
    python3 ldos.py $i
done
