#!/bin/bash

for ((i=0; i<=35; i++)); do
    python3 monolayer.py $i
done

