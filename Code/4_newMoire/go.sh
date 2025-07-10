#!/bin/bash

for ((i=0; i<$1; i++)); do
    echo "Index $i"
    python3 edc.py $i S3
done
