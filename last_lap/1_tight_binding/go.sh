#!/bin/bash

for i in {0..31}
do
    python3 select_best_result.py 0 $i
done
