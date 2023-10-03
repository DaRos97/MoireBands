#!/bin/bash

for ((i=0; i<100; i++))
do
    python main.py --grid --fold 3 --spread $i
#    python main.py --grid --fold 6 --spread $i
done
