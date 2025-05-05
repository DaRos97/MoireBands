#!/bin/bash

for ((i=0;i<$2;i++));
do
    python3 interlayer.py $1 $i
done
