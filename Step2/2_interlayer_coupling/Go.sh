#!/bin/bash

for i in {0..120}
do
    python3 interlayer_coupling.py S11 $i
done
