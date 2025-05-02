#!/bin/bash
if [ $1 == "y" ]
then
    m="yggdrasil"
    g="1"
else
    m="baobab"
    g="2"
fi


scp rossid@login${g}.${m}.hpc.unige.ch:~/1_monolayer_fit/Data/* ~/Desktop/git/MoireBands/1_monolayer_fit/Data/ 
