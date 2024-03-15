#!/bin/bash

echo "started script" >> testing.out

dataset="FEMNIST" #non - iid

for algo in noDefense fedCAM_cos ; do 
    for attack in AdditiveNoise; do 
        echo "running $algo non iid with 90% $attack attackers on $dataset" 
        python3 TestMain.py -algo $algo -attack $attack -ratio 0.9 -dataset $dataset| tee $algo.$dataset.$attack.out
    done
done

echo "run next"




