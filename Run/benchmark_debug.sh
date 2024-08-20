#!/bin/bash

echo "started script" >> testing.out
cd ..

experiment='debug'
dataset="FashionMNIST" #non - iid
architecture="CNNWithDropBlock"

for algo in noDefense; do 
   for attack in NoAttack; do  
     echo "running $algo non iid with 30% $attack attackers on $dataset" 
     python3 TestMain_new.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -architecture $architecture -sampling CAM | tee ./logs/$experiment.$algo.$dataset.$attack.out
   done
done

echo "run next"




