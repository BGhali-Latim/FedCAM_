#!/bin/bash

experiment='scaled'

echo "started script" >> progress_tests.out

for dataset in MNIST FashionMNIST; do
    for algo in fedCAM_dev; do 
       for attack in Scaled; do
         python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling CAM | tee "Results/TEST.$algo.$dataset.$attack.CAM"
         echo "$dataset.$algo.$attack.CAM" >> progress_tests.out
         python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset -experiment $experiment -sampling Dirichlet_per_class | tee "Results/TEST.$algo.$dataset.$attack.DirichletM"
         echo "$dataset.$algo.$attack.Dirichlet" >> progress_tests.out
       done
    done
done
echo "finished fledge" >> progress_tests.out




