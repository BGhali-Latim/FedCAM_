#!/bin/bash

echo "started script" >> testing.out

# Run fedCAM on Fashion MNIST
for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do
    echo "running with 30% $attack attackers" 
    python3 TestMain.py -algo fedCAM -attack $attack -ratio 0.3 | tee Results/fedcvae_0.3_$attack.out
done

# Run fedCam for missing
#for attack in AdditiveNoise; do 
#    echo "running with 30% $attack attackers" 
#    python3 TestMain.py -algo fedCam -attack $attack -ratio 0.3 | tee Results/fedCam_new_0.3_$attack.out 
#done
#
## Run fedGuard with GuardCNN for missing
#for attack in AdditiveNoise; do 
#    echo "running with 30% $attack attackers" 
#    python3 TestMain.py -algo fedGuard -attack $attack -ratio 0.3 | tee Results/fedGuard_withGuardCNN_0.3_$attack.out 
#done

echo "run next"

# NEXT
# RUn fedCam with 100 epochs (or fedCAM2 ???)
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do
#    echo "running with 30% $attack attackers" 
#    #python3 TestMain.py -algo noDefense -attack $attack -ratio 0.3 | tee Results/nodef_0.3_$attack.out 
#    python3 TestMain.py -algo fedCam -attack $attack -ratio 0.3 | tee Results/fedcam_100_0.3_$attack.out
#done

#STORAGE
#echo "running without attackers" 
#python3 TestMain.py -algo fedCam -attack NoAttack -ratio 0 | tee Results/baseline.out