#!/bin/bash

echo "started script" >> testing.out

## Get Baseline for FashionMNIST (with normalisation + 128)
#for attack in NoAttack; do
#    echo "running with 30% $attack attackers" 
#    python3 TestMain.py -algo noDefense -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/FashionMNIST_baseline_0.3_$attack.out
#done
#
## Run fedGuard on FashionMNIST 
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do
#    echo "running fedGuard with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedGuard -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/fashion_fedcam_0.3_$attack.out
#done

# Run fedCAM on FashionMNIST (with no normalisation + 128)
#for attack in SameValue ; do 
#    echo "running fedCam2 with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam -attack $attack -ratio 0.3 -dataset MNIST| tee Results/fashion_fedcam_0.3_$attack.out
#done
#
## Run fedCAM on CIFAR10
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do
#    echo "running fedCam with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam -attack $attack -ratio 0.3 -dataset CIFAR10| tee Results/cifar_fedcam_0.3_$attack.out
#done
#
## Run fedCVAE on CIFAR10 #TOFIX, call right config
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do
#    echo "running fedCVAE with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCVAE -attack $attack -ratio 0.3 -dataset CIFAR10| tee Results/cifar_fedcvae_0.3_$attack.out
#done

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

# Run missing
#for attack in SameValue ; do 
#    echo "running fedCam2 with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam2 -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/fashion_fedcam2_0.3_$attack.out
#done
#for attack in AdditiveNoise ; do 
#    echo "running fedgUARD with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedGuard -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/fashion_ffeguard_0.3_$attack.out
#done
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do
#    echo "running cos with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam_cos -attack $attack -ratio 0.3 -dataset MNIST| tee Results/fedcam_cos_iid_0.3_$attack.out
#done
#for attack in NoAttack AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do
#    echo "running cos with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam_cos_non_iid -attack $attack -ratio 0.3 -dataset MNIST| tee Results/fedcam_cos_non_iid_0.3_$attack.out
#done
#for attack in SameValue; do
#    echo "running cos with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedCam_cos_non_iid -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/fedcam_cos_fashion_non_iid_0.3_$attack.out
#done
#for attack in AdditiveNoise SameValue SignFlip; do #For recall etc
#    echo "running fedgUARD with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedGuard_1000 -attack $attack -ratio 0.3 -dataset MNIST| tee Results/mnist_guard10000.3_$attack.out
#done
#for attack in AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do #For recall etc
#    echo "running fedgUARD with 30% $attack attackers on $dataset" 
#    python3 TestMain.py -algo fedGuard_1000 -attack $attack -ratio 0.3 -dataset FashionMNIST| tee Results/fashion_guard1000_0.3_$attack.out
#done 

for attack in SameValue SignFlip AdditiveNoise ; do #For recall etc
    echo "running cos non iid with 30% $attack attackers on $dataset" 
    python3 TestMain.py -algo fedCam_cos_non_iid -attack $attack -ratio 0.3 -dataset MNIST| tee Results/mnist_cosnoniidnew_$attack.out
done