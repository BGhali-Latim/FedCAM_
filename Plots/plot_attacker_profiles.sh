#!/bin/bash

algo_dir="FedCAM_cos_a3"

# Compare attacks
# for attack in AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do 
for attack in AdditiveNoise SignFlip; do 
    python3 plot_passing_attackers.py \
    -input_path "../Results/MNIST/$algo_dir/non-IID_30_$attack"\
    -figname "non-IID_30_$attack"_"$algo_dir""attacker_profiles" \
    -figtitle "Passing $attack attackers data size per round"
done 










































