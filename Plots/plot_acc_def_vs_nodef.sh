#!/bin/bash

algo_dir="FedCAM_cos"
metric="test_accuracy"
dataset="MNIST"

# Compare in no attack scenario
python3 generate_samples.py -mode compare -baseline False -algo2_dir ../Results/$dataset/NoDefence -algo1_dir ../Results/$dataset/$algo_dir -experiment non-IID_30_NoAttack -metric "$metric""_100.json"
python3 plot_results_compare.py -columns "Baseline" "$algo_dir" -figname "$dataset""_NoAttack_accuracy_with_vs_without_$algo_dr" -figtitle "Performance comparison for a no attack scenario with and without $algo_dir on $dataset"

# Compare attacks
for attack in AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor; do 
#for attack in SameValue SignFlip; do 
    python3 generate_samples.py -mode compare -baseline ../Results/$dataset/NoDefence/non-IID_30_NoAttack/test_accuracy_100.json -algo2_dir ../Results/$dataset/$algo_dir -algo1_dir ../Results/MNIST/NoDefence -experiment non-IID_30_$attack -metric "$metric""_100.json"
    python3 plot_results_compare.py -columns "Baseline (NoAttack)" "Attack with $algo_dir" "Attack without $algo_dir" -figname "$dataset""_$attack""_0.3_with_vs_without_$algo_dr" -figtitle "Performance comparison for a 30% $attack attack scenario with and without $algo_dir on $dataset"
done










































