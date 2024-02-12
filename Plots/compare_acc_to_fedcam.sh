#!/bin/bash

algo_dir="FedCAM2"
metric="test_accuracy"

# Compare no attack scenario
python3 generate_samples.py -mode compare -baseline ../Results/MNIST/NoDefence/non-IID_30_NoAttack/test_accuracy_100.json -algo2_dir ../Results/MNIST/FedCAM/26_01_24 -algo1_dir ../Results/MNIST/$algo_dir -experiment non-IID_30_NoAttack -metric "$metric""_100.json"
python3 plot_results_compare.py -columns "Baseline (NoAttack)" "FedCAM" "$algo_dir"  -figname "$algo_dir""_vs_fedCAM_NoAttack" -figtitle "Performance comparison to fedCAM for no attack scenario"

# Compare attacks
for attack in AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do 
    python3 generate_samples.py -mode compare -baseline ../Results/MNIST/NoDefence/non-IID_30_NoAttack/test_accuracy_100.json -algo2_dir ../Results/MNIST/FedCAM/26_01_24 -algo1_dir ../Results/MNIST/$algo_dir -experiment non-IID_30_$attack -metric "$metric""_100.json"
    python3 plot_results_compare.py -columns "Baseline (NoAttack)" "FedCAM" "$algo_dir"  -figname "$attack""_0.3_with_""$algo_dir""_vs_fedCAM" -figtitle "Performance comparison to fedCAM for a 30% $attack attack scenario"
done










































