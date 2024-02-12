#!/bin/bash

algo_dir="FedCAM"

# Compare attacks
# for attack in NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do 
for attack in NaiveBackdoor SquareBackdoor; do 
    python3 generate_samples.py -mode compare -baseline False -algo1_dir ../Results/MNIST/$algo_dir  -algo2_dir ../Results/MNIST/NoDefence -experiment non-IID_30_$attack -metric "$attack""_accuracy_100.json"
    python3 plot_results_compare.py -columns "Attack with $algo_dir" "Attack without $algo_dir" -figname "$attack""_0.3_backdoor_acc_with_vs_without_$algo_dr" -figtitle "Backdoor accuracy comparison for a 30% $attack attack scenario with and without $algo_dir"
done










































