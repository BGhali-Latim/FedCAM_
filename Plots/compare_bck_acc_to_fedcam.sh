#!/bin/bash

algo_dir="FedGuard"

# Compare attacks
for attack in NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do 
    python3 generate_samples.py -mode compare \
    -baseline ../Results/MNIST/NoDefence/non-IID_30_$attack/"$attack""_accuracy_100.json" \
    -algo2_dir ../Results/MNIST/FedCAM/26_01_24 \
    -algo1_dir ../Results/MNIST/$algo_dir \
    -experiment non-IID_30_$attack \
    -metric "$attack""_accuracy_100.json"
    python3 plot_results_compare.py \
    -columns "No defense" "fedCam" "$algo_dir" \
    -figname "$attack""non-IID_0.3_backdoor_acc_fedCam_vs_$algo_dr" \
    -figtitle "Backdoor accuracy comparison for a 30% $attack attack scenario"
done










































