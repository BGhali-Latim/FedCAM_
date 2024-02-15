#!/bin/bash

metric="attacker_detection_precision"

# Compare attacks
for attack in SameValue; do 
    python3 generate_samples.py -mode several \
    -baseline False \
    -result_dir ../Results/MNIST/non-IID \
    -experiment_list "NoDefence/non-IID_0_$attack" "FedCAM/non-IID_30_$attack" "FedCAM_cos/non-IID_30_$attack" \
    -metric "$metric""_100.json"
    python3 plot_results_compare.py \
    -columns "NoDefense" "FedCAM" "FedCAM_cos"\
    -figname "$attack""_$metric""0.3_all_comparison" \
    -figtitle "Attacker detection comparison of algos in a $attack scenario"
done

#-experiment_list "NoDefence/non-IID_30_$attack" "FedCAM/26_01_24/non-IID_30_$attack" "FedCVAE/non-IID_30_$attack" \
#"FedGuard/29_01_24/non-IID_30_$attack" "FedCAM2/non-IID_30_$attack"\

#-columns "No defense" "FedCAM" "FedCVAE" "FedGuard" "FedCAM2"  \










































