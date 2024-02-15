#!/bin/bash

algo_dir="FedCAM2_128batch"
dataset="MNIST"

# In no attack scenario
#python3 generate_samples.py -mode single \
# -baseline False \
# -experiment ../Results/MNIST/$algo_dir/non-IID_30_NoAttack \
# -metric_list attacker_detection_recall_100.json attacker_detection_precision_100.json
#python3 plot_results_compare.py -columns "Detection rate" "Misclassification rate" -figname "NoAttack_$algo_dir""_detection" -figtitle "Detection performance for a no attack scenario with $algo_dir"

# Compare attacks
#for attack in AdditiveNoise SameValue SignFlip NaiveBackdoor SquareBackdoor MajorityBackdoor TargetedBackdoor; do 
for attack in SameValue SignFlip; do 
    python3 generate_samples.py -mode single \
     -baseline False \
     -experiment ../Results/$dataset/$algo_dir/non-IID_30_$attack \
     -metric_list attacker_detection_recall_100.json attacker_detection_precision_100.json
    python3 plot_results_compare.py -mode detection -columns "Detection rate" "Precision" -figname "$attack""_$algo_dir""_detection.png" -figtitle "Detection performance for a 30% $attack scenario with $algo_dir"
done










































