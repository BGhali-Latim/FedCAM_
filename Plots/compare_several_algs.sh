#!/bin/bash

metric="test_accuracy"
dataset="MNIST"

# Compare attacks
#for metric in test_accuracy; do 
#    for attack in NoAttack; do 
#        python3 generate_samples.py -mode several \
#        -baseline False \
#        -result_dir ../Results/cross-silo/MNIST \
#        -experiment_list  "NoDefence/non-IID_0_NoAttack" "FedCAM/non-IID_30_$attack""2" \
#        -metric "$metric""_100.json"
#        python3 plot_results_compare_smoothed.py \
#        -columns "Baseline" "FedCAM"\
#        -figname "4" \
#        -figtitle "$metric comparison of algos in a $attack scenario"
#        #-figname "$attack""_$metric""0.3_all_comparison" 
#    done
#done

#Backdoor
for metric in test_accuracy; do 
    for attack in AlternatedBackdoor; do 
        python3 generate_samples.py -mode several \
        -baseline False \
        -result_dir ../Results/poubelle\
        -experiment_list  "non-IID_0_NoAttack" "non-IID_30_$attack" "non-IID_03_$attack" \
        -metric "$metric""_100.json"
        python3 plot_results_compare_smoothed.py \
        -columns "Baseline" "test accuracy" "backdoor accuracy"\
        -figname "5" \
        -figtitle "Accuracies in a noDefense $attack scenario"
    done
done

#Activations
#for metric in test_accuracy; do 
#    for attack in SameValue; do 
#        python3 generate_samples.py -mode several \
#        -baseline False \
#        -result_dir ../Results/poubelle\
#        -experiment_list   "non-IID_0_NoAttack" "non-IID_30_$attack" "non-IID_31_$attack" "non-IID_41_$attack" \
#        -metric "$metric""_100.json"
#        python3 plot_results_compare_smoothed.py \
#        -columns "Baseline" "Activations 3 (tanh)" "Activations 3 (leaky relu)" "Activations 4 (leaky relu)"\
#        -figname "1" \
#        -figtitle "Accuracies in a noDefense $attack scenario"
#    done
#done

#-experiment_list "NoDefence/non-IID_30_$attack" "FedCAM/26_01_24/non-IID_30_$attack" "FedCVAE/non-IID_30_$attack" \
#"FedGuard/29_01_24/non-IID_30_$attack" "FedCAM2/non-IID_30_$attack"\

#-columns "No defense" "FedCAM" "FedCVAE" "FedGuard" "FedCAM2"  \
#"FedCAM_cos_a1/non-IID_30_$attack"










































