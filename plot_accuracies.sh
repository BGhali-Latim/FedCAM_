#!/bin/bash

echo "started script" >> testing.out 

# Compare attacks
cd ./Plots 
for metric in test_accuracy; do 
   for attack in AdditiveNoise SignFlip SameValue; do
        ls
       python3 generate_samples.py -mode several \
       -baseline False \
       -result_dir "./final_results/completing_missing_exps_0.2(convmixer100eps)/MNIST/FedCAM_dev" \
       -experiment_list  "../../../baseline/convmixer(100eps)" "../../../noDefense/convmixer(100eps)/non-IID_30_$attack" "non-IID_30_$attack" \
       "../../../others_100/CNN_fashionmnist/fedCVAE/non-IID_30_$attack" "../../../others_100/CNN_fashionmnist/fedGuard/non-IID_30_$attack"\
       -metric "$metric""_100.json"
       python3 plot_results_compare_smoothed.py \
       -columns "Baseline" "NoDefense" "FedCAM" "FedCVAE" "FedGuard"\
       -figname "$attack""_$metric""0.3_all_comparison" \
       -figtitle "$metric comparison of algos in a $attack scenario"
   done
done

# Compare Noattack
for metric in test_accuracy; do 
   for attack in NoAttack; do 
       python3 generate_samples.py -mode several \
       -baseline False \
       -result_dir "./final_results/completing_missing_exps_0.2(convmixer100eps)/MNIST/FedCAM_dev" \
        -experiment_list  "../../../baseline/convmixer(100eps)" "../../../noDefense/convmixer(100eps)/non-IID_30_$attack" "non-IID_30_$attack" \
        "../../../others_100/CNN_fashionmnist/fedCVAE/non-IID_30_$attack" "../../../others_100/CNN_fashionmnist/fedGuard/non-IID_30_$attack"\
       -metric "$metric""_100.json"
       python3 plot_results_compare_smoothed.py \
       -columns "Baseline" "NoDefense" "FedCAM" "FedCVAE" "FedGuard"\
       -figname "$attack""_$metric""0.3_all_comparison" \
       -figtitle "$metric comparison of algos in a $attack scenario"
   done
done



