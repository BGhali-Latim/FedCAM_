#!/bin/bash

echo "started script" >> testing.out

dataset="MNIST" #non - iid

#for algo in fedCAM2 fedCAM_cos fedCWR fedCVAE fedGuard; do 
#    for attack in NoAttack AdditiveNoise SameValue SignFlip SourcelessBackdoor AlternatedBackdoor; do 

#for algo in fedCAM_cos fedCWR; do 
#    for attack in AdditiveNoise SameValue SignFlip SourcelessBackdoor AlternatedBackdoor; do 
#        echo "running $algo non iid with 30% $attack attackers on $dataset" 
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset| tee $algo.$dataset.$attack.out
#    done
#done

for algo in fedCAM; do 
#    for attack in NoAttack SignFlip SameValue AdditiveNoise; do 
    for attack in SameValue SignFlip; do 
        echo "running $algo non iid with 30% $attack attackers on $dataset" 
        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset| tee $algo.$dataset.$attack.out
    done
done

#for algo in noDefense; do 
#    for attack in AlternatedBackdoor; do 
#        echo "running $algo non iid with 30% $attack attackers on $dataset" 
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset| tee $algo.$dataset.$attack.out
#    done
#done
#
#for algo in noDefense fedCAM; do 
#    for attack in SourcelessBackdoor; do 
#        echo "running $algo non iid with 30% $attack attackers on $dataset" 
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset| tee $algo.$dataset.$attack.out
#    done
#done

#for algo in fedCAM_cos fedCWR; do 
#    for attack in NoAttack AdditiveNoise SameValue SignFlip SourcelessBackdoor AlternatedBackdoor; do 
#        echo "running $algo non iid with 30% $attack attackers on $dataset" 
#        python3 TestMain.py -algo $algo -attack $attack -ratio 0.3 -dataset $dataset| tee $algo.$dataset.$attack.out
#    done
#done

# Compare attacks
#cd ./Plots 
#for metric in test_accuracy; do 
#    for attack in AdditiveNoise SignFlip SameValue; do 
#        python3 generate_samples.py -mode several \
#        -baseline False \
#        -result_dir ../Results/all_in_act_3/MNIST/ \
#        -experiment_list  "NoDefence/non-IID_30_NoAttack" "NoDefence/non-IID_30_$attack" "FedCAM/non-IID_30_$attack" \
#        "FedCAM2/non-IID_30_$attack" "FedCVAE/non-IID_30_$attack" "FedGuard/non-IID_30_$attack" "FedCWR/non-IID_30_$attack"\
#        -metric "$metric""_100.json"
#        python3 plot_results_compare_smoothed.py \
#        -columns "Baseline" "NoDefence" "FedCAM" "FedCAM2" "FedCVAE" "FedGuard" "FedCWR"\
#        -figname "$attack""_$metric""0.3_all_comparison" \
#        -figtitle "$metric comparison of algos in a $attack scenario"
#    done
#done

echo "run next"




