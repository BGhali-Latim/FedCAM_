import seaborn as sns 
import json 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_accuracies(input_path = "../Results/MNIST/FedCAM2/non-IID_30_MajorityBackdoor",
                    figname= "./MajorityBackdoor_attacker_profiles.png",
                    figtitle = "undetected MajorityBackdoor attackers data size per round"): 
    sns.set_theme()
    sns.set(rc={'figure.figsize':(11,9)})

    attackers = []
    round_nb = []
    with open(os.path.join(input_path,"attacker_profiles_100.json"),"r") as source : 
        profiles = np.array(json.load(source))   
    print(profiles[0])
    for idx,profile in enumerate(profiles) : 
        attackers.extend(profile)
        round_nb.extend([idx]*len(profile))
    ax = sns.scatterplot(x=round_nb, y=attackers,  color = 'black')

    undetected_attackers = []
    round_nb = []
    with open(os.path.join(input_path,"passing_attacker_profiles_100.json"),"r") as source : 
        profiles = np.array(json.load(source))   
    for idx,profile in enumerate(profiles) : 
        undetected_attackers.extend(profile)
        round_nb.extend([idx]*len(profile))
    if len(profiles)>0 :
        ax = sns.scatterplot(x=round_nb, y=undetected_attackers,  color = 'r')

    ax.set(xlabel="Rounds", ylabel="undetected attackers num samples", title=figtitle)

    smallest_attacker = min(attackers)
    ax.axhline(y = smallest_attacker, linestyle = '--', color='black')
    ax.text(-10, smallest_attacker, f"y = {smallest_attacker}")

    if len(undetected_attackers)>0 :
        smallest_undetected_attacker = min(undetected_attackers)
        ax.axhline(y = smallest_undetected_attacker, linestyle='--', color = 'r')
        ax.text(-10, smallest_undetected_attacker, f"y = {smallest_undetected_attacker}")
        ax.legend(['Attacker', 'undetected attacker', 'Smallest attacker', 'Smallest undetected attacker'])
    else :
        ax.legend(['Attacker','Smallest attacker'])

    detected = (len(attackers)-len(undetected_attackers))/len(attackers)
    ax.text(75, max(attackers)*0.75, f"Detected {(detected)*100:.2f}% of attackers")

    #plt.xticks(range(1,101))
    plt.savefig(os.path.join("./output",figname))
    plt.show()

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", type=str, default = "../Results/MNIST/FedCAM2/non-IID_30_SignFlip")
    parser.add_argument("-figname", type=str, default = "./MajorityBackdoor_attacker_profiles.png")
    parser.add_argument("-figtitle", type=str, default= "undetected MajorityBackdoor attackers data size per round")    
    args = parser.parse_args()
    plot_accuracies(input_path= args.input_path, 
                    figname= args.figname,
                    figtitle = args.figtitle)
                    #figname="../Results/FedCAM/non-IID_30_NaiveBackdoor/accuracy.png")