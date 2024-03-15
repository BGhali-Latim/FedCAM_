import seaborn as sns 
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def smooth(values, window): 
    smoothed_values = []
    for ind in range(len(values) - window + 1):
        smoothed_values.append(np.mean(values[ind:ind+window]))
    for ind in range(window -1): 
        smoothed_values.insert(0, np.nan)
    return smoothed_values

def plot_accuracies(input_path = "../Results/Baselines/MNIST_non-IID_CNN/test_accuracy_100.json",
                    output_path= "./demo.png",
                    figtitle = "",
                    columns = []): 
    sns.set_theme()
    #with open(input_path,"r") as source : 
    #    accuracies = np.array(json.load(source))   
    accuracies = pd.read_csv(input_path, index_col=0)
    accuracies.columns = columns                                             
    print(accuracies.head())

    for idx,column in enumerate(columns[::-1]) :
        config = column
        top_accuracies = accuracies[config]
        # Right lower corner
        #plt.text(60, 0.1*(idx+1), f"{column} best accuracy : {max(top_accuracies)*100:.2f}%\
        #     \nAchieved on round {np.argmax(top_accuracies)+1}")
        # Upper left corner
        plt.text(50 , 0.1*(idx+1), f"{column} best accuracy : {max(top_accuracies)*100:.2f}%\
         Achieved on round {np.argmax(top_accuracies)+1}")
        
    for column in accuracies.columns : 
        accuracies[column] = smooth(accuracies[column], 3)
    print(accuracies.head())

    sns.set(rc={'figure.figsize':(22,18)})
    ax = sns.lineplot(data=accuracies, dashes = False, palette = ['b','r','black'], linestyle ='solid')
    #ax = sns.lineplot(data=accuracies, dashes = False, linestyle ='solid')
    ax.set(xlabel="Rounds", 
           ylabel="Test accuracy", 
           title=figtitle)
    ax.set_ylim([0,1.1])
    #plt.xticks(range(1,101))
    plt.savefig(output_path)
    plt.show()

def plot_detection(input_path = "../Results/Baselines/MNIST_non-IID_CNN/test_accuracy_100.json",
                    output_path= "./demo.png",
                    figtitle = "",
                    columns = []): 
    sns.set_theme()
    #with open(input_path,"r") as source : 
    #    accuracies = np.array(json.load(source))   
    accuracies = pd.read_csv(input_path, index_col=0)
    accuracies.columns = columns                                             
    print(accuracies.head())
    sns.set(rc={'figure.figsize':(11,9)})
    ax = sns.lineplot(data=accuracies, dashes = False, color = 'g', linestyle ='solid')
    ax.set(xlabel="Rounds", 
           title=figtitle)
    ax.set_ylim([0,1.1])
    ax.set_xlim([0,accuracies.shape[0]])
    #plt.xticks(range(1,101))
    ax.axhline(y = 0.5, linestyle = '--', color='black')
    ax.text(-10, 0.5, "y = 0.5")
    plt.savefig(output_path)
    #plt.show()

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default = "accuracies")
    parser.add_argument("-figname", type=str, default = "demo")
    parser.add_argument("-figtitle", type=str) 
    parser.add_argument("-columns", type=str, nargs="+")   
    args = parser.parse_args()
    if args.mode == "detection" : 
        plot_detection(input_path="./accuracies.csv", 
                        output_path= f"./output/{args.figname}.png",
                        figtitle = args.figtitle,
                        columns = list(args.columns))
                        #output_path="../Results/FedCAM/non-IID_30_NaiveBackdoor/accuracy.png")
    else :
        plot_accuracies(input_path="./accuracies.csv", 
                output_path= f"./output/{args.figname}.png",
                figtitle = args.figtitle,
                columns = list(args.columns))
                #output_path="../Results/FedCAM/non-IID_30_NaiveBackdoor/accuracy.png")