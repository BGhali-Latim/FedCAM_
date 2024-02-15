import os
import json 
import pandas as pd
import argparse

dummy_dict = { "accuracies" : {
    "No attack" : [],
    "FedCam" : [],
        #{"accuracies" : [0.2,0.5,
        # "rounds" : [1,2,3,4]},
    "FedGuard" : [],
        #{"accuracies" : [0.25,0.4
        # "rounds" : [1,2,3,4]} 
    "No defense" : [],
}
}

Results_dir = "../Results/MNIST/FedCAM"
experiment_list = ["non-IID_10_AdditiveNoise",
                #"non-IID_50_AdditiveNoise",
                #"non-IID_50_SameValue",
                #"non-IID_50_SignFlip",
                #"non-IID_50_NaiveBackdoor",
                #"non-IID_50_SquareBackdoor"]
                ]
metric_list = ["test_accuracy_100.json"]
resultfile_name = "test_accuracy_100.json"
experiment_dirname = "non-IID_0_NoAttack"
# base = "../Results/Baselines/MNIST_non-IID_CNN/test_accuracy_100.json"
base = "../Results/MNIST/NoDefence/non-IID_0_NoAttack/test_accuracy_100.json"

def collect_results_several(Results_dir, experiment_list, resultfile_name, baseline = "../Results/MNIST/NoDefence/non-IID_0_NoAttack/test_accuracy_100.json"):
    results_dict = {}
    if baseline != "False" :
        with open(baseline,'r') as source:
            results_dict["Baseline"]=list(json.load(source))
    for experiment in experiment_list:
        with open(os.path.join(Results_dir,experiment,resultfile_name),'r') as source:
            results_dict[experiment]=list(json.load(source))
    return results_dict

def collect_results_single(Results_dir, experiment_dirname, metric_list, baseline = True):
    results_dict = {}
    print(experiment_dirname)
    if baseline != "False" :
        with open(baseline,'r') as source:
            results_dict["Baseline"]=list(json.load(source))
    for metric in metric_list:
        with open(os.path.join(experiment_dirname, metric),'r') as source:
            results_dict[metric]=list(json.load(source))
    return results_dict

def collect_results_compare(algo2_dir,algo1_dir, experiment, metric, baseline = True):
    results_dict = {}
    print(algo2_dir)
    print(algo1_dir)
    if baseline != "False" :
        with open(baseline,'r') as source:
            results_dict["Baseline"]=list(json.load(source))
    with open(os.path.join(algo2_dir,experiment, metric),'r') as source:
        results_dict[algo2_dir]=list(json.load(source))
    with open(os.path.join(algo1_dir,experiment, metric),'r') as source:
        results_dict[algo1_dir]=list(json.load(source))
    return results_dict

def collect_results_absc(Results_dir, experiment_dirname, metric_list):
    results_dict = {}
    with open(os.path.join(Results_dir,experiment_dirname, metric_list[0]),'r') as source:
        results_dict["absc"]=list(json.load(source))
    with open(os.path.join(Results_dir,experiment_dirname, metric_list[1]),'r') as source:
        results_dict["ord"]=list(json.load(source))
    return results_dict

#df = pd.DataFrame.from_dict(collect_results(Results_dir, experiment_list,resultfile_name))
#print(df.head())
#df.to_csv("accuracies.csv")
#with open ("demo.json","w") as f : 
#    json.dump(dummy_dict,f)

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str)
    parser.add_argument("-experiment", type=str)    
    parser.add_argument("-metric_list", nargs="+", default=["test_accuracy_100.json"])
    parser.add_argument("-metric", type=str)
    parser.add_argument("-experiment_list", nargs="+")
    parser.add_argument("-algo1_dir", type=str)
    parser.add_argument("-algo2_dir", type=str)
    parser.add_argument("-result_dir", type=str)
    parser.add_argument("-baseline", default = "False") 
    args = parser.parse_args()
    print(args.baseline)
    if args.mode == "single" : 
        results_dict = collect_results_single(args.result_dir, args.experiment, args.metric_list, args.baseline)
        df = pd.DataFrame.from_dict(results_dict)
        print(df.head())
        df.to_csv("accuracies.csv")
    elif args.mode == "several" : 
        results_dict = collect_results_several(args.result_dir, args.experiment_list, args.metric, args.baseline)
        df = pd.DataFrame.from_dict(results_dict)
        print(df.head())
        df.to_csv("accuracies.csv")
    elif args.mode == "compare": 
        results_dict = collect_results_compare(args.algo2_dir, args.algo1_dir, args.experiment, args.metric, args.baseline)
        df = pd.DataFrame.from_dict(results_dict)
        print(df.head())
        df.to_csv("accuracies.csv")
    elif args.mode == "withabsc": 
        results_dict = collect_results_absc(args.result_dir, args.experiment, args.metric_list)
        print(results_dict)
        with open("xs.json", "w") as f :
            json.dump(results_dict, f)
    else :
        print("please provide a valid mode")


