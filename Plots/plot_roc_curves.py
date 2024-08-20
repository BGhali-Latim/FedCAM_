import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

lambda_list = [0.1, 0.2, 0.3, 0.4 ,0.5, 0.6 ,0.7,  0.8, 0.9]
input_path = "ROC/"

sns.set_theme()
fig, ax = plt.subplots()

for lamda in lambda_list : 
    with open(os.path.join(input_path,f"for_roc_{lamda}/MNIST/FedCAM_dev/non-IID_30_SignFlip","ROC_pred_labels.json"),"r") as source : 
        pred_labels = np.array(json.load(source))   
    with open(os.path.join(input_path,f"for_roc_{lamda}/MNIST/FedCAM_dev/non-IID_30_SignFlip","ROC_real_labels.json"),"r") as source : 
        real_labels = np.array(json.load(source))   
    print(pred_labels)
    print(real_labels)
    fpr, tpr, _ = metrics.roc_curve(real_labels,  pred_labels)
    ax.plot(fpr,tpr, label = lamda)

plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
