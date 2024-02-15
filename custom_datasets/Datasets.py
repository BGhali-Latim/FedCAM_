import torch
from torch.utils.data import Dataset

class SyntheticLabeledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dt = self.data[idx, :]
        lbl = self.labels[idx]
        return dt, lbl
    
    #class OneClassDataset(Dataset): 
    #    def __init__(self, dataset, class_label):
    #        self.dataset = dataset
