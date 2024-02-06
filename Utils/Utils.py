import os
import json
import numpy as np
import random
import torch
from math import floor

from tqdm import tqdm

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

from Client.Client import Client
#from custom_datasets.Datasets import OneLabelDataset

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def distribute_iid_data_among_clients(num_clients, batch_size, dataset):
        if dataset == "MNIST" :
            data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        elif dataset == "CIFAR10" : 
            trans_cifar_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
            data = datasets.CIFAR10(root='./data', train=True, transform=trans_cifar_train, download=True)
        data_size = len(data) // num_clients
        return [
            DataLoader(Subset(data, range(i * data_size, (i + 1) * data_size)), batch_size=batch_size, shuffle=True, drop_last=False)
            for i in range(num_clients)]
    
    @staticmethod
    def distribute_non_iid_data_among_clients(num_clients, batch_size, dataset):
        if dataset == "MNIST" :
            train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
            size_def = 2000
        elif dataset == "CIFAR10" : 
            trans_cifar_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])    
            train_data = datasets.CIFAR10(root='./data', train=True, transform=trans_cifar_train, download=True)
            train_data.targets = torch.tensor(train_data.targets)
            train_data.data = torch.tensor(train_data.data)
            size_def = 1040

        indices = [torch.where(train_data.targets == idx)[0] for idx in range(0, 10)]
        #print(len(indices))
        #print([len(elem) for elem in indices])

        subdatasets = []
        backdoor_sets = []
        tuples_set = []

        #class_counts = [0 for i in range(10)]
        #class_distrib = [0 for i in range(10)]
        #sample_counts = 0s
        for k in tqdm(range(1,num_clients+1)):   
            sample_size = int(floor(size_def/(k+5)))+20

            while True :
                i, j = random.sample(range(0, 10), k=2)
                # print(i, '/', j)
                # print(size)
                # print(len(indices[i]), '/', len(indices[j]))
                if (len(indices[i])+len(indices[j]))>=2*sample_size : 
                    break

            #print(f"-----------\n step {k}")
            #print(i, '/', j)
            #print(len(indices[i]), '/', len(indices[j]))
            tuples_set.append([i, j])
            # if ((i == 1) or (j == 1)) and (attack_type == 1):
            #     backdoor_sets.append(k - 1)

            if len(indices[i])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=len(indices[i]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=2*sample_size-len(indices[i]), replace=False)
                #class_counts[i]+=len(indices[i])
                #class_counts[j]+=2*size-len(indices[i])
            elif len(indices[j])<sample_size :
                indice_i = np.random.choice(range(len(indices[i])), size=2*sample_size-len(indices[j]), replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=len(indices[j]), replace=False)
                #class_counts[i]+=2*size-len(indices[j])
                #class_counts[j]+=len(indices[j])
            else : 
                indice_i = np.random.choice(range(len(indices[i])), size=sample_size, replace=False)
                indice_j = np.random.choice(range(len(indices[j])), size=sample_size, replace=False)
                #class_counts[i]+=size
                #class_counts[j]+=size

            selected_i = indices[i][indice_i]
            selected_j = indices[j][indice_j]

            combined = torch.cat((selected_i, indices[i]))
            uniques, counts = combined.unique(return_counts=True)
            indices[i] = uniques[counts == 1]

            combined = torch.cat((selected_j, indices[j]))
            uniques, counts = combined.unique(return_counts=True)
            indices[j] = uniques[counts == 1]

            selected = torch.cat((selected_i, selected_j))

            data_selected = train_data.data[selected]
            label_selected = train_data.targets[selected]

            #print(data_selected)
            #print(data_selected.size())

            tmp = torch.utils.data.TensorDataset(((data_selected.float() / 255.) - 0.1307) / 0.3081, label_selected)

            subdatasets.append(torch.utils.data.DataLoader(
                tmp,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            ))
        #print(sum([len(loader) for loader in subdatasets]))
        #print([len(loader) for loader in subdatasets])
        return subdatasets

    @staticmethod
    def gen_database(num_clients, batch_size, dataset):
        return Utils.distribute_iid_data_among_clients(num_clients, batch_size, dataset)

    @staticmethod
    def gen_clients(config_fl, attack_type, train_data, backdoor_label = '0'):
        total_clients = config_fl["num_clients"]
        num_attackers = int(total_clients * config_fl["attackers_ratio"])

        if attack_type == "MajorityBackdoor" : # Clients are already ordered by data size in non-IID
            attacker_flags = [True] * num_attackers + [False] * (total_clients - num_attackers)
        elif attack_type == "TargetedBackdoor" : 
            attacker_flags = []
            for i in tqdm(range(total_clients)):
                for data,labels in train_data[i] :
                    if (backdoor_label in labels) : 
                        attacker_flags.append(True)
                    else : 
                        attacker_flags.append(False) 
        else :
            attacker_flags = [True] * num_attackers + [False] * (total_clients - num_attackers)
            np.random.shuffle(attacker_flags)
        
        clients = [Client(ids=i, dataloader=train_data[i], is_attacker=attacker_flags[i], attack_type=attack_type)
        for i in tqdm(range(total_clients))] 

        return clients

    @staticmethod
    def cvae_loss(recon_x, x, mu, logvar):
        mse = F.mse_loss(recon_x, x, reduction='mean')
        # MSE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld


    @staticmethod
    def plot_accuracy(accuracy, x_info="round", y_info="Test Accuracy", title_info= "provide a title", save_path=None):
        plt.plot(range(1, len(accuracy) + 1), accuracy)
        plt.xlabel(x_info)
        plt.ylabel(y_info)
        plt.title(title_info)
        plt.savefig(save_path)
        #plt.show()

    @staticmethod
    def plot_hist(data, x_info="Values", y_info="Frequencies", title_info= "provide a title", bins=1000, save_path=None):
        plt.title(title_info)
        plt.xlabel(x_info)
        plt.ylabel(y_info)
        plt.hist(data, bins=bins)
        plt.savefig(save_path)
        #plt.show()


    @staticmethod
    def plot_histogram(hp, nb_attackers_passed_defence_history, nb_attackers_history,
                       nb_benign_passed_defence_history, nb_benign_history, config_fl,
                       attack_type, defence, dir_path, success_rate, attacker_ratio):
        rounds = np.arange(1, hp["nb_rounds"] + 1)

        height_attackers_passed_defense = np.array(nb_attackers_passed_defence_history)
        height_remaining_attackers = np.array(nb_attackers_history) - height_attackers_passed_defense

        height_benign_passed_defense = np.array(nb_benign_passed_defence_history)
        height_remaining_benign = np.array(nb_benign_history) - height_benign_passed_defense

        plt.bar(rounds, height_attackers_passed_defense, color='red', edgecolor='black', alpha=0.5,
                label='Attackers Passed Defence')
        plt.bar(rounds, height_remaining_attackers, bottom=height_attackers_passed_defense, color='yellow',
                edgecolor='black', alpha=0.6, label='Total Attackers')

        plt.bar(rounds, height_benign_passed_defense, bottom=height_attackers_passed_defense + height_remaining_attackers, color='blue', edgecolor='black', alpha=0.5,
                label='Benign Clients Passed Defence')

        plt.bar(rounds, height_remaining_benign, bottom=height_benign_passed_defense + height_attackers_passed_defense + height_remaining_attackers, color='black',
                edgecolor='black', alpha=0.6, label='Total Benign Clients')

        plt.xlabel('Number of Rounds')
        plt.ylabel('Total Nb of Clients')
        plt.ylim(0, config_fl["nb_clients_per_round"])
        plt.title(f"Histogram for {attacker_ratio * 100}% of {attack_type} "
                  f"with {'Defence' if defence else 'No Defence'}")

        plt.legend()
        plt.text(1,45,f"Blocked {success_rate} of attacks", color = 'w', weight = "bold")
        plt.savefig(f"{dir_path}/{attack_type}_{'With defence' if defence else 'No defence'}_Histogram_{hp['nb_rounds']}.pdf")

        #plt.show()

    @staticmethod
    def test(model, device, loader):
        model.to(device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
    
    def get_class_accuracies(model, device, class_datasets, nb_classes):
        model.to(device).eval()
        class_accuracies = {}

        for target_class in range(nb_classes) :
            correct, total = 0, 0
            target_loader = DataLoader(class_datasets[target_class], shuffle=False,drop_last=False)
            with torch.no_grad():
                for data, labels in target_loader:    
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            class_accuracies[target_class] = accuracy
        return class_accuracies
    
    @staticmethod
    def get_loss(model, device, test_loader, criterion):
        model.to(device).eval()
        loss = 0
        with torch.no_grad():
            for data,labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                loss += criterion(predicted.float(), labels.float())
        return loss/len(test_loader)

    @staticmethod
    def test_backdoor(global_model, device, test_loader, attack_type, source, target, square_size):
        global_model.to(device).eval()
        total_source_labels, misclassified_as_target = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                tmp_data = data[labels==source].clone()
                tmp_labels = labels[labels==source].clone()

                if len(tmp_data) != 0 :
                    if attack_type == 'SquareBackdoor':
                        tmp_data[0, :square_size, :square_size] = 1.0

                    outputs = global_model(tmp_data)
                    _, predicted = torch.max(outputs, 1)

                    total_source_labels += tmp_labels.size(0)
                    misclassified_as_target += (predicted == target).sum().item()

        effectiveness = misclassified_as_target / total_source_labels if total_source_labels > 0 else 0
        return effectiveness

    @staticmethod
    def get_test_data(size_trigger, dataset):
        if dataset == "MNIST" :
            data_test = datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
        elif dataset == "CIFAR10" :
            trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            data_test = datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=trans_cifar_test,
                                        download=True)
        size_test = len(data_test) - size_trigger
        trigger_set, validation_set = random_split(data_test, [size_trigger, size_test])
        # Create data loaders
        trigger_loader = DataLoader(trigger_set, batch_size=size_trigger, shuffle=False, drop_last=False) if size_trigger else None
        test_loader = DataLoader(validation_set, batch_size=size_test, shuffle=False, drop_last=False)
        return trigger_loader, test_loader

    @staticmethod
    def select_clients(clients, nb_clients_per_round):
        selected_clients = random.sample(clients, nb_clients_per_round)
        return selected_clients


    @staticmethod
    def save_to_json(accuracies, dir_path, file_name):
        file_name = f"{dir_path}/{file_name}.json"
        with open(file_name, "w") as f:
            json.dump(accuracies, f)

    @staticmethod
    def read_from_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def aggregate_models(clients):
        aggregated_state_dict = {}
        total_samples = sum([client.num_samples for client in clients])

        # Initialize
        for name, param in clients[0].model.state_dict().items():
            aggregated_state_dict[name] = torch.zeros_like(param).float()
        # Aggregate the clients' models
        for client in clients:
            num_samples = client.num_samples
            weight_factor = num_samples / total_samples
            client_state_dict = client.get_model().state_dict()

            for name, param in client_state_dict.items():
                aggregated_state_dict[name] += weight_factor * param
        return aggregated_state_dict

    @staticmethod
    def one_hot_encoding(label, num_classes, device):
        one_hot = torch.eye(num_classes).to(device)[label]
        return one_hot.squeeze(1).to(device)

    # ****** Functions related to FedCVAE

    @staticmethod
    def get_prod_size(model):
        size = 0
        for param in model.parameters():
            size += np.prod(param.weight.shape)
        return size
    
    # ****** Functions related to FedGuard
    @staticmethod
    def sample_from_normal(nb_samples, dim_samples, device):
        return torch.normal(mean=0.0, std=1.0, size=(nb_samples,dim_samples), device=device)
    
    @staticmethod
    def sample_from_cat(nb_samples, device):
        return torch.randint(low=0, high=10, size=(nb_samples,1), device=device)


