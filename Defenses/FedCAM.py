from copy import deepcopy
import os
import gc
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
from Models.autoencoders import CVAE
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime 
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import random


class Server:
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cf = cf

        # Saving directory
        if not cf["with_defence"]:
            self.dir_path = f"Results/{self.cf['dataset']}/NoDefence/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        else :
            self.dir_path = f"Results/{self.cf['dataset']}/FedCAM/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"

        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        self.activation_size = cf["cvae_input_dim"]
        self.num_classes = cf["num_classes"]
        self.nb_rounds = cf["nb_rounds"]
        self.global_model = model.to(self.device) if model else MLP(self.activation_size).to(self.device)
        self.defence = cf["with_defence"]
        self.attack_type = attack_type
        self.attacker_ratio = attacker_ratio

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": attacker_ratio,
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"]
        }

        self.config_cvae = {
            "cvae_nb_ep": cf["cvae_nb_ep"],
            "cvae_lr": cf["cvae_lr"],
            "cvae_wd": cf["cvae_wd"],
            "cvae_gamma": cf["cvae_gamma"],
        }
        
        #self.train_for_test = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        #self.train_for_test_loader = DataLoader(self.train_for_test, cf["batch_size"], shuffle=False, drop_last=False)
        #self.class_indexes = [(self.train_for_test.targets == idx).nonzero().reshape(-1) for idx in range(self.cf["nb_classes"])]
        #self.class_datasets = [Subset(self.train_for_test,self.class_indexes[idx]) for idx in range(self.cf["nb_classes"])]

        print("distributing data among clients")
        if self.cf['data_dist'] == "IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        elif self.cf['data_dist'] == "non-IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        print("generating clients")

        if self.cf["dataset"] == "FEMNIST" :
            print("splitting train into trigger and test")
            self.train_data, self.trigger_loader, self.test_loader = Utils.split_train(
                self.train_data, cf["size_trigger"], cf["size_test"])
        else : 
            print("getting test data")
            self.trigger_loader, self.test_loader = Utils.get_test_data(cf["size_trigger"], cf["dataset"])
        
        #for sample in self.trigger_loader :
        #    print(f"trigger data sample : {sample[0].size()}")
        #    print(f"trigger label sample : {sample[1].size()}, {sample[1]}")
        #    
        #for sample in self.test_loader :
        #    print(f"test data sample : {sample[0].size()}")
        #    print(f"test label sample : {sample[1].size()}, {sample[1]}")
        
        self.clients = Utils.gen_clients(self.config_FL, self.attack_type, self.train_data, backdoor_label= cf["source"])              

        self.are_attackers = np.array([int(client.is_attacker) for client in self.clients])
        self.are_benign = np.array([int(not(client.is_attacker)) for client in self.clients])

        self.cvae_trained = False

        self.cvae = CVAE(
            input_dim=cf["cvae_input_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]
        ).to(self.device)


        self.accuracy = []
        self.accuracy_on_train = []
        self.class_accuracies = []
        self.accuracy_backdoor = []
        self.nb_attackers_history = []
        self.nb_attackers_passed_defence_history = []
        self.nb_benign_history = []
        self.nb_benign_passed_defence_history = []

        self.attacker_precision_hist = []
        self.attacker_recall_hist = []

        self.attacker_profiles_per_round = []
        self.passing_attacker_profiles_per_round = []
        self.benign_profiles_per_round = []
        self.passing_benign_profiles_per_round = []

        self.best_accuracy, self.best_round = 0, 0

        self.histo_selected_clients = torch.tensor([])

    def train_cvae(self):
        if self.cvae_trained:
            print("CVAE is already trained, skipping re-training.")
            return

        init_ep = 10
        labels_act = 0
        input_models_act =  torch.zeros(size=(init_ep, self.cf["size_trigger"], self.activation_size)).to(self.device)
        input_cvae_model = deepcopy(self.global_model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(input_cvae_model.parameters(), lr=self.cf["lr"], weight_decay=self.cf["wd"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(init_ep):
            for data, labels in self.trigger_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = input_cvae_model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                input_models_act[epoch] = input_cvae_model.get_activations(data)
                labels_act = labels
                break

        gm = compute_geometric_median(input_models_act.cpu(), weights=None)
        input_models_act = input_models_act - gm.median.to(self.device)
        input_models_act = torch.sigmoid(input_models_act).detach()

        num_epochs = self.config_cvae["cvae_nb_ep"]
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.config_cvae["cvae_lr"],
                                     weight_decay=self.config_cvae["cvae_wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300],
                                                         gamma=self.config_cvae["cvae_gamma"])

        for epoch in range(num_epochs):
            train_loss = 0
            loop = tqdm(input_models_act, leave=True)
            for batch_idx, activation in enumerate(loop):

                condition = Utils.one_hot_encoding(labels_act, self.num_classes, self.device)
                recon_batch, mu, logvar = self.cvae(activation, condition)
                loss = Utils.cvae_loss(recon_batch, activation, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1))

            scheduler.step()
        self.cvae_trained = True

    def compute_reconstruction_error(self, selected_clients):
        self.cvae.eval()

        clients_re = []

        clients_act = torch.zeros(size=(len(selected_clients), self.cf["size_trigger"], self.activation_size)).to(self.device)
        labels_cat = torch.tensor([]).to(self.device)

        for client_nb, client_model in enumerate(selected_clients):
            labels_cat = torch.tensor([]).to(self.device)
            for data, label in self.trigger_loader:
                data, label = data.to(self.device), label.to(self.device)
                activation = client_model.model.get_activations(data)
                clients_act[client_nb] = activation
                labels_cat = label
                break

        gm = compute_geometric_median(clients_act.cpu(), weights=None)

        #if self.cf["skip_cvae"]: 
        #    for client_act in clients_act : 
        #        mse = F.mse_loss(gm.median.to(self.device), client_act, reduction='mean').item()
        #        clients_re.append(mse)
        #    return clients_re

        clients_act = clients_act - gm.median.to(self.device)
        # clients_act = torch.abs(clients_act)
        clients_act = torch.sigmoid(clients_act)

        for client_act in clients_act:
            condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            recon_batch, _, _ = self.cvae(client_act, condition)
            mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
            clients_re.append(mse)

        return clients_re

    def run(self):
        t_start = datetime.datetime.now()

        if self.defence:
            if not self.cvae_trained:
                self.train_cvae()
                self.cvae_trained = True
                #if not self.cf["skip_cvae"]:
                #    self.train_cvae()
                #    self.cvae_trained = True
    
        # REVERT 
        total_attackers_passed = 0 
        total_attackers = 0

        for rounds in range(self.cf["nb_rounds"]):
            
            torch.cuda.empty_cache()

            selected_clients = Utils.select_clients(self.clients, self.config_FL["nb_clients_per_round"])
                                                    
            for client in tqdm(selected_clients):
                client.set_model(deepcopy(self.global_model).to(self.device))
                client.train(self.cf)

            if self.defence:
                clients_re = self.compute_reconstruction_error(selected_clients)
                clients_re_np = np.array(clients_re)
                valid_values = clients_re_np[np.isfinite(clients_re_np)]

                max_of_re = np.max(valid_values)
                mean_of_re = np.mean(valid_values)

                clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), max_of_re,
                                                  clients_re_np)

                selected_clients_array = np.array(selected_clients)
                good_updates = selected_clients_array[clients_re_without_nan < mean_of_re]
                for client in selected_clients_array[clients_re_without_nan >= mean_of_re]:
                    client.suspect()
            else:
                good_updates = selected_clients

            self.histo_selected_clients = torch.cat((self.histo_selected_clients,
                                                     torch.tensor([client.id for client in good_updates])))

            nb_attackers = np.array([client.is_attacker for client in selected_clients]).sum()
            nb_benign = np.array([not client.is_attacker for client in selected_clients]).sum()
            nb_attackers_passed = np.array([client.is_attacker for client in good_updates]).sum()
            nb_benign_passed = np.array([not client.is_attacker for client in good_updates]).sum()

            self.nb_attackers_history.append(nb_attackers)
            self.nb_attackers_passed_defence_history.append(nb_attackers_passed)
            self.nb_benign_history.append(nb_benign)
            self.nb_benign_passed_defence_history.append(nb_benign_passed)

            print("Total of Selected Clients ", len(selected_clients), ", Number of attackers ", nb_attackers,
                  ", Total of attackers passed defense ", nb_attackers_passed, " out of ", len(good_updates), " total updates")
            

            # Aggregation step
            self.global_model.load_state_dict(Utils.aggregate_models(good_updates))


            # Add the accuracy of the current global model to the accuracy list
            #self.accuracy_on_train.append(Utils.test(self.global_model, self.device, self.train_for_test_loader))
            #self.class_accuracies.append(Utils.get_class_accuracies(self.global_model, self.device, self.class_datasets, nb_classes=self.cf["nb_classes"]))
            self.accuracy.append(test_acc := Utils.test(self.global_model, self.device, self.test_loader))

            if test_acc > self.best_accuracy :
                self.best_accuracy, self.best_round = test_acc, rounds+1

            if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"] :
                self.accuracy_backdoor.append(Utils.test_backdoor(self.global_model, self.device, self.test_loader,
                                                                  self.attack_type, self.cf["source"],
                                                                  self.cf["target"], self.cf["square_size"]))

            print(f"Round {rounds + 1}/{self.cf['nb_rounds']} server test accuracy: {self.accuracy[-1] * 100:.2f}%")
            if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"] :
                print(f"Round {rounds + 1}/{self.cf['nb_rounds']} attacker accuracy: {self.accuracy_backdoor[-1] * 100:.2f}%")
            
            # Clean up after update step (to save GPU memory)
            for client in tqdm(selected_clients):
                client.remove_model()
            
            # METRICS
            total_attackers_passed += nb_attackers_passed
            total_attackers += nb_attackers

            round_suspects = np.array([int(client.is_suspect) for client in selected_clients])
            round_unsespected = np.array([int(not(client.is_suspect)) for client in selected_clients])
            round_attackers = np.array([int(client.is_attacker) for client in selected_clients])
            round_benign = np.array([int(not(client.is_attacker)) for client in selected_clients])

            TP = sum(round_suspects*round_attackers)
            FP = sum(round_suspects*round_benign)
            FN = sum(round_unsespected*round_attackers)

            self.attacker_precision_hist.append(TP/(TP+FP+1e-10))
            self.attacker_recall_hist.append(TP/(TP+FN+1e-10))
            print(f"attacker detection recall : {self.attacker_recall_hist[-1]}")

            attacker_profile_this_round = []
            passing_attacker_profile_this_round = []
            benign_profile_this_round = []
            passing_benign_profile_this_round = []

            for client in selected_clients:
                if client.is_attacker :
                    attacker_profile_this_round.append(client.num_samples)
                else : 
                    benign_profile_this_round.append(client.num_samples)
            
            for client in good_updates:
                if client.is_attacker :
                    passing_attacker_profile_this_round.append(client.num_samples)
                else : 
                    passing_benign_profile_this_round.append(client.num_samples)
            
            self.passing_attacker_profiles_per_round.append(passing_attacker_profile_this_round)
            self.attacker_profiles_per_round.append(attacker_profile_this_round)
            self.passing_benign_profiles_per_round.append(passing_benign_profile_this_round)
            self.benign_profiles_per_round.append(benign_profile_this_round)
        
        print(f"finished running server in {datetime.datetime.now() - t_start}")

        # Save the training config 
        Utils.save_to_json(self.cf, self.dir_path, "run_config.json")

        # Saving The accuracies of the Global model on the testing set and the backdoor set
        Utils.save_to_json(self.accuracy, self.dir_path, f"test_accuracy_{self.cf['nb_rounds']}")
        if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path,
                               f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.pdf"
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=1000,
                        save_path=save_path)
        
        # Saving the percentage of attackers blocked
        Utils.save_to_json((total_attackers_passed/total_attackers)*100, self.dir_path, f"successful_attacks")
        # Detection qulity metrics to JSON
        Utils.save_to_json(self.attacker_precision_hist, self.dir_path, f"attacker_detection_precision_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.attacker_recall_hist, self.dir_path, f"attacker_detection_recall_{self.cf['nb_rounds']}")
        # Attacker profiles to json
        Utils.save_to_json(self.attacker_profiles_per_round, self.dir_path, f"attacker_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.passing_attacker_profiles_per_round, self.dir_path, f"passing_attacker_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.benign_profiles_per_round, self.dir_path, f"benign_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.passing_benign_profiles_per_round, self.dir_path, f"passing_benign_profiles_{self.cf['nb_rounds']}")

        # Plotting the testing accuracy of the global model
        title_info = f"Test Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Test_Accuracy_{self.cf['nb_rounds']}.pdf"
        Utils.plot_accuracy(self.accuracy, x_info='Round', y_info='Test Accuracy', title_info=title_info,
                            save_path=save_path)

        if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
            # Plotting the backdoor accuracy
            title_info = f"Backdoor Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
            save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Backdoor_Accuracy_{self.cf['nb_rounds']}.pdf"
            Utils.plot_accuracy(self.accuracy_backdoor, x_info='Round', y_info='backdoor Accuracy',
                                title_info=title_info, save_path=save_path)

        ## Plotting the histogram of the defense system
        #Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
        #                     self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
        #                     self.attack_type, self.defence, self.dir_path, success_rate=f"{(1-(total_attackers_passed/total_attackers))*100:.2f}%",
        #                     attacker_ratio=self.attacker_ratio)

        # Print some stats 
        print(f"In total : Number of attackers : {total_attackers}, \
        \nTotal of attackers passed defense : {total_attackers_passed} ({(total_attackers_passed/total_attackers)*100:.2f}%)")
        
        print(f"Best test accuracy : {self.best_accuracy*100:.2f}. Achieved in {self.best_round} rounds")

