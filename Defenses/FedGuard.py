from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from Models.autoencoders import GuardCVAE
from Models.MLP import MLP
from Utils.Utils import Utils
from custom_datasets.Datasets import SyntheticLabeledDataset

class Server:
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.cf = cf
        self.cvae_trained = False
        self.attacker_ratio = attacker_ratio
        self.attack_type = attack_type

        # Saving directory
        if not cf["with_defence"]:
            self.dir_path = f"Results/{self.cf['dataset']}/NoDefence/{self.cf['data_dist']}_{int(self.attacker_ratio * 100)}_{self.attack_type}"
        else :
            self.dir_path = f"Results/{self.cf['dataset']}/FedGuard1000/{self.cf['data_dist']}_{int(self.attacker_ratio * 100)}_{self.attack_type}"
        
        os.makedirs(self.dir_path, exist_ok=True)

        self.num_classes = cf["num_classes"]
        self.nb_rounds = cf["nb_rounds"]
        self.global_model = model.to(self.device) if model else MLP(self.activation_size).to(self.device)
        self.defence = cf["with_defence"]
        

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": self.attacker_ratio,
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"]
        }

        self.cvae_training_config = {
            "nb_ep": cf["cvae_nb_ep"],
            "lr": cf["cvae_lr"],
            "wd": cf["cvae_wd"],
            "gamma": cf["cvae_gamma"],
        }

        # guardCvae = self.cvae = CVAE(
        #     encoder_layers=self.cf["guard_cvae_encoder"],
        #     decoder_layers=self.cf["guard_cvae_encoder"],
        #     condition_dim=self.cf["guard_cvae_condition_dim"],
        #     latent_dim=self.cf["guard_cvae_latent_dim"]).to(self.device) # TODO prepare this instead of following
        self.guardCvae = GuardCVAE(condition_dim=self.cf["condition_dim"]).to(self.device)

        print("distributing data among clients")
        if self.cf['data_dist'] == "IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        elif self.cf['data_dist'] == "non-IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        print("generating clients")
        self.clients = Utils.gen_clients(self.config_FL, self.attack_type, self.train_data, backdoor_label= cf["source"])

        self.are_attackers = np.array([int(client.is_attacker) for client in self.clients])
        self.are_benign = np.array([int(not(client.is_attacker)) for client in self.clients])

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

        _, self.test_loader = Utils.get_test_data(size_trigger=0, dataset="MNIST")

    def prepare_local_cvae(self): 
        for idx, client in enumerate(self.clients) : 
            client.set_guardCvae(deepcopy(self.guardCvae)) 
            print(f"Training guardCvae for client {idx+1}/{len(self.clients)}")
            client.train_guardCvae(self.cvae_training_config) 
        self.cvae_trained = True
    
    def compute_acc_loss(self, selected_clients, latent_space_samples, condition_samples):
        client_acc_losses = []
        loss = torch.nn.MSELoss()
        
        for client in selected_clients :
            synthetic_ds = client.generate_synthetic_data(latent_space_samples, Utils.one_hot_encoding(condition_samples, self.cf["nb_classes"], self.device))
            eval_data = SyntheticLabeledDataset(synthetic_ds, condition_samples)
            eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=4)
            acc_loss = Utils.get_loss(client.model,self.device,eval_loader,loss).cpu()
            client_acc_losses.append(acc_loss)
        
        return client_acc_losses
    
    def run(self):
        if self.defence:
            if not self.cvae_trained:
                self.prepare_local_cvae()
                self.cvae_trained = True
        
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
                latent_space_samples = Utils.sample_from_normal(self.cf["nb_samples"],self.cf["latent_dim"], device=self.device)
                condition_samples = Utils.sample_from_cat(self.cf["nb_samples"], device=self.device)
                clients_acc_losses = self.compute_acc_loss(selected_clients, latent_space_samples, condition_samples)
                clients_acc_losses_np = np.array(clients_acc_losses)
                valid_values = clients_acc_losses_np[np.isfinite(clients_acc_losses_np)]
                max_of_acc_loss = np.max(valid_values)
                mean_of_acc_loss = np.mean(valid_values)
                clients_acc_loss_without_nan = np.where(np.isnan(clients_acc_losses_np) | (clients_acc_losses_np == np.inf), max_of_acc_loss,
                                                  clients_acc_losses_np)
                
                selected_clients_array = np.array(selected_clients)
                good_updates = selected_clients_array[clients_acc_loss_without_nan < mean_of_acc_loss]
                for client in selected_clients_array[clients_acc_loss_without_nan >= mean_of_acc_loss]:
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
                  ", Total of attackers passed defense ", nb_attackers_passed, " for ", len(good_updates), "total updates")
        
            # Aggregation step
            self.global_model.load_state_dict(Utils.aggregate_models(good_updates))

            # Add the accuracy of the current global model to the accuracy list
            self.accuracy.append(Utils.test(self.global_model, self.device, self.test_loader))
            if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
                self.accuracy_backdoor.append(Utils.test_backdoor(self.global_model, self.device, self.test_loader,
                                                                  self.attack_type, self.cf["source"],
                                                                  self.cf["target"], self.cf["square_size"]))
                
            print(f"Round {rounds + 1}/{self.cf['nb_rounds']} server accuracy: {self.accuracy[-1] * 100:.2f}%")
            if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
                print(
                    f"Round {rounds + 1}/{self.cf['nb_rounds']} attacker accuracy: {self.accuracy_backdoor[-1] * 100:.2f}%")
            
            # Clean up after update step (to save GPU memory)
            for client in tqdm(selected_clients):
                client.remove_model()
                
            # REVERT
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
        
        # Saving The accuracies of the Global model on the testing set and the backdoor set
        Utils.save_to_json(self.accuracy, self.dir_path, f"test_accuracy_{self.cf['nb_rounds']}")
        if self.attack_type in ["NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path,
                               f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")
        
        # Save the training config 
        Utils.save_to_json(self.cf, self.dir_path, "run_config.json")
            
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

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.pdf"
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=1000,
                        save_path=save_path)
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
        # Plotting the histogram of the defense system
        Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
                             self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
                             self.attack_type, self.defence, self.dir_path, success_rate=f"{(1-(total_attackers_passed/total_attackers))*100:.2f}%")
        # Print some stats 
        print(f"""In total : Number of attackers : {total_attackers},
            Total of attackers passed defense : {total_attackers_passed} ({(total_attackers_passed/total_attackers)*100:.2f}%)""")












































































