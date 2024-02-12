import copy
import os
import numpy as np
import tqdm
import torch
import torch.nn.functional as F

from geom_median.torch import compute_geometric_median

from Models.autoencoders import CVAE
from Models.MLP import MLP
from Utils.Utils import Utils


class Server:
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cf = cf
        self.attacker_ratio = attacker_ratio
        self.attack_type = attack_type

        # Saving directory
        if not cf["with_defence"]:
            self.dir_path = f"Results/{self.cf['dataset']}/NoDefence/{self.cf['data_dist']}_{int(self.attacker_ratio * 100)}_{self.attack_type}"
        else :
            self.dir_path = f"Results/{self.cf['dataset']}/FedCVAE/{self.cf['data_dist']}_{int(self.attacker_ratio * 100)}_{self.attack_type}"

        os.makedirs(self.dir_path, exist_ok=True)


        self.eps = cf["eps"]
        self.nb_rounds = cf["nb_rounds"]
        self.activation_size = cf["activation_size"]

        self.global_model = model.to(self.device) if model else MLP(self.activation_size).to(self.device)

        self.defence = cf["with_defence"]
        self.attack_type = attack_type

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": self.attacker_ratio,
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"]
        }

        print("distributing data among clients")
        if self.cf['data_dist'] == "IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        elif self.cf['data_dist'] == "non-IID" :
            self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"])
        print("generating clients")
        self.clients = Utils.gen_clients(self.config_FL, self.attack_type, self.train_data)

        self.validation_loader, self.test_loader = Utils.get_test_data(cf["validation_size"], dataset=cf["dataset"])

        # Done
        total_weights = sum(param.numel() for param in self.global_model.parameters()  if param.dim() > 1)

        # selecting the indices that will be fed to the CVAE
        self.indices = np.random.choice(total_weights, self.cf["selected_weights_dim"], replace=False)

        self.cvae = CVAE(
            input_dim=self.cf["selected_weights_dim"],
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

    def one_hot_encoding(self, current_round):
        one_hot = torch.zeros(self.cf['condition_dim']).to(self.device)
        one_hot[current_round] = 1.0
        return one_hot

    def gen_surrogate_vectors(self, selected_clients):

        surrogate_vectors = [torch.cat([p.data.view(-1) for p in client.model.parameters() if p.dim() > 1])[
                self.indices].detach().cpu() for client in selected_clients]

        return surrogate_vectors

    def process_surrogate_vectors(self, surrogate_vectors):
        geo_median = compute_geometric_median(surrogate_vectors, weights=None, eps=self.cf["eps"], maxiter=self.cf["iter"])  # equivalent to `weights = torch.ones(n)`.
        geo_median = geo_median.median
        processed_vectors = [surrogate_vector - geo_median for surrogate_vector in surrogate_vectors]
        return processed_vectors

    def compute_reconstruction_error(self, processed_vectors, current_round):
        self.cvae.eval()
        clients_re = []
        condition = self.one_hot_encoding(current_round).unsqueeze(0).to(self.device)

        for processed_vector in processed_vectors:
            processed_vector = processed_vector.unsqueeze(0).to(self.device)
            recon_batch, _, _ = self.cvae(processed_vector, condition)
            mse = F.mse_loss(recon_batch, processed_vector, reduction='mean').item()
            clients_re.append(mse)

        return clients_re

    def run(self):

        total_attackers_passed = 0 
        total_attackers = 0

        for rounds in range(self.cf["nb_rounds"]):
            selected_clients = Utils.select_clients(self.clients, self.config_FL["nb_clients_per_round"])

            for client in selected_clients:
                client.set_model(copy.deepcopy(self.global_model).to(self.device))
                client.train(self.cf)

            if self.defence:
                surrogate_weights = self.gen_surrogate_vectors(selected_clients)
                processed_vectors = self.process_surrogate_vectors(surrogate_weights)

                clients_re = self.compute_reconstruction_error(processed_vectors, rounds)
                clients_re_np = np.array(clients_re)

                mean_of_re = np.mean(clients_re_np)

                selected_clients_np = np.array(selected_clients)
                good_updates = selected_clients_np[clients_re_np < mean_of_re]
                for client in selected_clients_np[clients_re_np >= mean_of_re]:
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
                  ", Total of attackers passed defense ", nb_attackers_passed, " from ", len(good_updates))

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
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path, f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.pdf"
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=1000,
                        save_path=save_path)

        # Saving the percentage of attackers blocked
        Utils.save_to_json((total_attackers_passed/total_attackers)*100, self.dir_path, f"successful_attacks")
        # Detection qulity metrics to JSON
        Utils.save_to_json(self.attacker_precision_hist, self.dir_path, f"Attacker_detection_precision_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.attacker_recall_hist, self.dir_path, f"Attacker_detection_recall_{self.cf['nb_rounds']}")
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

        # Plotting the histogram of the defense system
        Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
                             self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
                             self.attack_type, self.defence, self.dir_path)

