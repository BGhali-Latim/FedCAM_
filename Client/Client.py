import torch
from tqdm import tqdm
from Utils.clientUtils import Utils

class Client:
    def __init__(self, ids=None, is_attacker=False, dataloader=None, model=None, attack_type=None, device=None, is_suspect=False) -> None:
        self.id = ids
        self.is_attacker = is_attacker
        self.is_suspect = is_suspect
        self.dataloader = dataloader
        self.model = model
        self.attack_type = attack_type
        self.device = device
        self.num_samples = len(dataloader.dataset) if dataloader else 0
        self.label_counts = {idx : 0 for idx in range(10)}
        for _, targets in self.dataloader :
            for target in targets : 
                self.label_counts[target.item()] += 1 

    def get_id(self):
        return self.id

    def is_attacker(self):
        return self.is_attacker
    
    def suspect(self):
        self.is_suspect = True
    
    def is_suspect(self):
        return self.is_suspect

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def get_attack_type(self):
        return self.attack_type

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model.to(self.device)
    
    def remove_model(self):
        self.model = None

    def set_data(self, data):
        self.dataloader = data
        self.num_samples = len(data.dataset)


    def train(self, hp):
        if self.model is None:
            raise ValueError("The model is not set. Use set_model method to set the model.")

        if self.is_attacker and self.attack_type not in ["NoAttack", "NaiveBackdoor", "SquareBackdoor", "NoiseBackdoor", "MajorityBackdoor", "TargetedBackdoor"]:
            self.apply_attack()
            return self.model

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(hp["num_epochs"]):
            for data, labels in self.dataloader:
                data, labels = data.to(device), labels.to(device)

                if self.is_attacker and self.attack_type == "NaiveBackdoor":
                    labels[labels == hp["source"]] = hp["target"]
                if self.is_attacker and self.attack_type in ["MajorityBackdoor", "TargetedBackdoor","SquareBackdoor"]:
                    data, labels = square_backdoor(data, labels, hp["source"], hp["target"], hp["square_size"])
                if self.is_attacker and self.attack_type == "NoiseBackdoor":
                    data, labels = noise_backdoor(data, labels, hp["source"], hp["target"], hp["back_noise_avg"], hp["back_noise_std"])


                outputs = self.model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

        return self.model
    
    def apply_attack(self):
        if self.attack_type == "AdditiveNoise":
            self.additive_noise()
        elif self.attack_type == "SameValue":
            self.same_value()
        elif self.attack_type == "SignFlip":
            self.sign_flip()
        elif self.attack_type == "SameSample":
            self.same_value_from_sample()
        else:
            raise ValueError("Unknown or unsupported attack type for direct parameter manipulation.")

    def additive_noise(self):
        for param in self.model.parameters():
            noise = torch.normal(mean=0.0, std=1, size=param.data.shape, device=param.device)
            param.data += noise

    def same_value(self):
        for param in self.model.parameters():
            param.data.fill_(0.01)

    def sign_flip(self):
        for param in self.model.parameters():
            param.data *= -1
    
    def save_sample_params(self): 
        sample_path = f"sample_params.sdct"
        torch.save(self.model.state_dict(), sample_path)
    
    def same_value_from_sample(self):
        sample_path = f"sample_params.sdct"
        self.model.load_state_dict(torch.load(sample_path))
    
    # FedGuard functions
    
    def set_guardCvae(self, guard_model): 
        self.guardCvae = guard_model
    
    def train_guardCvae(self, training_config):

        num_epochs = training_config["nb_ep"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = torch.optim.Adam(self.guardCvae.parameters(), lr=training_config["lr"],
                                     weight_decay=training_config["wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300],
                                                         gamma=training_config["gamma"])
        
        for epoch in range(num_epochs):
            train_loss = 0
            loop = tqdm(self.dataloader, leave=True)
            for batch_idx, (data, labels) in enumerate(loop):
                data, labels = data.to(device), labels.to(device)
                condition = Utils.one_hot_encoding(labels, self.guardCvae.condition_dim, device)
                recon_batch, mu, logvar = self.guardCvae(data, condition)
                loss = Utils.cvae_loss(recon_batch, data, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1))
            scheduler.step()
        
    def generate_synthetic_data(self, latent_space_samples, condition_samples): 
        synth_images = self.guardCvae.decode(latent_space_samples, condition_samples)
        return synth_images.view((-1,1,28,28))
    
    def increment_label_counts(self, global_label_counts): 
        return {idx : global_label_counts[idx]+self.label_counts[idx] for idx in range(10)}


def square_backdoor(data, labels, source, target, square_size):
    # Create a white square
    data = data.clone()
    labels = labels.clone()
    #print(data[labels==source].size())
    #data[labels == source][:, 0, :square_size, :square_size] = 1.0
    if data[labels == source].size(0) !=0 : # Condition for non iid where there can be no corresponding label
        data[labels == source][0, :square_size, :square_size] = 1.0
    labels[labels == source] = target
    return data, labels

def noise_backdoor(data, labels, source, target, mean, std):
    # Create a white square
    data = data.clone()
    labels = labels.clone()
    data[labels == source] += torch.normal(mean=mean, std=std, size=data[labels == source].size(), device= data.device)
    labels[labels == source] = target
    return data, labels