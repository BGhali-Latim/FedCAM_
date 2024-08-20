import torch

from Models.MLP import MLP, DropoutMLP
from Models.CNN import CNN, GuardCNN, CifarCNN, Net, CifarCNN2, CNNWithDropBlock
from Models.ConvMixer_reimpl import ConvMixer

from Configs.Base import cf_CIFAR, cf_FEMNIST, cf_MNIST, cf_FashionMNIST
from Configs.Defense import cf_fedCAM, cf_fedCAM_dev, cf_fedCAM_prod, cf_fedCVAE, cf_fedGuard, cf_FLEDGE

from Samplers import IID_sampler, CAMSampler, DirichletSampler, NaturalSampler

import Server.Server as noDef
import Defenses.FedCAMServer as fedCAM

class ConfigHandler(): 
    def __init__(self, experiment_name, sampler, dataset, architecture, defense): 
        self.sampler = sampler 
        self.dataset = dataset
        self.architecture = architecture 
        self.defense = defense
        self.name = experiment_name

        # Set the device to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model_dict = { 
            "CNN" : CNN(),
            "CNNWithDropBlock" : CNNWithDropBlock(),
            "CifarCNN" : CifarCNN(),
            "ConvMixer" : ConvMixer(dim = 576, depth = 1, kernel_size = 5, patch_size = 9, device = self.device),
        }
        # Set learning configurations 
        self.base_configs = {
            "MNIST" : cf_MNIST.configs, 
            "FashionMNIST" : cf_FashionMNIST.configs,
            "FEMNIST" : cf_FEMNIST.configs,
            "CIFAR10" : cf_CIFAR.configs 
        }
        self.defense_configs = {
            "fedCAM" : cf_fedCAM.config,
            "fedCAM_dev" : cf_fedCAM_dev.config,
            "fedCAM_prod" : cf_fedCAM_prod.config,
            "fedCVAE" : cf_fedCVAE.config,
            "fedGuard" : cf_fedGuard.config,
            "FLEDGE" : cf_FLEDGE
        }
        self.defense_servers = {
            "fedCAM" : fedCAM,
            "noDefense" : noDef
        }
    
    def set_sampler(self, config): 
        self.sampler_dict = {
            "IID" : IID_sampler.ClientSampler(config), 
            "CAM" : CAMSampler.CAMSampler(config), 
            "Dirichlet_per_class" : DirichletSampler.DirichletSampler(config, method = "Dirichlet_per_class"),
            "Dirichlet_per_client" : DirichletSampler.DirichletSampler(config, method = "Dirichlet_per_client"),
        }
        try :
            sampler = self.sampler_dict[self.sampler]
        except KeyError : 
            print("Please choose a valid client data sampling strategy")
        print(self.sampler)
        return sampler

    def load_base_config(self): 
        try :
            cf = self.base_configs[self.dataset][self.architecture]
        except KeyError : 
            print("Please choose a valid model/dataset pair")
        print(cf)
        return cf
        
    def load_defense_config(self): 
        try :
            cf = self.defense_configs[self.defense]
        except KeyError : 
            print("Please choose a valid defense config")
        print(cf)
        return cf
    
    def make_model(self): 
        try :
            model = self.model_dict[self.architecture]
        except KeyError : 
            print("Please choose a valid model")
        print(model)
        return model
    
    def make_defense_server(self, attack_type, attacker_ratio): 
        # Make config dict
        config = self.load_base_config()
        # Make server elements 
        sampler = self.set_sampler(config)
        model = self.make_model().to(self.device)
        # Import and instanciate server
        try :
            fl = self.defense_servers[self.defense]
            print(fl)
        except KeyError : 
            print("Please choose a valid defense")
        if self.defense == "noDefense": 
            return fl.Server(
                cf=config,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=self.dataset,
                sampler = sampler,
                experiment_name=self.name,
            )
        else : 
            config.update(self.load_defense_config())
            return fl.DefenseServer(
                cf=config,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=self.dataset,
                sampler = sampler,
                experiment_name=self.name,)