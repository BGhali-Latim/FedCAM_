import unittest
import torch
from Models.MLP import MLP, DropoutMLP
from Models.CNN import CNN, GuardCNN, CifarCNN, Net, CifarCNN2
from Configs.cf_NoDef import configs_noDef as cf_ndf
from Configs.cf_fedCAM import configs_fedCAM as cf
from Configs.cf_fedCAM_cos import configs_fedCAM as cf_cos
from Configs.cf_fedCAM_cos_non_iid import configs_fedCAM as cf_cos_non_iid
from Configs.cf_fedCAM2 import configs_fedCAM2
from Configs.cf_fedCVAE import configs_fedCVAE as cvae
from Configs.cf_NoDef_cifar import configs_noDef as cf_ndf_cifar
from Configs.cf_fedCAM_cifar import configs_fedCAM as cf_cifar
from Configs.cf_fedCVAE_cifar import configs_fedCVAE as cvae_cifar
from Configs.cf_fedGuard import configs_fedGuard
from Configs.cf_fedGuard_1000 import configs_fedGuard as cf_fedGuard_1000
import argparse
import random

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class TestMain(unittest.TestCase):

    def test_main(self, algo="fedCam", attack_type="NoAttack", attacker_ratio=0.3, dataset = "FashionMNIST"):
        random.seed(42)
        print("started")
        # Test for the fedCam algorithm
        if algo == "noDefense":
            from Defenses.FedCAM import Server
            #model = DropoutMLP(cf["cvae_input_dim"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN2().to(device)
                server = Server(
                    cf=cf_ndf_cifar, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            else :
                model = CNN().to(device)
                server = Server(cf=cf_ndf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedCam":
            from Defenses.FedCAM import Server
            #model = DropoutMLP(cf["cvae_input_dim"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN().to(device)
                server = Server(
                    cf=cf_cifar, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            else :
                model = CNN().to(device)
            #model = GuardCNN().to(device)
                server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedCam_cos":
            from Defenses.FedCAM_cos import Server
            #model = DropoutMLP(cf["cvae_input_dim"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN().to(device)
                server = Server(
                    cf=cf_cifar, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            else :
                model = CNN().to(device)
            #model = GuardCNN().to(device)
                server = Server(cf=cf_cos, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedCam_cos_non_iid":
            from Defenses.FedCAM_cos import Server
            #model = DropoutMLP(cf["cvae_input_dim"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN().to(device)
                server = Server(
                    cf=cf_cifar, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            else :
                model = CNN().to(device)
            #model = GuardCNN().to(device)
                server = Server(cf=cf_cos_non_iid, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedCam2":
            from Defenses.FedCAM2 import Server
            #model = DropoutMLP(cf["cvae_input_dim"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN().to(device)
            else :
                model = CNN().to(device)
            #model = GuardCNN().to(device)
            server = Server(cf=configs_fedCAM2, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        # Test for the fedCvae algorithm
        elif algo == "fedCVAE":  # FedCVAE in this case
            from Defenses.FedCVAE import Server
            # model = MLP(cf["activation_size"]).to(device)
            if dataset == "CIFAR10" :
                model = CifarCNN().to(device)
                server = Server(
                    cf=cvae_cifar, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            else :
                model = CNN().to(device)
                server = Server(cf=cvae, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedGuard":
            from Defenses.FedGuard import Server
            # model = MLP(configs_fedCAM["cvae_input_dim"]).to(device)
            model = GuardCNN()
            #model = CNN().to(device)
            server = Server(cf=configs_fedGuard, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        elif algo == "fedGuard_1000":
            from Defenses.FedGuard import Server
            # model = MLP(configs_fedCAM["cvae_input_dim"]).to(device)
            model = CNN()
            #model = CNN().to(device)
            server = Server(cf=cf_fedGuard_1000, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio)
            server.run()
        else:
            # Print a message if the algorithm argument is not valid
            print("Please specify a valid -algo argument (e.g., fedCam, fedCvae)")

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="This script corresponds to the implementation of FedCVAE and FedCAM")

    # Add an -algo argument to specify the algorithm
    parser.add_argument("-algo", type=str, help="The name of the defense system")
    parser.add_argument("-attack", type=str, help="The type of attack")
    parser.add_argument("-ratio", type=float, help="The ratio of attackers")
    parser.add_argument("-dataset", type=str, default="FashionMNIST")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the TestMain class
    test_instance = TestMain()

    # Call the test_main function with the specified algorithm from the arguments
    if args.algo:
        test_instance.test_main(algo=args.algo, attack_type=args.attack, attacker_ratio=args.ratio, dataset=args.dataset)
    else:
        # Print a message if the -algo argument is not specified in the command line
        print("Please specify the -algo argument in the command line.")
