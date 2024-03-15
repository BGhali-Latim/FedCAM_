import unittest
import torch
from Models.MLP import MLP, DropoutMLP
from Models.CNN import CNN, GuardCNN, CifarCNN, Net, CifarCNN2
from Configs.cf_NoDef import configs_noDef as cf_ndf
from Configs.cf_fedCAM import configs_fedCAM as cf
from Configs.cf_fedCVAE import configs_fedCVAE as cvae
from Configs.cf_fedGuard import configs_fedGuard
#from Configs.cf_fedGuard_1000 import configs_fedGuard as cf_fedGuard_1000
import argparse
import random

class TestMain(unittest.TestCase):

    def test_main(self, algo="fedCam", attack_type="NoAttack", attacker_ratio=0.3, dataset = "FashionMNIST"):
        # Fix seed
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Set the device to GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        # Set model
        model = CNN().to(device)

        # Import defese server
        if algo == "noDefense":
            from Defenses.FedCAM import Server
            server = Server(cf=cf_ndf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCAM":
            from Defenses.FedCAMServer import DefenseServer
            server = DefenseServer(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCAM_cos":
            from Defenses.FedCAM_cos import Server
            server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCAM2_cos":
            from Defenses.FedCAM_cos_2 import Server
            server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCWR":
            from Defenses.FedCWR import Server
            server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCAM2":
            from Defenses.FedCAM2 import Server
            server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCVAE":  # FedCVAE in this case
            from Defenses.FedCVAE import Server
            server = Server(cf=cvae, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedGuard":
            from Defenses.FedGuard import Server
            server = Server(cf=configs_fedGuard, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        else:
            print("Please specify a valid -algo argument (e.g., fedCam, fedCvae)") 
        
        # Run scenario
        print("started")
        server.run()

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
