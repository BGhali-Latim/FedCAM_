import unittest
import argparse
import random
import torch
from Run import Handler 

class TestMain(unittest.TestCase):

    def test_main(
        self,
        dataset="FashionMNIST",
        architecture = "CNN",
        experiment="debug",
        sampling = "IID",
        algo="fedCam",
        attack_type="NoAttack",
        attacker_ratio=0.3,
        lamda = 0.2
    ):
        # Fix seed
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Make server
        config_handler = Handler.ConfigHandler(experiment_name=experiment, sampler=sampling, dataset=dataset, architecture=architecture, defense=algo)
        server = config_handler.make_defense_server(attack_type=attack_type, attacker_ratio=attacker_ratio)

        # Run scenario
        print("started")
        server.run()


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script corresponds to the implementation of FedCVAE and FedCAM"
    )

    # Add an -algo argument to specify the algorithm
    parser.add_argument("-algo", type=str, help="The name of the defense system")
    parser.add_argument("-attack", type=str, help="The type of attack")
    parser.add_argument("-ratio", type=float, help="The ratio of attackers")
    parser.add_argument("-dataset", type=str, default="FashionMNIST")
    parser.add_argument("-experiment", type=str, default="debug")
    parser.add_argument("-sampling", type=str, default="IID")
    parser.add_argument("-lamda", type=float, default=0.2)
    parser.add_argument("-architecture", type=str, default="CNN")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the TestMain class
    test_instance = TestMain()

    # Call the test_main function with the specified algorithm from the arguments
    if args.algo:
        test_instance.test_main(
            algo=args.algo,
            architecture=args.architecture,
            attack_type=args.attack,
            attacker_ratio=args.ratio,
            dataset=args.dataset,
            experiment=args.experiment,
            sampling = args.sampling,
            lamda = args.lamda
        )
    else:
        # Print a message if the -algo argument is not specified in the command line
        print("Please specify the -algo argument in the command line.")
