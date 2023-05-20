#!/usr/bin/env python3
# Copyright (c) 2022, Benedikt Kolbeinsson

"""This script tests a model."""


################################### Import ###################################
import argparse
import logging
import torch

from torch.utils.data import DataLoader

from networks.network import Network, NetworkConfig
from utils.cityscapes_loader import CityscapesLoader
from utils.evaluation import Evaluator
from utils.pascal_voc_loader import PascalVOCLoader
from utils.trainer import TrainerConfig
from utils.utils import set_seed
from utils.uavid_loader import UAVidLoader
from utils.vaihingen_buildings_loader import VaihingenBuildingsLoader


#################################### Setup ####################################

def make_parser():
    """Creat an argument parser"""

    parser = argparse.ArgumentParser(description=__doc__)

    # ------------ Optional arguments ------------ #
    # Network
    parser.add_argument("--network", "-n", metavar='NET', type=str, action="store", default=TrainerConfig.network,
                        help="Network architecture", dest="network")
    # Hyperparameters
    parser.add_argument("--batch_size", "-b", metavar='B', type=int, action="store", default=TrainerConfig.batch_size,
                        help="Batch size", dest="batch_size")
    # Diffusion parameters
    parser.add_argument("--n_timesteps", metavar='T', type=int, action="store", default=NetworkConfig.n_timesteps,
                        help="Number of timesteps", dest="n_timesteps")
    parser.add_argument("--n_scales", metavar='L', type=int, action="store", default=NetworkConfig.n_scales,
                        help="Number of scales", dest="n_scales")
    parser.add_argument("--max_patch_size", metavar='P', type=int, action="store", default=NetworkConfig.max_patch_size,
                        help="Max patch size", dest="max_patch_size")
    parser.add_argument("--scale_procedure", metavar='SP', type=str, action="store", default=NetworkConfig.scale_procedure,
                        help="Scale procedure", dest="scale_procedure")
    # Ensemble
    parser.add_argument("--ensemble", metavar='E', type=int, action="store", default=1,
                        help="Number of models to ensemble", dest="ensemble")
    # Directories
    parser.add_argument("--checkpoint_dir", metavar='CD', type=str, action="store", default=TrainerConfig.checkpoint_dir,
                        help="Checkpoint directory", dest="checkpoint_dir")
    parser.add_argument("--log_dir", metavar='LG', type=str, action="store", default=TrainerConfig.log_dir,
                        help="Log directory", dest="log_dir")
    # Dataset
    parser.add_argument("--dataset", metavar='DS', type=str, action="store", default="uavid",
                        help="Dataset to be used", dest="dataset_selection")
    # Checkpoint
    parser.add_argument("--load_checkpoint", metavar='FILE', type=str, action="store", default=TrainerConfig.load_checkpoint,
                        help="Load checkpoint from a .pt file", dest="load_checkpoint")
    # Other
    parser.add_argument("--seed", "-s", metavar='S', type=int, action="store", default=TrainerConfig.seed,
                        help="Set random seed for deterministic results", dest="seed")
    parser.add_argument("--n_workers", metavar='W', type=int, action="store", default=TrainerConfig.n_workers,
                        help="Number of workers", dest="n_workers")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbosity (-v, -vv, etc)")

    return parser

def box_text(text, title=None):
    """Add a title and a box around text"""
    lines = text.splitlines()
    width = max(len(line) for line in lines) + 4
    if title:
        title = ' ' + title + ' '
        message = '┌{:─^{width}}┐\n'.format(title, width=width)
    else:
        message = '┌{:─^{width}}┐\n'.format('', width=width)
        
    for line in lines:
        message += '│{:^{width}}│\n'.format(line, width=width)
    message += '└{:─^{width}}┘'.format('', width=width)
    return message

def print_all_arguments():
    """Print all arguments"""
    message = ''
    for key, value in vars(ARGS).items():
        message += '{: >21}: {: <21}\n'.format(str(key), str(value))
    print(box_text(message, 'ARGUMENTS'))

def setup_logging():
    """Set logging level"""
    base_loglevel = logging.WARNING
    loglevel = max(base_loglevel - ARGS.verbose * 10, logging.DEBUG)
    logging.basicConfig(level=loglevel,
                        format='%(message)s')



#################################### Code ####################################






#################################### Main ####################################

def main():
    """Main entry point of the module"""
    # logging setup
    setup_logging()

    # print arguments
    print_all_arguments()

    # make deterministic (optional)
    if ARGS.seed is not None:
        set_seed(ARGS.seed)

    # define dataset
    if ARGS.dataset_selection == "cityscapes":
        test_dataset = CityscapesLoader(root='../data/cityscapes/', split='test', is_transform=True)
    elif ARGS.dataset_selection == "pascal":
        test_dataset = PascalVOCLoader(root='../data/VOC2012/', split='test', is_transform=True)
    elif ARGS.dataset_selection == "vaihingen":
        # Dataset can be downloaded from https://drive.google.com/open?id=1nenpWH4BdplSiHdfXs0oYfiA5qL42plB
        test_dataset = VaihingenBuildingsLoader(root='../data/Vaihingen_buildings/', split='test', is_transform=True)
    elif ARGS.dataset_selection == "uavid":
        test_dataset = UAVidLoader(root='../data/UAVid/', split='val', is_transform=True)

    # define dataset loader
    test_dataloader = DataLoader(test_dataset, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.n_workers)

    # define the model
    network_config = NetworkConfig(
        n_timesteps=ARGS.n_timesteps, 
        n_scales=ARGS.n_scales, 
        max_patch_size=ARGS.max_patch_size, 
        scale_procedure=ARGS.scale_procedure,
        n_classes=test_dataset.n_classes
        )
    model = Network(network_config)

    # load checkpoint if specified
    checkpoint = None
    if ARGS.load_checkpoint is not None:
        checkpoint = torch.load(ARGS.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    # use GPU if available
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model.to(device)
    logging.info("Using device: {}".format(device))

    # evaluate
    evaluator = Evaluator(model, network_config, device, test_data_loader=test_dataloader)
    evaluator.test(ensemble=ARGS.ensemble)

    



if __name__ == "__main__":
    PARSER = make_parser()
    ARGS = PARSER.parse_args()
    main()
