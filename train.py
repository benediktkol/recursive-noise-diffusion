#!/usr/bin/env python3
# Copyright (c) 2022, Benedikt Kolbeinsson

"""This script trains a diffusion model."""


################################### Import ###################################
import argparse
import logging
import torch

from torch.utils.data import DataLoader

from networks.network import Network, NetworkConfig
from utils.cityscapes_loader import CityscapesLoader
from utils.pascal_voc_loader import PascalVOCLoader
from utils.trainer import Trainer, TrainerConfig
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
    parser.add_argument("--epochs", "-e", metavar='E', type=int, action="store", default=TrainerConfig.max_epochs,
                        help="Max number of epochs", dest="epochs")
    parser.add_argument("--batch_size", "-b", metavar='B', type=int, action="store", default=TrainerConfig.batch_size,
                        help="Batch size", dest="batch_size")
    parser.add_argument("--learning_rate", "-l", metavar='LR', type=float, action="store", default=TrainerConfig.learning_rate,
                        help="Learning rate", dest="learning_rate")
    parser.add_argument("--momentum", "-m", metavar='M', type=float, action="store", default=TrainerConfig.momentum,
                        help="Momentum", dest="momentum")
    parser.add_argument("--weight_decay", "-w", metavar='WD', type=float, action="store", default=TrainerConfig.weight_decay,
                        help="Weight decay", dest="weight_decay")
    parser.add_argument("--lr_decay", "-d", metavar='D', type=bool, action="store", default=TrainerConfig.lr_decay,
                        help="Use learning rate decay", dest="lr_decay")
    parser.add_argument("--lr_decay_gamma", "-g", metavar='G', type=float, action="store", default=TrainerConfig.lr_decay_gamma,
                        help="Learning rate decay gamma", dest="lr_decay_gamma")
    # Diffusion parameters
    parser.add_argument("--n_timesteps", metavar='T', type=int, action="store", default=NetworkConfig.n_timesteps,
                        help="Number of timesteps", dest="n_timesteps")
    parser.add_argument("--n_scales", metavar='L', type=int, action="store", default=NetworkConfig.n_scales,
                        help="Number of scales", dest="n_scales")
    parser.add_argument("--max_patch_size", metavar='P', type=int, action="store", default=NetworkConfig.max_patch_size,
                        help="Max patch size", dest="max_patch_size")
    parser.add_argument("--scale_procedure", metavar='SP', type=str, action="store", default=NetworkConfig.scale_procedure,
                        help="Scale procedure (loop or linear)", dest="scale_procedure")
    # Diffusion other options
    parser.add_argument("--train_on_n_scales", metavar='NS', type=int, action="store", default=NetworkConfig.n_scales + 1,
                        help="Only train first NS scales", dest="train_on_n_scales")
    parser.add_argument("--not_recursive", action="store_true", default=False,
                        help="Do not use recursive diffusion", dest="not_recursive")
    # Directories
    parser.add_argument("--checkpoint_dir", metavar='CD', type=str, action="store", default=TrainerConfig.checkpoint_dir,
                        help="Checkpoint directory", dest="checkpoint_dir")
    parser.add_argument("--log_dir", metavar='LG', type=str, action="store", default=TrainerConfig.log_dir,
                        help="Log directory", dest="log_dir")
    # Dataset
    parser.add_argument("--dataset", metavar='DS', type=str, action="store", default=TrainerConfig.dataset_selection,
                        help="Dataset to be used", dest="dataset_selection")
    # Checkpoint
    parser.add_argument("--load_checkpoint", metavar='FILE', type=str, action="store", default=TrainerConfig.load_checkpoint,
                        help="Load checkpoint from a .pt file", dest="load_checkpoint")
    parser.add_argument("--weights_only", action="store_true", default=False,
                        help="Load weights only", dest="weights_only")
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
        train_dataset = CityscapesLoader(root='../data/cityscapes/', split='train', is_transform=True)
        val_dataset = CityscapesLoader(root='../data/cityscapes/', split='val', is_transform=True)
    elif ARGS.dataset_selection == "pascal":
        train_dataset = PascalVOCLoader(root='../data/VOC2012/', split='train', is_transform=True, img_size=512)
        val_dataset = PascalVOCLoader(root='../data/VOC2012/', split='val', is_transform=True, img_size=512)
    elif ARGS.dataset_selection == "vaihingen":
        train_dataset = VaihingenBuildingsLoader(root='../data/Vaihingen_buildings/', split='train', is_transform=True)
        val_dataset = VaihingenBuildingsLoader(root='../data/Vaihingen_buildings/', split='val', is_transform=True)
    elif ARGS.dataset_selection == "uavid":
        train_dataset = UAVidLoader(root='../data/UAVid/', split='train', is_transform=True)
        val_dataset = UAVidLoader(root='../data/UAVid/', split='val', is_transform=True)

    assert ARGS.dataset_selection in ["cityscapes", "pascal", "vaihingen", "uavid"], "Supported datasets are: cityscapes, pascal, vaihingen, uavid"

    # define dataset loader
    train_dataloader = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.n_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.n_workers)

    # define the model
    network_config = NetworkConfig(
        n_timesteps=ARGS.n_timesteps, 
        n_scales=ARGS.n_scales, 
        max_patch_size=ARGS.max_patch_size, 
        scale_procedure=ARGS.scale_procedure,
        n_classes=train_dataset.n_classes
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
        model.cuda()
    logging.info("Using device: {}".format(device))

    # define trainer
    trainer_config = TrainerConfig(
        max_epochs=ARGS.epochs, batch_size=ARGS.batch_size, 
        learning_rate=ARGS.learning_rate, momentum=ARGS.momentum,
        weight_decay=ARGS.weight_decay, lr_decay=ARGS.lr_decay,
        lr_decay_gamma=ARGS.lr_decay_gamma, checkpoint_dir=ARGS.checkpoint_dir,
        log_dir=ARGS.log_dir, load_checkpoint=ARGS.load_checkpoint,
        n_workers=ARGS.n_workers, network=ARGS.network, 
        train_on_n_scales=ARGS.train_on_n_scales, not_recursive=ARGS.not_recursive,
        dataset_selection=ARGS.dataset_selection, 
        device=device, checkpoint=checkpoint, weights_only=ARGS.weights_only
        )
    trainer = Trainer(model, network_config, trainer_config, train_dataloader, val_dataloader)

    # train model
    trainer.train()
    



if __name__ == "__main__":
    PARSER = make_parser()
    ARGS = PARSER.parse_args()
    main()
