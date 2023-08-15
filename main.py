import os
import time
import random
import tqdm
import argparse
import yaml

import tifffile
import torch
import torch.optim as optim

from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import logging

import matplotlib.pyplot as plt

from utils import *
from model import ResVAE
from dataset import CellDataset
from criteria import LossVAE, LossRegression

from train import train, test

import warnings


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    parser = argparse.ArgumentParser(description='VAE Training')
    parser.add_argument('--config', default='config/VAE-latentdim_128-sigma_0.001-bg_0.5.yaml', type=str, help='Path to the config file.')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    task = config['task']

    # Set random seed
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set logger
    log_dir = os.path.join(config['log_dir'], task, timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'train.log')
    logger = get_logger(log_path)

    # Set checkpoint path
    ckpt_path = os.path.join(config['ckpt_dir'], task, timestamp)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)


    # Set tensorboard writer
    writer = SummaryWriter(log_dir)

    logger.info('Task: {}'.format(task))
    logger.info('Device: {}'.format(device))
    logger.info('Config path: {}'.format(args.config))
    logger.info('Log path: {}'.format(log_path))
    logger.info('Checkpoint path: {}'.format(ckpt_path))
    logger.info('Random seed: {}'.format(config['seed']))
    logger.info('Batch size: {}'.format(config['batch_size']))
    logger.info('Data path: {}'.format(config['data_path']))
    logger.info('Number of workers: {}'.format(config['num_workers']))
    logger.info('Shuffle: {}'.format(config['shuffle']))
    logger.info('Mean: {}'.format(config['mean']))
    logger.info('Std: {}'.format(config['std']))
    logger.info('Image size: {}'.format(config['image_size']))

    logger.info('Lr scheduler: {}'.format(config['lr_scheduler']))
    logger.info('Learning rate: {}'.format(config['lr']))
    # logger.info('Step size: {}'.format(config['step_size']))
    # logger.info('Learning rate decay: {}'.format(config['lr_decay']))

    logger.info('In channels: {}'.format(config['in_channels']))
    logger.info('Image size: {}'.format(config['image_size']))
    logger.info('Latent dim: {}'.format(config['latent_dim']))
    logger.info('Use batch norm: {}'.format(config['use_bn']))
    logger.info('Dropout rate: {}'.format(config['dropout']))
    logger.info('Layer list: {}'.format(config['layer_list']))

    logger.info('Loss type: {}'.format(config['loss']['loss_type']))

    logger.info('Epochs: {}'.format(config['epochs']))
    logger.info('Checkpoint save interval (epochs): {}'.format(config['save_interval']))



    # Set dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']), 
        transforms.Resize((config['image_size'], config['image_size'])), 
    ])

    train_mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['image_size'], config['image_size'])),
    ])
    logger.info('Train transform: {}'.format(train_transform))
    logger.info('Train mask transform: {}'.format(train_mask_transform))

    logger.info('Loading dataset from {} ...'.format(config['data_path']))
    train_dataset = CellDataset(config['data_path'], transform=train_transform, transform_mask=train_mask_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'], pin_memory=True, drop_last=True)

    # Set model
    logger.info('Building model ...')
    model = ResVAE(config['in_channels'], config['latent_dim'], config['use_bn'], config['dropout'], config['layer_list']).to(device)
    logger.info('Model: {}'.format(model))

    # Set optimizer
    logger.info('Building optimizer ...')
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    if config['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['lr_decay'])
    elif config['lr_scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['lr_min'])
    else:
        raise NotImplementedError
    
    logger.info('Optimizer: {}'.format(optimizer))
    logger.info('Scheduler: {}'.format(scheduler))

    # Set criterion
    logger.info('Building criterion ...')
    loss_type = config['loss']['loss_type']
    logger.info('Loss type: {}'.format(loss_type))
    if loss_type == 'LossVAE':
        criterion = LossVAE(config['loss']['sigma'], config['loss']['bg_var']).to(device)
        logger.info('Sigma: {}'.format(config['loss']['sigma']))
        logger.info('Background variance: {}'.format(config['loss']['bg_var']))
    elif loss_type == 'LossRegression':
        criterion = LossRegression(config['loss']['bg_var'], config['latent_dim']).to(device)
        logger.info('Background variance: {}'.format(config['loss']['bg_var']))
    else:
        raise NotImplementedError

    # Start training
    logger.info('Start training ...')
    train(model, optimizer, scheduler, criterion, train_loader, device, writer, logger, config['epochs'], config['save_interval'], ckpt_path, config['mean'], config['std'])

    # Test
    logger.info('Start testing ...')
    test(model, train_loader, device, logger, config['mean'], config['std'])

# python -W ignore main.py --config config/VAE-latentdim_128-sigma_0.001-bg_0.5.yaml