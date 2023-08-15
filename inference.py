import os
import time
import random
import argparse
import yaml

import numpy as np
import torch

from torchvision import transforms

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib

from utils import *
from model import ResVAE
from dataset import CellDataset

import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='VAE Inference')
    parser.add_argument('--config', default='config/VAE-latentdim_128-sigma_0.001-bg_0.5.yaml', type=str, help='Path to the config file.')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    task = config['task']

    for dirpath, dirnames, filenames in os.walk(os.path.join(config['ckpt_dir'], task)):
        for filename in filenames:
            if 'Epoch_500' in filename:
                ckpt_path = os.path.join(dirpath, filename)
                break
            break
        
    output_path = os.path.join('output/', task)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set random seed
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
        transforms.Resize((config['img_size'], config['img_size']))
    ])

    trans_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['img_size'], config['img_size']))
    ])

    dataset = CellDataset(config['data_path'], transform=trans, transform_mask=trans_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = ResVAE(config['in_channels'], config['latent_dim'], config['use_bn'], config['dropout'], config['layer_list']).to(device)
    model.load_state_dict(torch.load(ckpt_path)['model'])

    model.eval()
    for i, (data, mask) in enumerate(dataloader):
        data, mask = data.to(device), mask.to(device)
        with torch.no_grad():
            recon, mu, logvar, fg_var = model(data)
            original = data.cpu().detach().numpy()
            recon = recon.cpu().detach().numpy()
            mu = mu.cpu().detach().numpy()
            logvar = logvar.cpu().detach().numpy()
            break

    torch.cuda.empty_cache()

    pca = PCA(n_components=2)
    pca.fit(mu)
    mu_pca = pca.transform(mu)

    # plt.figure()
    # plt.scatter(mu_pca[:, 0], mu_pca[:, 1])
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(os.path.join(output_path, 'latent_space.png'))

    original = original.transpose(0, 2, 3, 1) * np.array(config['std']) + np.array(config['mean'])
    recon = recon.transpose(0, 2, 3, 1) * np.array(config['std']) + np.array(config['mean'])

    original = original.transpose(0, 3, 1, 2)
    recon = recon.transpose(0, 3, 1, 2)

    plt.figure(figsize=(15, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(original[i][0], cmap='gray', vmin=2000, vmax=15000)
        plt.colorbar()
    plt.suptitle('Original Images', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'original.png'))

    plt.figure(figsize=(15, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(recon[i][0], cmap='gray', vmin=2000, vmax=15000)
        plt.colorbar()
    plt.suptitle('Reconstructed Images', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'reconstruction.png'))