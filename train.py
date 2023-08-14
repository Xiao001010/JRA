import os
import time
import random
import tqdm
import argparse
import yaml

import tifffile
import numpy as np
import torch
import torch.optim as optim

import tensorboardX
from tensorboardX import SummaryWriter
import logging

import matplotlib.pyplot as plt
import matplotlib

# Define the training function
def train_one_epoch(epoch, model, optimizer, criterion, train_loader, device, writer, logger):
    logger.info('Train Epoch: {}'.format(epoch))
    train_loss = 0
    train_kl_loss = 0
    train_recon_loss = 0
    for batch_idx, (data, mask) in tqdm.tqdm(enumerate(train_loader)):
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(data)
        loss, kl_loss, recon_loss = criterion(x_hat, data, mu, log_var, mask)
        loss.backward()
        train_loss += loss.item()
        train_kl_loss += kl_loss.item()
        train_recon_loss += recon_loss.item()
        optimizer.step()
        if batch_idx %  (len(train_loader)//5) == 0:
            writer.add_scalar('iteration/loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('iteration/kl_loss', kl_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('iteration/recon_loss', recon_loss.item(), epoch * len(train_loader) + batch_idx)
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL Loss: {:.6f}\tRecon Loss: {:.6f}\tlr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), kl_loss.item(), recon_loss.item(), optimizer.param_groups[0]['lr']))

    train_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return data, x_hat, train_loss, train_kl_loss, train_recon_loss
    
def train(model, optimizer, schedular, criterion, train_loader, device, writer, logger, epochs, save_interval=10, save_path=None, mean=None, std=None):
    model.train()
    for epoch in range(1, epochs + 1):
        data, x_hat, train_loss, kl_loss, recon_loss = train_one_epoch(epoch, model, optimizer, criterion, train_loader, device, writer, logger)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('epoch/loss', train_loss, epoch+1)
        writer.add_scalar('epoch/kl_loss', kl_loss, epoch+1)
        writer.add_scalar('epoch/recon_loss', recon_loss, epoch+1)
        schedular.step()
        if epoch % save_interval == 0:
            if save_path:
                ckt_path = os.path.join(save_path, 'VAE-Epoch_{}-Loss_{:.4f}.pth'.format(epoch, train_loss))
                checkpoint = {'epoch': epoch,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'schedular': schedular.state_dict()}
                torch.save(checkpoint, ckt_path)
                logger.info('Model saved at {}'.format(ckt_path))

                data = data.cpu().detach().numpy()
                x_hat = x_hat.cpu().detach().numpy()
                
                vis_idx = random.randint(0, len(data))
                if mean:
                    # print('=================1', data.shape)
                    data = (data.transpose(0, 2, 3, 1) * std + mean).transpose(0, 3, 1, 2)
                    x_hat = (x_hat.transpose(0, 2, 3, 1) * std + mean).transpose(0, 3, 1, 2)
                # print('=================2', data.shape)
                # print('=================3', len(train_loader.dataset), len(train_loader), len(data), len(x_hat))
                # print('=================4', vis_idx, data[vis_idx].shape, x_hat[vis_idx].shape)
                writer.add_image('original1', data[vis_idx][0][np.newaxis, :, :], epoch+1)
                writer.add_image('original2', data[vis_idx][1][np.newaxis, :, :], epoch+1)
                writer.add_image('original3', data[vis_idx][2][np.newaxis, :, :], epoch+1)
                writer.add_image('original4', data[vis_idx][3][np.newaxis, :, :], epoch+1)
                writer.add_image('reconstruction1', x_hat[vis_idx][0][np.newaxis, :, :], epoch+1)
                writer.add_image('reconstruction2', x_hat[vis_idx][1][np.newaxis, :, :], epoch+1)
                writer.add_image('reconstruction3', x_hat[vis_idx][2][np.newaxis, :, :], epoch+1)
                writer.add_image('reconstruction4', x_hat[vis_idx][3][np.newaxis, :, :], epoch+1)
                vis_path = os.path.dirname(logger.handlers[-1].baseFilename)
                tifffile.imwrite(os.path.join(vis_path, 'original_{}.tif'.format(epoch)), data[vis_idx])
                tifffile.imwrite(os.path.join(vis_path, 'reconstruction_{}.tif'.format(epoch)), x_hat[vis_idx])
    return model

# Define the testing function
def test(model, test_loader, device, logger, mean=None, std=None):
    matplotlib.use('Agg')
    model.eval()
    with torch.no_grad():
        for i, (data, mask) in tqdm.tqdm(enumerate(test_loader)):
            data = data.to(device)
            mask = mask.to(device)
            x_hat, mu, log_var = model(data)
            break

    data = data.cpu().detach().numpy()
    x_hat = x_hat.cpu().detach().numpy()

    if mean:
        data = (data.transpose(0, 2, 3, 1) * std + mean).transpose(0, 3, 1, 2)
        x_hat = (x_hat.transpose(0, 2, 3, 1) * std + mean).transpose(0, 3, 1, 2)

    vis_path = os.path.dirname(logger.handlers[-1].baseFilename)
    # visualize original images
    plt.figure(figsize=(15, 10))
    # plt.suptitle('Original Images', fontsize=20)
    for i in range(16): 
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i][0], cmap='gray', vmin=2000, vmax=15000)
        plt.colorbar()
    plt.savefig(os.path.join(vis_path, 'original.png'))

    # visualize reconstructed images
    plt.figure(figsize=(15, 10))
    # plt.suptitle('Reconstructed Images', fontsize=20)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(x_hat[i][0], cmap='gray', vmin=2000, vmax=15000)
        plt.colorbar()
    plt.savefig(os.path.join(vis_path, 'reconstruction.png'))


