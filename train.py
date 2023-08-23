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
import torch.nn.functional as F

import tensorboardX
from tensorboardX import SummaryWriter
import logging

import matplotlib.pyplot as plt
import matplotlib

# Define the train one epoch function
def train_one_epoch(epoch, model, optimizer, criterion, train_loader, device, writer, logger):
    logger.info('Train Epoch: {}'.format(epoch))
    train_loss = 0
    train_kl_loss = 0
    train_recon_loss = 0
    train_mse_loss = 0

    for batch_idx, (data, mask) in tqdm.tqdm(enumerate(train_loader)):
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        x_hat, mu, log_var, pred_var = model(data)
        # predict_var = (pred_var.shape[1] >= 4)
        # predict_fg_only = (pred_var.shape[1] == 4)
        if pred_var.shape[1] >= 4:
            # print('predict fg var: ', fg_var[0])
            loss, kl_loss, recon_loss = criterion(x_hat, data, mu, log_var, mask, pred_var)
        else:
            loss, kl_loss, recon_loss = criterion(x_hat, data, mu, log_var, mask)
        loss.backward()
        optimizer.step()

        mse_loss = F.mse_loss(x_hat, data)
        train_loss += loss.item()
        train_kl_loss += kl_loss.item()
        train_recon_loss += recon_loss.item()
        train_mse_loss += mse_loss.item()
        
        if batch_idx %  (len(train_loader)//5) == 0:
            writer.add_scalar('iteration/loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('iteration/kl_loss', kl_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('iteration/recon_loss', recon_loss.item(), epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('iteration/mse_loss', mse_loss.item(), epoch * len(train_loader) + batch_idx)
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL Loss: {:.6f}\tRecon Loss: {:.6f}\tMSE Loss: {:.6f}\tLR: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), kl_loss.item(), recon_loss.item(), mse_loss.item(), optimizer.param_groups[0]['lr']))
            if pred_var.shape[1] == 8:
                writer.add_scalar('iteration/fg_var1', pred_var[0][0].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var2', pred_var[0][1].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var3', pred_var[0][2].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var4', pred_var[0][3].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/bg_var1', pred_var[0][4].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/bg_var2', pred_var[0][5].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/bg_var3', pred_var[0][6].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/bg_var4', pred_var[0][7].item(), epoch * len(train_loader) + batch_idx)
            elif pred_var.shape[1] == 4:
                writer.add_scalar('iteration/fg_var1', pred_var[0][0].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var2', pred_var[0][1].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var3', pred_var[0][2].item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('iteration/fg_var4', pred_var[0][3].item(), epoch * len(train_loader) + batch_idx)
            else:
                pass


    train_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_mse_loss /= len(train_loader)
    logger.info('====> Epoch: {} Average loss: {:.4f}\tKL Loss: {:.4f}\tRecon Loss: {:.4f}\tMSE Loss: {:.4f}\tLR: {:.6f}'.format(
        epoch, train_loss, train_kl_loss, train_recon_loss, train_mse_loss, optimizer.param_groups[0]['lr']))
    
    return train_loss, train_kl_loss, train_recon_loss, train_mse_loss

# define the validate one epoch function
def validate_one_epoch(epoch, model, criterion, val_loader, device, writer, logger):
    logger.info('Val Epoch: {}'.format(epoch))
    val_loss = 0
    val_kl_loss = 0
    val_recon_loss = 0
    val_mse_loss = 0

    with torch.no_grad():
        for batch_idx, (data, mask) in tqdm.tqdm(enumerate(val_loader)):
            data = data.to(device)
            mask = mask.to(device)
            x_hat, mu, log_var, pred_var = model(data)
            if pred_var.shape[1] >= 4:
                loss, kl_loss, recon_loss = criterion(x_hat, data, mu, log_var, mask, pred_var)
            else:
                loss, kl_loss, recon_loss = criterion(x_hat, data, mu, log_var, mask)
            mse_loss = F.mse_loss(x_hat, data)
            val_loss += loss.item()
            val_kl_loss += kl_loss.item()
            val_recon_loss += recon_loss.item()
            val_mse_loss += mse_loss.item()
            if batch_idx % (len(val_loader)//5) == 0:
                writer.add_scalar('iteration/val_loss', loss.item(), epoch * len(val_loader) + batch_idx)
                writer.add_scalar('iteration/val_kl_loss', kl_loss.item(), epoch * len(val_loader) + batch_idx)
                writer.add_scalar('iteration/val_recon_loss', recon_loss.item(), epoch * len(val_loader) + batch_idx)
                writer.add_scalar('iteration/val_mse_loss', mse_loss.item(), epoch * len(val_loader) + batch_idx)
                logger.info('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL Loss: {:.6f}\tRecon Loss: {:.6f}\tMSE Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(val_loader.dataset),
                    100. * batch_idx / len(val_loader), loss.item(), kl_loss.item(), recon_loss.item(), mse_loss.item()))
                
    val_loss /= len(val_loader)
    val_kl_loss /= len(val_loader)
    val_recon_loss /= len(val_loader)
    val_mse_loss /= len(val_loader)
    logger.info('====> Val Epoch: {} Average loss: {:.4f}\tKL Loss: {:.4f}\tRecon Loss: {:.4f}\tMSE Loss: {:.4f}'.format(
        epoch, val_loss, val_kl_loss, val_recon_loss, val_mse_loss))
    
    return data, x_hat, val_loss, val_kl_loss, val_recon_loss, val_mse_loss

# Define the training function   
def train(model, optimizer, schedular, criterion, train_loader, val_loader, device, writer, logger, num_epoch, save_interval=10, save_path=None, mean=None, std=None):
    for epoch in range(1, num_epoch + 1):
        model.train()
        train_loss, train_kl_loss, train_recon_loss, train_mse_loss = train_one_epoch(epoch, model, optimizer, criterion, train_loader, device, writer, logger)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('epoch/train_loss', train_loss, epoch+1)
        writer.add_scalar('epoch/train_kl_loss', train_kl_loss, epoch+1)
        writer.add_scalar('epoch/train_recon_loss', train_recon_loss, epoch+1)
        writer.add_scalar('epoch/train_mse_loss', train_mse_loss, epoch+1)
        schedular.step()

        model.eval()
        data, x_hat, val_loss, val_kl_loss, val_recon_loss, val_mse_loss = validate_one_epoch(epoch, model, criterion, val_loader, device, writer, logger)

        writer.add_scalar('epoch/val_loss', val_loss, epoch+1)
        writer.add_scalar('epoch/val_kl_loss', val_kl_loss, epoch+1)
        writer.add_scalar('epoch/val_recon_loss', val_recon_loss, epoch+1)
        writer.add_scalar('epoch/val_mse_loss', val_mse_loss, epoch+1)

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
            vis_path = os.path.dirname(logger.handlers[-1].baseFilename)
            tifffile.imwrite(os.path.join(vis_path, 'original_{}.tif'.format(epoch)), data[vis_idx])
            tifffile.imwrite(os.path.join(vis_path, 'reconstruction_{}.tif'.format(epoch)), x_hat[vis_idx])

            logger.info('Original image saved at {}'.format(os.path.join(vis_path, 'original_{}.tif'.format(epoch))))
            logger.info('Reconstructed image saved at {}'.format(os.path.join(vis_path, 'reconstruction_{}.tif'.format(epoch))))

    return model

# Define the testing function
def test(model, test_loader, device, logger, mean=None, std=None):
    matplotlib.use('Agg')
    model.eval()
    with torch.no_grad():
        for i, (data, mask) in tqdm.tqdm(enumerate(test_loader)):
            data = data.to(device)
            mask = mask.to(device)
            x_hat, mu, log_var, pred_var = model(data)
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


