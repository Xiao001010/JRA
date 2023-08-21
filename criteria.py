import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the loss function: KL divergence + reconstruction loss
class LossVAE(nn.Module):

    def __init__(self, sigma=0.1, bg_var=1.0):
        super(LossVAE, self).__init__()
        self.sigma = sigma
        self.bg_var = bg_var
        self.GaussianNLL = nn.GaussianNLLLoss(reduction='none')

    def forward(self, x_hat, x, mu, log_var, mask, pred_var=None):
        kl_loss = self.kl_divergence(mu, log_var)
        recon_loss = self.reconstruction_loss(x_hat, x, mask, pred_var=pred_var)
        loss = kl_loss + recon_loss
        return loss, kl_loss, recon_loss

    @staticmethod
    def kl_divergence(mu, log_var):
        """
        Compute the KL divergence between given distribution q(z|x) and standard normal distribution
        :param mu: mean vector produced by the encoder, tensor of shape (B, latent_dim)
        :param log_var: log sigma vector produced by the encoder, tensor of shape (B, latent_dim)
        :return: KL divergence between q(z|x) and p(z), where p(z)~N(0,I).
        """
        kl = 0.5 * torch.sum((torch.exp(log_var) + torch.square(mu) - 1 - log_var), -1)
        return torch.mean(kl)

    def reconstruction_loss(self, x_hat, x, mask, pred_var=None):
        """
        Compute the reconstruction loss
        :param x: 2D
        :param x_hat: output of the decoder, considered as the mean of a distribution
        :return: reconstruction
        """
        var = torch.zeros_like(x_hat)
        mask = mask.repeat(1, 4, 1, 1)
        if pred_var is None: 
            fg_var = self.sigma * self.sigma
        elif pred_var.shape[1] == 4:
            fg_var = pred_var.view(x_hat.shape[0], 4, 1, 1).expand_as(x_hat)
        elif pred_var.shape[1] == 8:
            fg_var = pred_var[:, :4].view(x_hat.shape[0], 4, 1, 1).expand_as(x_hat)
            self.bg_var = pred_var[:, 4:].view(x_hat.shape[0], 4, 1, 1).expand_as(x_hat)
        else:
            raise NotImplementedError
        # else:
        #     fg_var = fg_var.view(x_hat.shape[0], 4, 1, 1).expand_as(x_hat)
        var = var + (mask==0).float()*self.bg_var + mask.float() * fg_var
        # torch.set_printoptions(threshold=torch.inf)
        # print(var)
        # plt.imshow(var[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.show()
        var = var.to(x.device)
        loss = torch.mean(torch.sum(self.GaussianNLL(x_hat, x, var).reshape(x.shape[0], -1), dim=1))
        return loss

class GaussianRegression(nn.Module):
    def __init__(self, input_dim, bg_var=1.0, reduction='none'):
        super(GaussianRegression, self).__init__()
        self.bg_var = bg_var
        self.fc = nn.Linear(input_dim*2, 4)
        self.reduction = reduction

    def forward(self, latent, x_hat, x, mask):
        fg_var = torch.exp(self.fc(latent))
        # print(fg_var[0, 0])
        # bg_var = out[:, :4]
        # fg_var = out[:, 4:]
        var = torch.zeros_like(x_hat)
        mask = mask.repeat(1, 4, 1, 1)
        fg_var = fg_var.view(-1, 4, 1, 1).expand_as(x_hat)
        # print(mask.shape, fg_var.shape, var.shape)
        var = var + (mask==0).float()*self.bg_var + mask.float() * fg_var
        var = var.to(x.device)
        # print(var[0, 0, :, :])
        loss = torch.mean(torch.sum(F.gaussian_nll_loss(x_hat, x, var, reduction=self.reduction).reshape(x.shape[0], -1), dim=1))
        return loss

class LossRegression(nn.Module):
    def __init__(self, bg_var=1.0, latent_dim=128):
        super(LossRegression, self).__init__()
        self.GaussianNLL = GaussianRegression(latent_dim, bg_var, reduction='none')

    def forward(self, x_hat, x, mu, log_var, mask):
        kl_loss = self.kl_divergence(mu, log_var)
        # print(torch.cat([mu, log_var], dim=1).shape, x_hat.shape, x.shape, mask.shape)
        recon_loss = self.GaussianNLL(torch.cat([mu, log_var], dim=1), x_hat, x, mask)
        loss = kl_loss + recon_loss
        return loss, kl_loss, recon_loss
    
    @staticmethod
    def kl_divergence(mu, log_var):
        """
        Compute the KL divergence between given distribution q(z|x) and standard normal distribution
        :param mu: mean vector produced by the encoder, tensor of shape (B, latent_dim)
        :param log_var: log sigma vector produced by the encoder, tensor of shape (B, latent_dim)
        :return: KL divergence between q(z|x) and p(z), where p(z)~N(0,I).
        """
        kl = 0.5 * torch.sum((torch.exp(log_var) + torch.square(mu) - 1 - log_var), -1)
        return torch.mean(kl)


if __name__ == "__main__":
    x = torch.randn(2, 4, 28, 28)
    mask = torch.zeros(2, 1, 28, 28)
    mask[:, :, 10:20, 10:20] = 1
    x_hat = torch.randn(2, 4, 28, 28)
    latent_dim = 128
    mu = torch.randn(2, latent_dim)
    log_var = torch.randn(2, latent_dim)
    critiria = LossRegression(bg_var=1.0, latent_dim=latent_dim)
    loss, kl_loss, recon_loss = critiria(x_hat, x, mu, log_var, mask)
    print(critiria)
    print(loss, kl_loss, recon_loss)
    print(loss.shape, kl_loss.shape, recon_loss.shape)