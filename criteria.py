import torch
import torch.nn as nn

# Define the loss function: KL divergence + reconstruction loss
class LossVAE(nn.Module):

    def __init__(self, sigma=0.1, bg_var=0.5):
        super(LossVAE, self).__init__()
        self.sigma = sigma
        self.bg_var = bg_var

    def forward(self, x_hat, x, mu, log_var, mask):
        kl_loss = self.kl_divergence(mu, log_var)
        recon_loss = self.reconstruction_loss(x_hat, x, mask)
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

    def reconstruction_loss(self, x_hat, x, mask):
        """
        Compute the reconstruction loss
        :param x: 2D
        :param x_hat: output of the decoder, considered as the mean of a distribution
        :return: reconstruction
        """
        var = torch.zeros_like(x_hat)
        mask = mask.repeat(1, 4, 1, 1)
        var = var + (mask==0).float()*self.bg_var + mask.float() * self.sigma *self.sigma
        # torch.set_printoptions(threshold=torch.inf)
        # print(var)
        # plt.imshow(var[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.show()
        var = var.to(x.device)
        criterion = nn.GaussianNLLLoss(reduction='none').to(x.device)
        loss = torch.mean(torch.sum(criterion(x, x_hat, var).reshape(x.shape[0], -1), dim=1))
        return loss
