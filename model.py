import math

import torch
import torch.nn as nn

import torch.nn.functional as F

# Define VAE model
# calculate the output shape of the convolutional layer
def calc_activation_shape(dim, ksize=(5, 5), stride=(1, 1), padding=(0, 0), dilation=(1, 1), output_padding=(0, 0),
                          transposed=False):
    def shape_each_dim(i):
        if transposed:
            odim_i = (dim[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (ksize[i] - 1) + 1 + output_padding[i]
        else:
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            odim_i = odim_i / stride[i] + 1
        return math.floor(odim_i)

    return shape_each_dim(0), shape_each_dim(1)

# Define the encoder residual block
class EncoderBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ln_shape, use_batch_norm=True, stride=1, downsample=None):
        super(EncoderBottleneckBlock, self).__init__()
        ln_shape = calc_activation_shape(ln_shape, ksize=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm_1 = nn.BatchNorm2d(planes) if use_batch_norm else nn.LayerNorm([planes, *ln_shape])

        ln_shape = calc_activation_shape(ln_shape, ksize=(5, 5), stride=(stride, stride), padding=(2, 2))
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.norm_2 = nn.BatchNorm2d(planes) if use_batch_norm else nn.LayerNorm([planes, *ln_shape])

        ln_shape = calc_activation_shape(ln_shape, ksize=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.norm_3 = nn.BatchNorm2d(planes * self.expansion) if use_batch_norm else nn.LayerNorm([planes * self.expansion, *ln_shape])

        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)

        out = self.norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.norm_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.norm_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the encoder
class Encoder(nn.Module):

    def __init__(self, in_channels, latent_dim, use_batch_norm, dropout, layers):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.layers = layers

        self.ln_shape = (32, 32)

        self.conv_1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.inplanes = 8

        self.ln_shape = calc_activation_shape(self.ln_shape, ksize=(3, 3), stride=(2, 2), padding=(1, 1))
        self.norm_layer_1 = nn.BatchNorm2d(8) if use_batch_norm else nn.LayerNorm([8, *self.ln_shape])
        self.relu = nn.LeakyReLU(inplace=True)

        self.layer_1 = self._make_layer(8, layers[0], stride=2)
        self.layer_2 = self._make_layer(8, layers[1], stride=2)
        self.layer_3 = self._make_layer(16, layers[2], stride=2)
        self.layer_4 = self._make_layer(32, layers[3])

        self.conv_1x1 = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1),
                                      nn.BatchNorm2d(16) if use_batch_norm else nn.LayerNorm([16, *self.ln_shape]),
                                      nn.LeakyReLU(inplace=True))

        linear_dim = 16 * self.ln_shape[0] * self.ln_shape[1]
        self.fc_mu = nn.Linear(linear_dim, latent_dim)
        self.fc_var = nn.Linear(linear_dim, latent_dim)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        ln_shape = calc_activation_shape(self.ln_shape, ksize=(5, 5), stride=(stride, stride), padding=(2, 2))
        if stride != 1 or self.inplanes != planes * EncoderBottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * EncoderBottleneckBlock.expansion,
                          kernel_size=5, padding=2, stride=stride, bias=False),
                nn.BatchNorm2d(planes * EncoderBottleneckBlock.expansion) if self.use_batch_norm else nn.LayerNorm([planes * EncoderBottleneckBlock.expansion, *ln_shape]),
            )

        layers = [EncoderBottleneckBlock(self.inplanes, planes, self.ln_shape, self.use_batch_norm, stride, downsample)]

        self.inplanes = planes * EncoderBottleneckBlock.expansion

        for i in range(1, blocks):
            layers.append(EncoderBottleneckBlock(self.inplanes, planes, ln_shape, self.use_batch_norm))

        self.ln_shape = ln_shape

        return nn.Sequential(*layers)

    def encode(self, x):
        """
        Pass the input to the encoder and get the latent distribution
        :param x: input of shape (B, C, H, W)
        :return: vectors mu and log_var produced by the encoder
        """
        # Compute encoder output
        x = self.conv_1(x)
        x = self.norm_layer_1(x)
        x = self.relu(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.conv_1x1(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def forward(self, x):
        """
        Get the latent encoding of the data_loaders and sample z from a learned distribution
        :param x: input of shape (B, C, H, W)
        :return: sample from the distribution q_zx,
                 a list containing mu and sigma vectors
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return [z, mu, log_var]

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        :param mu: vector of means produced by the encoder
        :param log_var: vector of log variances produced by the encoder
        :return: sample from the distribution parametrized by mu and var
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

# Define the decoder residual block
class DecoderBottleneckBlock(nn.Module):
    expansion = 4  # expansion factor

    def __init__(self, in_channels, planes, ln_shape, use_batch_norm=False, upsample=None, stride=2, output_padding=0):
        super(DecoderBottleneckBlock, self).__init__()

        self.upsample = upsample
        self.stride = stride
        self.use_batch_norm = use_batch_norm

        ln_shape = calc_activation_shape(ln_shape, ksize=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), output_padding=(0, 0), transposed=True)
        self.conv_1 = nn.ConvTranspose2d(in_channels, planes, kernel_size=1, stride=1, padding=0)
        self.norm_1 = nn.BatchNorm2d(planes) if self.use_batch_norm else nn.LayerNorm([planes, *ln_shape])

        ln_shape = calc_activation_shape(ln_shape, ksize=(5, 5), stride=(stride, stride), padding=(2, 2), dilation=(1, 1), output_padding=(output_padding, output_padding), transposed=True)
        self.conv_2 = nn.ConvTranspose2d(planes, planes, kernel_size=5, stride=self.stride, padding=2, output_padding=output_padding)
        self.norm_2 = nn.BatchNorm2d(planes) if self.use_batch_norm else nn.LayerNorm([planes, *ln_shape])

        ln_shape = calc_activation_shape(ln_shape, ksize=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), output_padding=(0, 0), transposed=True)
        self.conv_3 = nn.ConvTranspose2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.norm_3 = nn.BatchNorm2d(planes * self.expansion) if self.use_batch_norm else nn.LayerNorm([planes * self.expansion, *ln_shape])

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.norm_1(self.conv_1(x)))
        x = self.relu(self.norm_2(self.conv_2(x)))
        x = self.relu(self.norm_3(self.conv_3(x)))

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity
        x = self.relu(x)

        return x

# Define the decoder
class Decoder(nn.Module):

    def __init__(self, latent_dim, use_batch_norm, dropout, layers=[1, 1, 1, 1]):
        """
        :param latent_dim: size of the latent space
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.layers = layers
        # self.ln_shape = (8, 8)
        self.ln_shape = (2, 2)

        self.in_channels = 16
        linear_dim = self.in_channels * self.ln_shape[0] * self.ln_shape[1]
        self.dense_1 = nn.Linear(latent_dim, linear_dim)

        # Build the residual decoder
        self.layer_1 = self._make_layer(layers[3], planes=32)
        self.layer_2 = self._make_layer(layers[2], planes=32, stride=2, output_padding=1)
        self.layer_3 = self._make_layer(layers[1], planes=16, stride=2, output_padding=1)
        self.layer_4 = self._make_layer(layers[0], planes=8, stride=2, output_padding=1)
        self.layer_5 = self._make_layer(1, planes=8, stride=2, output_padding=1)

        self.upconv_1 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, stack, planes, stride=1, output_padding=0):
        sub_layers = []
        upsample = None

        # Initialize upsampling
        ln_shape = calc_activation_shape(self.ln_shape, ksize=(1, 1), stride=(stride, stride), padding=(0, 0), dilation=(1, 1), output_padding=(output_padding, output_padding), transposed=True)
        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, planes * DecoderBottleneckBlock.expansion, kernel_size=1,
                               stride=stride, output_padding=output_padding),
            nn.BatchNorm2d(planes * DecoderBottleneckBlock.expansion) if self.use_batch_norm else nn.LayerNorm([planes * DecoderBottleneckBlock.expansion, *ln_shape])
        )

        # First stack layer
        sub_layers.append(DecoderBottleneckBlock(self.in_channels, planes, self.ln_shape, use_batch_norm=self.use_batch_norm, upsample=upsample, stride=stride,
                                                 output_padding=output_padding))
        self.in_channels = planes * DecoderBottleneckBlock.expansion

        # Other stack layers
        for i in range(stack - 1):
            sub_layers.append(DecoderBottleneckBlock(self.in_channels, planes, ln_shape, use_batch_norm=self.use_batch_norm, upsample=None, stride=1))

        self.ln_shape = ln_shape

        return nn.Sequential(*sub_layers)

    def forward(self, z):
        """
        Reconstruct the image from the latent code
        :param z: sample from the latent distribution
        :return: reconstruction of the sample z
        """
        x = self.dense_1(z)
        x = x.reshape(-1, 16, 2, 2)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x_hat = self.upconv_1(x)

        # return torch.sigmoid(x_hat)
        return x_hat

# Define the VAE: encoder + decoder
class ResVAE(nn.Module):

    def __init__(self, in_channels, latent_dim, use_batch_norm=False, dropout=0.0, layer_list=None, pred_var=None):
        super(ResVAE, self).__init__()

        if layer_list is None:
            layer_list = [3, 4, 6, 3]

        self.pred_var = pred_var

        self.encoder = Encoder(in_channels, latent_dim, use_batch_norm, dropout, layer_list)
        self.decoder = Decoder(latent_dim, use_batch_norm, dropout, layer_list)
        # self.var_fc = nn.Linear(latent_dim, pred_var)
        if pred_var is not None:
            self.var_fc = nn.Linear(latent_dim, pred_var)
            # self.var_fc = nn.Sequential(
            #     nn.Linear(latent_dim, 64),
            #     nn.LeakyReLU(),
            #     nn.Linear(64, pred_var)
            # )
        # self.fg_var_fc = nn.Sequential(
        #     nn.Linear(latent_dim, 64),
        #     nn.softplus()
        # )
        self.latent_dim = latent_dim

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        if self.pred_var:
            var = F.softplus(self.var_fc(z))
            # var = torch.exp(self.var_fc(z))
        else:
            var = None
        return x_hat, mu, log_var, var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return z
    

if __name__ == '__main__':
    model = ResVAE(4, 128)
    print(model)
    x = torch.randn(2, 4, 32, 32)
    x_hat, mu, log_var, var = model(x)
    print(x_hat.shape, mu.shape, log_var.shape, var.shape)
    print(var.shape[1])