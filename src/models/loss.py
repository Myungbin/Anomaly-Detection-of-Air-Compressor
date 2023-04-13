import torch
import torch.nn as nn


def huber_loss(y_pred, y_true, delta=1.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(loss)


class MSE_SN_Loss(nn.Module):
    def __init__(self, noise_factor=0.5):
        super(MSE_SN_Loss, self).__init__()
        self.noise_factor = noise_factor
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, outputs):
        noise = self.noise_factor * torch.randn_like(inputs)
        inputs_noisy = inputs + noise
        return self.mse_loss(outputs, inputs_noisy)


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss
