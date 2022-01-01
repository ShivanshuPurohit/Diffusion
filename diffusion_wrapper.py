import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import einsum

def from_pil_image(x):
    """Converts from a PIL image to a tensor."""
    x = TF.to_tensor(x)
    if x.ndim == 2:
        x = x[..., None]
    return x * 2 - 1

def beta_schedule(timesteps):
    #Returns a beta schedule for the diffusion model.
    return 1.0 - np.exp(-0.1 * timesteps)


class DiffusionModel(nn.Module):
    def __init__(self, net, input_shape, timesteps=1000):
        super().__init__()
        self.net = net
        self.input_shape = input_shape
        self.timesteps = timesteps
        self.register_buffer('beta_schedule', torch.from_numpy(beta_schedule(timesteps)))
        self.register_buffer('alpha_schedule', torch.from_numpy(1.0-beta_schedule(timesteps)))
        self.register_buffer('alpha_bar_schedule', torch.from_numpy(np.cumprod(1.0-beta_schedule(timesteps))))

    @torch.no_grad()
    def generate(self, n, *args, **kwargs):
        was_training = self.net.training
        self.net.eval()
        device = self.beta_schedule.device
        x = torch.randn((n,)+self.input_shape, device=device)
        for t in reversed(range(self.timesteps)):
            timestep = torch.full((n,), t, device=device)
            x = (self.alpha_schedule[t] ** -0.5) * (x - ((1.0 - self.alpha_schedule[t]) \
                * (1.0 - self.alpha_bar_schedule[t] ** -0.5) \
                * self.net(x, timestep, *args, **kwargs)))
            if t > 0:
                z = torch.randn((n,)+self.input_shape, device=device)
                x += (self.beta_schedule[t] ** -0.5) * z
        self.net.train(was_training)
        return x

    def forward(self, x, *args, **kwargs):
        device = self.beta_schedule.device
        x = x.to(device)
        noise = torch.randn(x.shape, device=device)
        timestep = torch.randint(0, self.timesteps, (x.shape[0],), device=device)
        alpha_bar = torch.gather(self.alpha_bar_schedule, 0, timestep)
        noised = einsum("b, b ... -> b ...", alpha_bar ** 0.5, x) + \
            einsum("b, b ... -> b ...", (1.0 - alpha_bar) ** 0.5, noise)
        predicted_noise = self.net(noised, timestep, *args, **kwargs)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
