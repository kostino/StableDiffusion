import torch
from torch import nn
import torch.nn.functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # BxCxHxW -> Bx128xHxW
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # Bx128xHxW -> Bx128xHxW
            VAE_ResidualBlock(128, 128),
            # Bx128xHxW -> Bx128xHxW
            VAE_ResidualBlock(128, 128),
            # Bx128xHxW -> Bx128xH//2xW//2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # Bx128xHxW -> Bx256xHxW
            VAE_ResidualBlock(128, 256),
            # Bx256xHxW -> Bx256xHxW
            VAE_ResidualBlock(256, 256),
            # Bx128xHxW -> Bx128xH//4xW//4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),
            # Bx256xHxW -> Bx256xHxW
            VAE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),
            # Bx256xHxW -> Bx256xHxW
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):

                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        std = variance.sqrt()

        # Z=N(0,1) -> X=N(mu, sigma)
        # X = mean + std * Z

        x = mean + std * noise
        x *= 0.18215

        return x
    