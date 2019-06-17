import torch
import torch.nn as nn

from net.utils import Reshaper, DoubleLayer


class CNNVAE(nn.Module):
    def __init__(self, weights=None):
        super(CNNVAE, self).__init__()

        self.encoder = nn.Sequential(
            Reshaper(3, 64, 64),
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Reshaper(4096),
            DoubleLayer(4096, 256)
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 4096),
            Reshaper(64, 8, 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        if weights:
            self.load_state_dict(torch.load(weights, map_location='cpu'))
            self.eval()

    @staticmethod
    def reparam(mean, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, var = self.encoder(x)
        x = self.reparam(mean, var)

        x = self.decoder(x)

        return x

