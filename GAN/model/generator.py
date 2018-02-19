# coding:utf-8

from torch import nn

class generator(nn.Module):
    def __init__(self, noise_dim=96):
        super(generator, self).__init__()
        # in discriminator, conv first then fc
        # in generator fc first
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(),
            nn.BatchNorm1d(7*7*128)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.conv(x)
        return x
