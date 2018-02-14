# coding:utf-8

from torch import nn

from .BasicModule import BasicModule

class DCautoencoder(BasicModule):
    def __init__(self):
        super(DCautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), #b, 16, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1) #b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # b, 16, 5, 5
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),# b, 8, 15, 15
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), # b, 1, 28, 28
            # Tanh can restrict every elem in [-1, 1], beacuse input is normalized to [-1, 1]
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
