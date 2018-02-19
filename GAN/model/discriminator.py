# coding:utf-8

from torch import nn

class discriminator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),# output (batch_size, 32, 24, 24)
            # use LeakyReLU
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2), # output (batch_size, 32, 12, 12)
            nn.Conv2d(32, 64, 5, 1),# output(batch_size, 64, 8, 8)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)# output(batch_size, 64, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)# x (batch_size, 1024)
        x = self.fc(x)
        return x
