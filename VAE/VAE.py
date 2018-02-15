# coding:utf-8

from torch import nn
import torch.nn.Functional as F

# define vae network
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # encoder and decoder can be MLP or CNN, even RNN
        # 28*28=784
        self.fc1 = nn.Linear(784, 400)
        # fc21 fc22 are parallel, one product sigma=log(std) other product mu
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def decode(self, z):
        # z is the intermediary code, constructed by sigma and mu
        h3 = F.relu(self.fc3(z))
        # sigmod let output in [-1, 1]
        return F.sigmod(self.fc4(h3))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
