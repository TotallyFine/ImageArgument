# coding:utf-8

import torch
from torch.nn import MSELoss
# define loss function using KL divergence, and MSE loss
def loss_function(self, mu, logvar, output, target):
    # KL divergence
    # -0.5(1+log(sigma**2)-mu**2-exp(log(sigma**2)))
    KLD = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(logvar).add_(1)
    KLD = KLD.mul_(-0.5)
    # try to reconstruction the data, using mse as loss
    reconstrucion_loss = MSELoss(size_average=True)
    BCE = reconstruction_loss(output, target)
    
    return KLD + BCE
