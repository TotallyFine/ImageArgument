# coding:utf-8

import torch
import torchvision.transforms as tfs
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from torchvision.dataset import MNIST

import numpy as np

from model import generaotr, discriminator, generator_loss, discriminator_loss

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5
    
def deprocess_img(x):
    return (x + 1.0) / 2.0
    
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

def get_optimizer(net):
    optimizer = torch.optimizer.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.99))
    return optimizer

def train_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, 
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x) # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络
            
            # 生成网络
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络


      
def main():
    train_set = MNIST('./mnist', train=True, download=True,      transform=preprocess_img)
    train_data = DataLoader(train_set, batch_size=128, sampler=ChunkSampler(50000, 0))
    val_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)
    val_data = DataLoader(val_set, batch_size=128, sampler=ChunkSampler(5000, 50000))
    
    D_net = discriminator()
    G_net = generator()
    D_optim = get_optimizer(D_net)
    G_optim = get_optimizer(G_net)
    
    train_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, discriminator_loss, generator_loss, num_epochs=5)
    
if __name__ == '__main__':
    main()
    
