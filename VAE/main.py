# coding:utf-8

from torch.autograd import Variable
from torch.tuils.data import DataLoader
from torch.optim import Adam

from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from torchvision.datasets import save_image

from .config import opt
from .VAE import VAE
from .Loss import loss_function

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def main():
    net = VAE()
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        # normalize img, network's output activated by sigmoid,so it spread in [-1, 1]
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    dataset = MNIST(opt.root, transform=img_tfs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = Adam(net.parameters(), lr=1e-3)
    
    for epoch in range(40):
        for img, _ dataloader:
            img = img.view(img.size(0), -1)
            img = Variable(img)
            
            output, mu, logvar = net(img)
            loss = loss_function(mu, logvar, output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                save = to_img(output)
                save_image(save, opt.img_save_path+'{}_img.png'.format(epoch))
