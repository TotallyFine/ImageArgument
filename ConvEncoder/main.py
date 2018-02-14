# coding:utf-8
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

from Model import DCautoencoder
from config import opt

def main():
    net = DCautoencoder()
    
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_set = MNIST(opt.root, transform=im_tfs)
    print('build train_set success')
    #print(train_set)
    #print(train_set[0])
    train_data = DataLoader(train_set, batch_size=32, shuffle=True)
    print('build dataloader success')
    #print(train_data[0])
    # mean squared error between input and target
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print('begin train')
    # train the autoencoder
    for epoch in range(40):
        for im, _ in train_data:
            
            im = Variable(im)
            output = net(im)
            loss = criterion(output, im) / im.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print('epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.data[0]))
            pic = to_img(output.data)
            
            save_image(pic, opt.img_save_path+'/image_{}.png'.format(epoch+1))
    
    net.save(opt.model_save_path)
    
def to_img(x):
    '''
    change output to image
    '''
    x = 0.5 * (x + 1.)
    # restrict output to [0, 1]
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

if __name__ == '__main__':
    main()
