# coding:utf-8
import torch
from torch import nn
from torch.autograd import Variable

def discriminator_loss(logits_real, logits_fake):
    size = logits_real.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float()
    false_labels = Variable(torch.zeros(size, 1)).float()
    loss = nn.BCEWithLogitsLoss(logits_real, true_labels) + bce_loss(logits_real, false_labels)
    return loss
    
def generator_loss(logits_fake):
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float()
    loss = nn.BCEWithLogitsLoss(logits_fake, true_labels)
    return loss
