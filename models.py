import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# ResNet Classifier
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.embDim = 128 * block.expansion
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out
    def get_embedding_dim(self):
        return self.embDim


#######################################################################
#Conditional GAN Generator
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        
        def block(in_feat, out_feat, kernel, stride, padding, bias=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride, padding, bias=False)]
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, opt.ngf * 4, 4, 1, 0, bias=False),
            *block(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            *block(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(opt.ngf,opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        labels = self.label_emb(labels)
        labels = torch.reshape(labels, (labels.size(0), labels.size(1), 1 , 1))
        gen_input = torch.cat((labels, noise), 1)
        img = self.model(gen_input)

        return img


#######################################################################
#Conditional GAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.linear_expand = nn.Linear(opt.n_classes, int(opt.img_size**2))
    
        def block(in_feat, out_feat, kernel, stride, padding, bias=False, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels+1, opt.ndf, 4, 2, 1, bias=False, normalize=False),#(64,7,128,128)
            *block(opt.ndf , opt.ndf * 2, 4, 2, 1, bias=False),
            *block(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.Conv2d(opt.ndf* 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels, opt):
        # Concatenate label embedding and image to produce input
        labels = self.label_emb(labels)
        labels = self.linear_expand(labels)
        labels = torch.reshape(labels, (labels.size(0), 1, opt.img_size, opt.img_size))
        d_in = torch.cat((img, labels), 1)
        validity = self.model(d_in)
        
        return validity