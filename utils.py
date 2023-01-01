import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def make_weights_for_balanced_classes(images, nclasses): 
    count = [0] * nclasses 
    for item in images: 
        count[item[1]] += 1 
    weight_per_class = [0.] * nclasses 
    N = float(sum(count)) 
    for i in range(nclasses): 
        weight_per_class[i] = N/float(count[i]) 
    weight = [0] * len(images) 
    for idx, val in enumerate(images): 
        weight[idx] = weight_per_class[val[1]] 
    return weight 


def gain_sample_w(dataset, batch_size,weights):
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size,sampler=sampler, num_workers=8, pin_memory=True)
    return loader