import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torch
import torchvision.models as models
import pdb
from torch import nn

# vgg19=models.vgg19()

# vgg19.features = nn.Sequential(*(vgg19.features[i] for i in range(30)))
vgg19 = models.vgg19_bn()
vgg19.load_state_dict(torch.load('vgg19_bn.pth'))


vgg19.features=nn.Sequential(vgg19.features)




input = Variable(torch.FloatTensor(20, 3, 256, 256))


output=feature(input)

print output.size()

