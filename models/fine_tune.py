from __future__ import print_function
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
import models.UNet as net
from misc import *
import models.dehaze  as net
from myutils.vgg16 import Vgg16
from myutils import utils
import pdb

import torchvision.models as models

squeezenet = models.squeezenet1_0(pretrained=True)
