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
# import models.UNet as net
from misc import *
import models.derain_residual  as net2
import models.derain_dense  as net1

from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import torch.nn.functional as F

import torchvision.models as models


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix_class',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=120, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=586, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')


ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize





netG=net1.vgg19ca()
netG.load_state_dict(torch.load('./classification/netG_epoch_9.pth'))
print(netG)



netG.train()
criterion_class = nn.CrossEntropyLoss().cuda()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)



label_d = torch.FloatTensor(opt.batchSize)


target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)





# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netG.cuda()


target, input = target.cuda(), input.cuda()

target = Variable(target)
input = Variable(input)


residue_net=net2.Dense_rain_residual()
residue_net.load_state_dict(torch.load('./residual_heavy/netG_epoch_6.pth'))
residue_net=residue_net.cuda()

label_d = Variable(label_d.cuda())



# get optimizer
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)


# NOTE training loop
ganIterations = 0
for epoch in range(opt.niter):
  if epoch > opt.annealStart:
    adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)

  for i, data in enumerate(dataloader, 0):

    ### Get the rainy image and coreesponding ground truth label (0: Heavy, 1:Medium, 2: Light)##
    input_cpu, target_cpu, label_cpu = data


    target_label=label_cpu
    target_label=target_label.long().cuda()
    target_label=Variable(target_label)

    batch_size = target_cpu.size(0)

    target_cpu, input_cpu = target_cpu.float().cuda(), input_cpu.float().cuda()


    ### Using the Heavy rain Label (0) to estimate the residual first ##
    z = 0
    label_cpu = torch.FloatTensor(opt.batchSize).fill_(z)
    label_cpu=label_cpu.long().cuda()
    label_cpu=Variable(label_cpu)


    # get paired data
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)



    netG.zero_grad() # start to update G
    residue_net.zero_grad() # start to update G

    ### Using the Heavy rain Label (0) to get the residual ##
    output=residue_net(input, label_cpu)
    residue=input-output

    ### Using the estimated resiudal to predict the label ##

    label = netG(residue)


    #label_final=label.max(1)[1]
    #zz1=label_final.data.cpu().numpy()
    #zz2=target_label.data.cpu().numpy()

    #print(zz1)
    #print(zz2)

    netG.zero_grad() # start to update G

    class_loss = criterion_class(label, target_label)

    class_loss.backward()
    L_img = class_loss

    optimizerG.step()
    ganIterations += 1

    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f D(G(z)): %f / %f'
          % (epoch, opt.niter, i, len(dataloader),
             class_loss.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0]))

  if epoch % 1 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
