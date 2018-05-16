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

from misc import *
import models.derain_dense as net



from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import torch.nn.functional as F



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
  default=512, help='the height / width of the original input image')
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


opt.dataset='pix2pix_val'
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y


ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# get models
# netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG=net.Dense_rain()

if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)




netG.train()
criterionCAE = nn.L1Loss()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)




val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.FloatTensor(opt.batchSize)


target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
depth = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
ato = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)


val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_depth = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_ato = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)




# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netG.cuda()
criterionCAE.cuda()




target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

target = Variable(target)
input = Variable(input)
# input = Variable(input,requires_grad=False)
# depth = Variable(depth)
ato = Variable(ato)

# Initialize VGG-16
vgg = Vgg16()
utils.init_vgg16('./models/')
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.cuda()


label_d = Variable(label_d.cuda())

# get randomly sampled validation images and save it
val_iter = iter(valDataloader)
data_val = val_iter.next()

val_input_cpu, val_target_cpu, label = data_val

val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()


val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
vutils.save_image(val_target, '%s/real_target.png' % opt.exp, normalize=True)
vutils.save_image(val_input, '%s/real_input.png' % opt.exp, normalize=True)


# pdb.set_trace()
# get optimizer
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
# NOTE training loop
ganIterations = 0
for epoch in range(opt.niter):
  if epoch > opt.annealStart:
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)


  for i, data in enumerate(dataloader, 0):

    input_cpu, target_cpu, label_cpu = data
    batch_size = target_cpu.size(0)

    target_cpu, input_cpu = target_cpu.float().cuda(), input_cpu.float().cuda()
    label_cpu=label_cpu.long().cuda()
    label_cpu=Variable(label_cpu)


    # get paired data
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)



    x_hat1 = netG(input, label_cpu)

    residual, x_hat = x_hat1

    netG.zero_grad() # start to update G


    L_img_ = criterionCAE(x_hat, target)

    # L_res = lambdaIMG * L_res_
    L_img = lambdaIMG * L_img_

    if lambdaIMG <> 0:
      #L_img.backward(retain_graph=True) # in case of current version of pytorch
      L_img.backward(retain_variables=True)
      # L_res.backward(retain_variables=True)

    # Perceptual Loss 1
    features_content = vgg(target)
    f_xc_c = Variable(features_content[1].data, requires_grad=False)

    features_y = vgg(x_hat)
    content_loss =  1.8*lambdaIMG* criterionCAE(features_y[1], f_xc_c)
    content_loss.backward(retain_variables=True)

    # Perceptual Loss 2
    features_content = vgg(target)
    f_xc_c = Variable(features_content[0].data, requires_grad=False)

    features_y = vgg(x_hat)
    content_loss1 =  1.8*lambdaIMG* criterionCAE(features_y[0], f_xc_c)
    content_loss1.backward(retain_variables=True)


    optimizerG.step()
    ganIterations += 1

    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f D(G(z)): %f / %f'
          % (epoch, opt.niter, i, len(dataloader),
             L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0]))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0]))
      trainLogger.flush()
    if ganIterations % opt.evalIter == 0:
      val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
      for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)
        ###  We use a random label here just for intermediate result visuliztion (No need to worry about the label here) ##

        label_result=float(label_cpu.data.cpu().float().numpy())
        label_result=float(label_result)
        label=label_cpu
        residual_val, x_hat_val = netG(val_inputv, label)



        val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
      vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
        (opt.exp, label_result, ganIterations), normalize=True, scale_each=False)

  if epoch % 1 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
