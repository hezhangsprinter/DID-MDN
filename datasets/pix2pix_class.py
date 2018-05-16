import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np


IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class pix2pix(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    imgs = make_dataset(root)
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    index_sub = np.random.randint(0, 3)
    label=index_sub



    if index_sub==0:
        index = np.random.randint(0,4000)
        path='/media/openset/Z/derain2018/facades/DID-MDN-training/Rain_Heavy/train2018new'+'/'+str(index)+'.jpg'


    if index_sub==1:
        index = np.random.randint(0,4000)
        path='/media/openset/Z/derain2018/facades/DID-MDN-training/Rain_Medium/train2018new'+'/'+str(index)+'.jpg'

    if index_sub==2:
        index = np.random.randint(0,4000)
        path='/media/openset/Z/derain2018/facades/DID-MDN-training/Rain_Light/train2018new'+'/'+str(index)+'.jpg'



    img = self.loader(path)


    w, h = img.size


    # NOTE: split a sample into imgA and imgB
    imgA = img.crop((0, 0, w/2, h))
    imgB = img.crop((w/2, 0, w, h))


    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    return imgA, imgB, label

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.imgs)
