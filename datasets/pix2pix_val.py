import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
# from guidedfilter import guidedfilter
# import guidedfilter.guidedfilter as guidedfilter





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

class pix2pix_val(data.Dataset):
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
    # index = np.random.randint(self.__len__(), size=1)[0]
    # index = np.random.randint(self.__len__(), size=1)[0]

    path = self.imgs[index]

    path=self.root+'/'+str(index)+'.jpg'

    index_folder = np.random.randint(0,4)
    label=index_folder

    # path='/home/openset/Desktop/derain2018/facades/DB_Rain_test/Rain_Heavy/test2018'+'/'+str(index)+'.jpg'
    img = self.loader(path)

    # # NOTE: img -> PIL Image
    # w, h = img.size
    # w, h = 1024, 512
    # img = img.resize((w, h), Image.BILINEAR)
    # # NOTE: split a sample into imgA and imgB
    # imgA = img.crop((0, 0, w/2, h))
    # imgB = img.crop((w/2, 0, w, h))
    # if self.transform is not None:
    #   # NOTE preprocessing for each pair of images
    #   imgA, imgB = self.transform(imgA, imgB)
    # return imgA, imgB


    # w, h = 1536, 512
    # img = img.resize((w, h), Image.BILINEAR)
    #
    #
    # # NOTE: split a sample into imgA and imgB
    # imgA = img.crop((0, 0, w/3, h))
    # imgC = img.crop((2*w/3, 0, w, h))
    #
    # imgB = img.crop((w/3, 0, 2*w/3, h))

    # w, h = 1024, 512
    # img = img.resize((w, h), Image.BILINEAR)
    #
    # r = 16
    # eps = 1
    #
    # # I = img.crop((0, 0, w/2, h))
    # # pix = np.array(I)
    # # print
    # # base[idx,:,:,:]=guidedfilter(pix[], pix[], r, eps)
    # # base[]=guidedfilter(pix[], pix[], r, eps)
    # # base[]=guidedfilter(pix[], pix[], r, eps)
    #
    #
    # # base = PIL.Image.fromarray(numpy.uint8(base))
    #
    # # NOTE: split a sample into imgA and imgB
    # imgA = img.crop((0, 0, w/3, h))
    # imgC = img.crop((2*w/3, 0, w, h))
    #
    # imgB = img.crop((w/3, 0, 2*w/3, h))
    # imgA=base
    # imgB=I-base
    # imgC = img.crop((w/2, 0, w, h))
    w, h = img.size
    # w, h = 586*2, 586

    # img = img.resize((w, h), Image.BILINEAR)


    # NOTE: split a sample into imgA and imgB
    imgA = img.crop((0, 0, w/2, h))
    # imgC = img.crop((2*w/3, 0, w, h))

    imgB = img.crop((w/2, 0, w, h))


    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
      imgA, imgB = self.transform(imgA, imgB)

    return imgA, imgB, path

  def __len__(self):
    return len(self.imgs)
