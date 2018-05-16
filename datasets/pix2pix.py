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
    # index = np.random.randint(self.__len__(), size=1)[0]
    # index = np.random.randint(self.__len__(), size=1)[0]+1
    # index = np.random.randint(self.__len__(), size=1)[0]

    # index_folder = np.random.randint(1,4)
    index_folder = np.random.randint(0,1)

    index_sub = np.random.randint(2, 5)

    label=index_folder


    if index_folder==0:
        path='/home/openset/Desktop/derain2018/facades/training2'+'/'+str(index)+'.jpg'



    if index_folder==1:
      if index_sub<4:
        path='/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Heavy/train2018new'+'/'+str(index)+'.jpg'
      if index_sub==4:
        index = np.random.randint(0,400)
        path='/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Heavy/trainnew'+'/'+str(index)+'.jpg'

    if index_folder==2:
      if index_sub<4:
        path='/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Medium/train2018new'+'/'+str(index)+'.jpg'
      if index_sub==4:
        index = np.random.randint(0,400)
        path='/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Medium/trainnew'+'/'+str(index)+'.jpg'

    if index_folder==3:
      if index_sub<4:
        path='/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Light/train2018new'+'/'+str(index)+'.jpg'
      if index_sub==4:
        index = np.random.randint(0,400)
        path='/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Light/trainnew'+'/'+str(index)+'.jpg'



    # img = self.loader(path)

    img = self.loader(path)

    # NOTE: img -> PIL Image
    # w, h = img.size
    # w, h = 1024, 512
    # img = img.resize((w, h), Image.BILINEAR)
    # pix = np.array(I)
    #
    # r = 16
    # eps = 1
    #
    # I = img.crop((0, 0, w/2, h))
    # pix = np.array(I)
    # base=guidedfilter(pix, pix, r, eps)
    # base = PIL.Image.fromarray(numpy.uint8(base))
    #
    #
    #
    # imgA=base
    # imgB=I-base
    # imgC = img.crop((w/2, 0, w, h))

    w, h = img.size
    # img = img.resize((w, h), Image.BILINEAR)


    # NOTE: split a sample into imgA and imgB
    imgA = img.crop((0, 0, w/2, h))
    # imgC = img.crop((2*w/3, 0, w, h))

    imgB = img.crop((w/2, 0, w, h))


    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
      imgA, imgB = self.transform(imgA, imgB)

    return imgA, imgB, label

  def __len__(self):
    # return 679
    print(len(self.imgs))
    return len(self.imgs)
