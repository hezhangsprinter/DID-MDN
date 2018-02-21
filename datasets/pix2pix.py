import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import lingua as sla
# import guidedfilter.guidedfilter as guidedfilter
from guidedfilter import guidedfilter


def tikhonov_filter(s, lmbda, npd=16):
    r"""Lowpass filter based on Tikhonov regularization.
    Lowpass filter image(s) and return low and high frequency
    components, consisting of the lowpass filtered image and its
    difference with the input image. The lowpass filter is equivalent to
    Tikhonov regularization with `lmbda` as the regularization parameter
    and a discrete gradient as the operator in the regularization term,
    i.e. the lowpass component is the solution to
    .. math::
      \mathrm{argmin}_\mathbf{x} \; (1/2) \left\|\mathbf{x} - \mathbf{s}
      \right\|_2^2 + (\lambda / 2) \sum_i \| G_i \mathbf{x} \|_2^2 \;\;,
    where :math:`\mathbf{s}` is the input image, :math:`\lambda` is the
    regularization parameter, and :math:`G_i` is an operator that
    computes the discrete gradient along image axis :math:`i`. Once the
    lowpass component :math:`\mathbf{x}` has been computed, the highpass
    component is just :math:`\mathbf{s} - \mathbf{x}`.
    Parameters
    ----------
    s : array_like
      Input image or array of images.
    lmbda : float
      Regularization parameter controlling lowpass filtering.
    npd : int, optional (default=16)
      Number of samples to pad at image boundaries.
    Returns
    -------
    sl : array_like
      Lowpass image or array of images.
    sh : array_like
      Highpass image or array of images.
    """

    grv = np.array([-1.0, 1.0]).reshape([2, 1])
    gcv = np.array([-1.0, 1.0]).reshape([1, 2])
    Gr = sla.fftn(grv, (s.shape[0]+2*npd, s.shape[1]+2*npd), (0, 1))
    Gc = sla.fftn(gcv, (s.shape[0]+2*npd, s.shape[1]+2*npd), (0, 1))
    A = 1.0 + lmbda*np.conj(Gr)*Gr + lmbda*np.conj(Gc)*Gc
    if s.ndim > 2:
        A = A[(slice(None),)*2 + (np.newaxis,)*(s.ndim-2)]
    sp = np.pad(s, ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2), 'symmetric')
    slp = np.real(sla.ifftn(sla.fftn(sp, axes=(0, 1)) / A, axes=(0, 1)))
    sl = slp[npd:(slp.shape[0]-npd), npd:(slp.shape[1]-npd)]
    sh = s - sl
    return sl.astype(s.dtype), sh.astype(s.dtype)


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
