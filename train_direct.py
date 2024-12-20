"""
This can be used to run training as well.
This file is used to run training, similar to train.py. 
Only difference is, all the code for model building steps are defined rather than imported.  
"""
from __future__ import print_function

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

import math

import datetime
import uuid
import tensorboard_logger


from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


def geometric_transforms(severity=1):
    # Severity controls the intensity of transformations
    return transforms.Compose([
        # transforms.RandomRotation(degrees=30 * severity),
        transforms.RandomAffine(
            degrees=(-30 * severity, 30 * severity),
            translate=(0.1 * severity, 0.1 * severity),
            scale=(1 - 0.1 * severity, 1 + 0.1 * severity),
            shear=10 * severity
        )
    ])


class NCIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with noisy test set.
    --- Modified to generate noise-augmented test sets

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        noise_test (array, optional): If present, it indicates that the test set has to be
            augmented with noise. Options are:
            - type: gaussian, speckle, poisson
            - val: the std for the gaussian and speckle noise (not used for poisson)

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 noise_test=None, clip_noise=False,
                 normalize_transform=None, apply_geometric_transform=False, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_transform = normalize_transform
        self.apply_geometric_transform = apply_geometric_transform


        self.train = train  # training set or test set
        self.noise_test = noise_test
        # if self.noise_test is not None and self.noise_test['type'] == 'poisson':
        #     self.noise_test['val'] = 0
        self.noise_test_data = None
        self.clip_noise = clip_noise

        if not self._check_integrity():
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]

            # load test data
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))

            # Load the noise data (to be used on to the original test data - depending on the kind of noise)
            # the test data are noised by means of a custom Transform class
            if self.noise_test is not None:
                filetest = f + '_gauss_' + str(self.noise_test['val'])
                file = os.path.join(self.root, self.base_folder, filetest)

                if not os.path.exists(file):  # generate the noise samples
                    if self.noise_test['type'] == 'gaussian' or self.noise_test['type'] == 'speckle':
                        self.noise_test_data = np.random.normal(0, self.noise_test['val'] ** 0.5, self.test_data.shape)
                        torch.save(self.noise_test_data, file)
                else:
                    self.noise_test_data = torch.load(file)

            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # Simulate Rotated , Sheared image
        if self.apply_geometric_transform is not None:
            geom_trnsfrm = geometric_transforms(severity=1)
            img = geom_trnsfrm(img)

        if self.transform is not None:
            img = self.transform(img)

        if not self.train and self.noise_test is not None:  # generate noisy data
            noisemap = None
            if self.noise_test_data is not None:
                noisemap = self.noise_test_data[index]

            '''
            t = NoiseTransform(mode=self.noise_test['type'],
                               value=self.noise_test['val'],
                               noisemap=noisemap)
            img = t(img)
            '''

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.normalize_transform is not None:
            img = self.normalize_transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class NCIFAR100(NCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


##############################################################
## Parser Setup : it's used to parse command line arguments ##
############################################################## 
parser = argparse.ArgumentParser(description='PyTorch ResNet and PP-ResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default], cifar100, svhn)')
parser.add_argument('--arch', default='resnet', type=str, help='architecture (resnet, densenet, [... more to come ...])')

parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
parser.add_argument('--milestones', default='[80, 120]', type=str, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--extra-epochs', default=0, type=int, help='number of extra epochs to run')
parser.add_argument('--extra-milestones', default='[160]', type=str, help='extra epoch milestones for the scheduler')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 5e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 20)')
parser.add_argument('--expansion', default=1, type=int, help='total expansion of Kernels (default: 1)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)') # for densenet
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-efficient', dest='efficient', action='store_false',
                    help='To not use bottleneck block')

parser.add_argument('--pushpull', action='store_true', help='use Push-Pull layer as 1st layer (default: False)')
parser.add_argument('--use-cuda', action='store_true', help='Use Cuda (default: False)')
parser.add_argument('--pp-block1', action='store_true', help='use 1st PushPull residual block')
parser.add_argument('--pp-all', action='store_true', help='use all PushPull residual block')

parser.add_argument('--train-alpha', action='store_true', help='whether to learn the values of alpha ')
parser.add_argument('--alpha-pp', default=1, type=float, help='inhibition factor (default: 1.0)')
parser.add_argument('--scale-pp', default=2, type=float, help='upsampling factor for PP kernels (default: 2)')

parser.add_argument('--lpf-size', default=None, type=int, help='Size of the LPF for anti-aliasing (default: 1)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='resnet20', type=str, help='name of experiment')

parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)
args = parser.parse_args()


best_prec1 = 0
use_cuda = torch.cuda.is_available() & args.use_cuda
## This is where result is saved
experiment_dir = 'experiments/'


"""
Model Setup : 
ResNet-20 with Push-Pull: implemented on top of the official PyTorch ResNet implementation
"""

class PPmodule2d(nn.Module):
    """
    Implementation of the Push-Pull layer from:
    [1] N. Strisciuglio, M. Lopez-Antequera, N. Petkov,
    Enhanced robustness of convolutional networks with a push–pull inhibition layer,
    Neural Computing and Applications, 2020, doi: 10.1007/s00521-020-04751-8

    It is an extension of the Conv2d module, with extra arguments:

    * :attr:`alpha` controls the weight of the inhibition. (default: 1 - same strength as the push kernel)
    * :attr:`scale` controls the size of the pull (inhibition) kernel (default: 2 - double size).
    * :attr:`dual_output` determines if the response maps are separated for push and pull components.
    * :attr:`train_alpha` controls if the inhibition strength :attr:`alpha` is trained (default: False).


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        alpha (float, optional): Strength of the inhibitory (pull) response. Default: 1
        scale (float, optional): size factor of the pull (inhibition) kernel with respect to the pull kernel. Default: 2
        dual_output (bool, optional): If ``True``, push and pull response maps are places into separate channels of the output. Default: ``False``
        train_alpha (bool, optional): If ``True``, set alpha (inhibition strength) as a learnable parameters. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 alpha=1, scale=2, dual_output=False,
                 train_alpha=False):
        super(PPmodule2d, self).__init__()

        self.dual_output = dual_output
        self.train_alpha = train_alpha

        # Note: the dual output is not tested yet
        if self.dual_output:
            assert (out_channels % 2 == 0)
            out_channels = out_channels // 2

        # Push kernels (is the one for which the weights are learned - the pull kernel is derived from it)
        self.push = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        """
        # Bias: push and pull convolutions will have bias=0.
        # If the PP kernel has bias, it is computed next to the combination of the 2 convolutions
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            # Inizialize bias
            n = in_channels
            for k in self.push.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        """

        # Configuration of the Push-Pull inhibition
        if not self.train_alpha:
            # when alpha is an hyper-parameter (as in [1])
            self.alpha = alpha
        else:
            # when alpha is a trainable parameter
            k = 1
            self.alpha = nn.Parameter(k * torch.ones(1, out_channels, 1, 1), requires_grad=True)
            r = 1. / math.sqrt(in_channels * out_channels)
            self.alpha.data.uniform_(.5-r, .5+r)  # math.sqrt(n) / 2)  # (-stdv, stdv)

        self.scale_factor = scale
        push_size = self.push.weight[0].size()[1]

        # compute the size of the pull kernel
        if self.scale_factor == 1:
            pull_size = push_size
        else:
            pull_size = math.floor(push_size * self.scale_factor)
            if pull_size % 2 == 0:
                pull_size += 1

        # upsample the pull kernel from the push kernel
        self.pull_padding = pull_size // 2 - push_size // 2 + padding
        self.up_sampler = nn.Upsample(size=(pull_size, pull_size),
                                      mode='bilinear',
                                      align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # with torch.no_grad():
        if self.scale_factor == 1:
            pull_weights = self.push.weight
        else:
            pull_weights = self.up_sampler(self.push.weight)
        # pull_weights.requires_grad = False

        bias = self.push.bias
        if self.push.bias is not None:
            bias = -self.push.bias

        push = self.relu(self.push(x))
        pull = self.relu(F.conv2d(x,
                                  -pull_weights,
                                  bias,
                                  self.push.stride,
                                  self.pull_padding, self.push.dilation,
                                  self.push.groups))

        alpha = self.alpha
        if self.train_alpha:
            # alpha is greater or equal than 0
            alpha = self.relu(self.alpha)

        if self.dual_output:
            x = torch.cat([push, pull], dim=1)
        else:
            x = push - alpha * pull
            # + self.bias.reshape(1, self.push.out_channels, 1, 1) #.repeat(s[0], 1, s[2], s[3])
        return x


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if self.filt_size == 1:
            a = np.array([1.])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None]*a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

def get_pad_layer_1d(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = args.expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None, size_lpf=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes)
        else:
            if size_lpf is None:
                self.conv1 = conv3x3(inplanes, planes, stride=stride)
            else:
                self.conv1 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=inplanes),
                                       conv3x3(inplanes, planes), )

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # instead of conv->bn->relu
        # bn->relu->conv
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        # Also Add Dropout
        out = F.dropout(out, p=0.1, training=self.training) # self.training comes from BaseClass
        out = self.conv2(out)
        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, size_lpf=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            if size_lpf is None:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                       padding=1, bias=False, stride=stride)
            else:
                self.conv2 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=planes),
                                           nn.Conv2d(planes, planes, kernel_size=3,
                                                     padding=1, bias=False),)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PushPullBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, train_alpha=False, size_lpf=None):
        super(PushPullBlock, self).__init__()
        if stride == 1:
            self.pp1 = PPmodule2d(inplanes, planes, kernel_size=3, padding=1, bias=False,
                                  # alpha=alpha_pp, scale=scale_pp,
                                  train_alpha=train_alpha)
        else:
            if size_lpf is None:
                self.pp1 = PPmodule2d(inplanes, planes, kernel_size=3, padding=1, bias=False,
                                      # alpha=alpha_pp, scale=scale_pp,
                                      train_alpha=train_alpha, stride=stride)
            else:
                self.pp1 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=inplanes),
                                     PPmodule2d(inplanes, planes, kernel_size=3,
                                                padding=1, bias=False, train_alpha=train_alpha), )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pp2 = PPmodule2d(planes, planes, kernel_size=3, 
                              padding=1, bias=False,  # alpha=alpha_pp, scale=scale_pp,
                              train_alpha=train_alpha)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.pp1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pp2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetCifar(nn.Module):
    """
    ResNet with Push-Pull for CIFAR: implemented on top of the official PyTorch ResNet implementation

    args:
        use_pp1 (bool, optional): if ''True'', use the Push-Pull layer to replace the first conv layer (default: False)
        pp_all (bool, optional): if ''True'', use the Push-Pull layer to replace all conv layers (default: False)
        pp_block1 (bool, optional): if ''True'', use the Push-Pull layer to replace all conv layers in the 1st residual block (default: False)
        train_alpha (bool, optional): if ''True'', the inhibition strength 'alpha' is trainable (default: False)
        size_lpf (int, optional): if specified, it uses an LPF filter of size ('size_lpf' x 'size_lpf') before downsampling operation (Zhang's paper) (default: None)
    """
    def __init__(self, block, layers, num_classes=10,
                 use_pp1=False, pp_all=False,
                 pp_block1=False, train_alpha=False, size_lpf=None):

        self.inplanes = 16
        super(ResNetCifar, self).__init__()

        if use_pp1:
            self.conv1 = PPmodule2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, train_alpha=train_alpha)
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        if pp_all:
            # Use push-pull inhibition at all layers
            self.layer1 = self._make_layer(PushPullBlock, 16, layers[0], train_alpha=train_alpha)
            self.layer2 = self._make_layer(PushPullBlock, 32, layers[1], train_alpha=train_alpha,
                                           stride=2, size_lpf=size_lpf)
            self.layer3 = self._make_layer(PushPullBlock, 64, layers[2], train_alpha=train_alpha,
                                           stride=2, size_lpf=size_lpf)
        else:
            # use push-pull inhibition in the first residual block only
            if pp_block1:
                self.layer1 = self._make_layer(PushPullBlock, 16, layers[0], train_alpha=train_alpha)
            else:
                self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2, size_lpf=size_lpf)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, size_lpf=size_lpf)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, train_alpha=False, size_lpf=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if size_lpf is None:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                # downsample according to Nyquist (from the paper of Zhang)
                downsample = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=self.inplanes),
                                           nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                                           nn.BatchNorm2d(planes * block.expansion)
                                           )

        layers = []
        if block is PushPullBlock:
            layers.append(block(self.inplanes, planes, stride, downsample, train_alpha=train_alpha, size_lpf=size_lpf))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, size_lpf=size_lpf))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if block is PushPullBlock:
                layers.append(block(self.inplanes, planes, train_alpha=train_alpha, size_lpf=size_lpf))
            else:
                layers.append(block(self.inplanes, planes, size_lpf=size_lpf))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def main():
    # Data loading code
    global best_prec1 # TODO : Strangely not including this as global gives error ... ??
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn')

    if args.dataset == 'cifar10':
        nclasses = 10
        dataset_train = NCIFAR10('./data', train=True,
                                 transform=transform_train,
                                 normalize_transform=normalize)
        dataset_test = NCIFAR10('./data', train=False,
                                transform=transform_test,
                                normalize_transform=normalize)
    else:
        raise RuntimeError('no other data set implementations available')
    '''
    elif args.dataset == 'cifar100':
        nclasses = 100
        dataset_train = NCIFAR100('./data', train=True, transform=transform_train,
                                  normalize_transform=normalize, download=True)
        dataset_test = NCIFAR100('./data', train=False, transform=transform_test,
                                 normalize_transform=normalize, download=True)
    elif args.dataset == 'svhn':
        nclasses = 10
        dataset_train = NSVHN('./data', split='train', transform=transform_train,
                              normalize_transform=normalize)
        dataset_test = NSVHN('./data', split='test', transform=transform_test,
                             normalize_transform=normalize)
    '''

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    # --------------------------------------------------------------------------------
    # create model
    output_dir = experiment_dir + 'resnet-cifar/'

    rnargs = {'use_pp1': args.pushpull,
                'pp_block1': args.pp_block1,
                'pp_all': args.pp_all,
                'train_alpha': args.train_alpha,
                'size_lpf': args.lpf_size}
    
    model = ResNetCifar(BasicBlock, [3, 3, 3], **rnargs)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))


    logger = None
    if args.tensorboard:
        ustr = datetime.datetime.now().strftime("%y-%m-%d_%H-%M_") + uuid.uuid4().hex[:3]
        logger = tensorboard_logger.Logger(experiment_dir + "tensorboard/" + args.name + '/' + ustr)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # --------------------------------------------------------------------------------

    use_cuda = torch.cuda.is_available()

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    if use_cuda:
        model = model.cuda()

    # optionally resume from a checkpoint
    epoch = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    lr_milestones = json.loads(args.milestones)
    if args.extra_epochs > 0:
        lr_milestones = list(set(lr_milestones + json.loads(args.extra_milestones)))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_milestones,
                                                     gamma=0.1)
    scheduler.step(epoch)

    directory = output_dir + "%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)

    for epoch in range(args.start_epoch, args.epochs + args.extra_epochs):
        fileout = open(output_dir + args.name + '/log.txt', "a+")
        # adjust_learning_rate(logger, optimizer, epoch + 1, args.epochs)
        scheduler.step()
        print('lr(', epoch, '): ', scheduler.get_lr())

        # train for one epoch
        train(logger, train_loader, model, criterion, optimizer, epoch, fileout)

        # evaluate on validation set
        prec1 = validate(logger, val_loader, model, criterion, epoch, fileout)
        fileout.close()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, output_dir)

    print('Best accuracy: ', best_prec1)
    fileout = open(output_dir + args.name + '/log.txt', "a+")
    fileout.write('Best accuracy: {}\n'.format(best_prec1))
    fileout.close()


def train(logger, train_loader, model, criterion, optimizer, epoch, file=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        output = output.detach()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i,
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses, top1=top1))
            if file is not None:
                file.write('Epoch: [{0}][{1}/{2}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time, loss=losses,
                                                                              top1=top1))

        if logger is not None:
            if i % (args.print_freq / 2) == 0:
                log_alpha_histograms(logger, epoch * len(train_loader) + i, model)
            logger.log_scalar('train_loss', losses.avg, epoch * len(train_loader) + i)
            logger.log_scalar('train_acc', top1.avg, epoch * len(train_loader) + i)


def to_np(x):
    return x.detach().cpu().numpy()


def log_alpha_histograms(logger, step, model):
    mode = 'train'
    # Log histograms of weights and grads.
    for h_name, h in zip(['model'], [model]):
        for tag, value in h.named_parameters():
            if 'alpha' in tag:
                tag = h_name + '/' + tag.replace('.', '/')
                logger.log_histogram(tag, to_np(value), step)
                # False is temporary to avoid this logging to happen
                if value.grad is not None:
                    logger.log_histogram(tag + '/grad', to_np(value.grad), step, bins=np.linspace(-.2, .2, 100))


def validate(logger, val_loader, model, criterion, epoch, file=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader),
                                                                  batch_time=batch_time, loss=losses,
                                                                  top1=top1))
            if file is not None:
                file.write('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(i, len(val_loader),
                                                                              batch_time=batch_time, loss=losses,
                                                                              top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    file.write(' * Prec@1 {top1.avg:.3f} \n'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        logger.log_scalar('val_loss', losses.avg, epoch)
        logger.log_scalar('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = output_dir + args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/" + filename
    torch.save(state, filename)
    # if is_best:
    #    shutil.copyfile(filename, 'resnet/runs/%s/' % (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(logger, optimizer, epoch, totepochs):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 180th epochs"""
    if args.dataset == 'cifar10' or 'cifar100':
        lr = args.lr * ((0.1 ** int(epoch >= totepochs * 0.50)) * (0.1 ** int(epoch >= totepochs * 0.75)) *
                        (0.1 ** int(epoch >= totepochs * 0.95)))

        # in the case some extra epochs are needed (full PP network case)
        lr = args.lr * (0.2 ** int(epoch >= totepochs * 1.1))
    elif args.dataset == 'svhn':
        lr = args.lr * ((0.1 ** int(epoch >= totepochs * 0.5)) *
                        (0.1 ** int(epoch >= totepochs * 0.75)))

    # log to TensorBoard
    if args.tensorboard:
        logger.log_scalar('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
