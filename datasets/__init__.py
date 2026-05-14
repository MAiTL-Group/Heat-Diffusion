import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
import sigpy.mri as mr
import sigpy as sp
import random
import mat73
import math
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader
from utils import *
import dill

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def get_mat_paths(dir_path, file_list):
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isdir(filepath):
            get_mat_paths(filepath, file_list)
        elif filename.endswith('.mat'):
            file_list.append(filepath)

class FastMRIv2DataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIv2DataSet, self).__init__()
        self.config = config
        if mode in ('train', 'retro'):
            self.kspace_dir = config.data.train_kspace_dir
        elif mode in ('sample', 'datashift'):
            self.kspace_dir = config.data.sample_kspace_dir
        else:
            raise NotImplementedError

        self.mode = mode
        self.crop_size = config.data.image_size
        self.file_list = []
        get_mat_paths(self.kspace_dir, self.file_list)
        self.num_slices = len(self.file_list)
        print(self.num_slices)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file = self.file_list[idx]

        if self.mode == 'sample':
            try:
                ksp = scio.loadmat(data_file)['ksp'] 
            except:
                ksp = mat73.loadmat(data_file)['ksp'] 
            calib = ksp 
            ksp = np.expand_dims(ksp,0)
            calib = np.expand_dims(calib,0)
            need_crop = 1
            if need_crop:
                NCH, NRO, NPE = ksp.shape
                ksp = sp.resize(ksp, (NCH, self.crop_size, self.crop_size))
                calib = sp.resize(calib, (NCH, 32, 32))
            return ksp, calib, self.file_list[idx]
        else: 
            try:
                img = scio.loadmat(data_file)['img']
            except:
                img = mat73.loadmat(data_file)['img']
            img = np.squeeze(img,0)
            return img

    def __len__(self):
        return int(np.sum(self.num_slices))

def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset)
    if config.data.dataset == 'single_channel':
        dataset = FastMRIv2DataSet(config, mode)
    elif config.data.dataset == 'fastMRIv2':
        dataset = FastMRIv2DataSet(config, mode)
    else:
        raise NotImplementedError
    
    if mode == 'train':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
    else:
        data = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=False)
    print(mode, "data loaded")
    return data


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
