import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
# import tensorflow as tf
import logging
from sigpy.mri import poisson
import sigpy.plot as spi
import math

def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False


def save_mat(save_dict, variable, file_name, index=0, normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.mat')
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def hfssde_save_mat(config, variable, variable_name='recon', normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    save_dict = config.sampling.folder
    file_name = config.training.sde + '_acc' + config.sampling.acc + '_acs' + config.sampling.acs \
                    + '_epoch' + str(config.sampling.ckpt)
    file = os.path.join(save_dict, str(file_name) + '.mat')
    datadict = {variable_name: np.squeeze(variable)}
    scio.savemat(file, datadict)


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.glob(os.path.join(folder, pattern),recursive=True)]
    return sorted(files)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def to_tensor(x):
    re = np.real(x)
    im = np.imag(x)
    x = np.concatenate([re, im], 1)
    del re, im
    return torch.from_numpy(x)

def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty:starty + cropy, startx: startx + cropx]

def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x

def Gaussian_mask(nx, ny, Rmax, t, Fourier=True):
    if nx % 2 == 0:
        ix = np.arange(-nx//2, nx//2)
    else:
        ix = np.arange(-nx//2, nx//2 + 1)

    if ny % 2 == 0:
        iy = np.arange(-ny//2, ny//2)
    else:
        iy = np.arange(-ny//2, ny//2 + 1)

    wx = Rmax * ix / (nx / 2)
    wy = Rmax * iy / (ny / 2)

    rwx, rwy = np.meshgrid(wx, wy)
    if Fourier:
        R = np.exp(-((rwx ** 2 + rwy ** 2)* t ** 2) / 2 )
    else:
        R = np.exp(-(rwx ** 2 + rwy ** 2) / (2 * t ** 2))
        
    W = R.astype(np.float32)

    return W

def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

def pad_or_crop_tensor(input_tensor, target_shape):
    input_shape = input_tensor.shape
    pad_width = []

    for i in range(len(target_shape)):
        diff = target_shape[i] - input_shape[i]
        pad_before = max(0, diff // 2)
        pad_after = max(0, diff - pad_before)

        pad_width.append((pad_before, pad_after))
    padded_tensor = np.pad(input_tensor, pad_width, mode='constant')

    cropped_tensor = padded_tensor[:target_shape[0], :target_shape[1], :target_shape[2], :target_shape[3]]

    return cropped_tensor

def get_mask(config, caller):
    if caller == 'sde':
        if config.training.mask_type == 'low_frequency':
            mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acs + '.mat'
        elif config.training.mask_type == 'center':
            mask_file = 'mask/' +  config.training.mask_type + '_acc2.mat'
        else:
            mask_file = 'mask/' +  config.training.mask_type + "_acc" + config.training.acc \
                                                + '_acs' + config.training.acs + '.mat'
    elif caller == 'sample':
        mask_file = 'mask/' +  config.sampling.mask_type + "_acc" + config.sampling.acc \
                                                + '_acs' + config.sampling.acs + '.mat'
    elif caller == 'acs':
        mask_file = 'mask/low_frequency_acs18.mat'
    mask = scio.loadmat(mask_file)['mask']
    mask = mask.astype(np.complex128)
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).to(config.device)

    return mask

def get_mask_basic(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False,min_acc=2,linear_w=1,linear_density=1,pf=1):
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor
  elif type.endswith('1d'):
    Nsamp = size // acc_factor
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    else:
      for i in range(batch_size):
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)
    if fix:
      mask_vec = torch.zeros([1, size * size])
      samples = np.random.choice(size * size, int(Nsamp))
      mask_vec[:, samples] = 1
      mask_b = mask_vec.view(size, size)
      mask[:, ...] = mask_b
    else:
      for i in range(batch_size):
        mask_vec = torch.zeros([1, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[i, ...] = mask_b
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)
    mean = size // 2
    std = size * (15.0 / 96)
    Nsamp_center = int(size * center_fraction)
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[... , int_samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[... , c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, :, int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))
      mask[..., samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))
        mask[i, :, :, samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  elif type == 'regular1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = int(Nsamp - Nsamp_center)
      mask[..., 4:-1:acc_factor] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = int(Nsamp - Nsamp_center)
        mask[i, :, :, 4:-1:acc_factor] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  elif type == 'poisson':
    mask = poisson((img.shape[-2], img.shape[-1]), accel=acc_factor)
    mask = torch.from_numpy(mask)
  elif type == 'poisson1d':
    mask_pattern = abs(poisson((size, 2), accel=acc_factor)[:,1])
    mask = torch.zeros_like(img)
    mask[..., :] = torch.from_numpy(mask_pattern)
  elif type == 'regularlinear':
    mask = torch.zeros_like(img)

    for i in range(batch_size):
      Nsamp_center_half = int(size * center_fraction/2)
      n_half = int(size/2)
      n_half_regular = n_half-Nsamp_center_half
      Nsamp_half = int(n_half_regular/acc_factor)
      Nsample_linear = int(Nsamp_half*linear_w)
      Nsample_const = Nsamp_half - Nsample_linear
      const_acc = round((n_half_regular-0.5*min_acc*Nsample_linear)/(Nsample_const + 0.5*Nsample_linear/linear_density))
      max_acc  = round(const_acc/linear_density)
      seg = max_acc - min_acc + 1
      Nsamp_seg = int(Nsample_linear/seg)
      if seg>1:
        Nsamp_seg_last = Nsamp_half-Nsamp_seg*(seg-1)
      arr1 = [1];arr2 = [2]
      for j in range(min_acc,max_acc):
        for k in range(Nsamp_seg):
            arr1.append(arr1[-1] + j)
            arr2.append(arr2[-1] + j)
      if seg>1:
        for x in range(Nsamp_seg_last):
          if arr1[-1] + max_acc < int(n_half_regular*linear_w):
            arr1.append(arr1[-1] + max_acc)
          if arr2[-1] + max_acc < int(n_half_regular*linear_w):
            arr2.append(arr2[-1] + max_acc)
      while arr1[-1] + const_acc<n_half_regular:
        arr1.append(arr1[-1] + const_acc)
      while arr2[-1] + const_acc<n_half_regular:
        arr2.append(arr2[-1] + const_acc)
      mask_p1 = np.ones(size -2*n_half_regular)
      mask_p2 = np.zeros(n_half_regular);mask_p2[arr1]=1
      mask_p3 = np.zeros(n_half_regular);mask_p3[arr2]=1;mask_p3 = np.flip(mask_p3)
      mask1d = np.concatenate((mask_p3,mask_p1,mask_p2))
      mask = torch.zeros_like(img)
      mask[i, :, :, :] = torch.from_numpy(mask1d.astype(complex)).unsqueeze(0).unsqueeze(0).repeat(mask.shape[1],mask.shape[2],1)

  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')
  if pf<1:
      pf_line = round(mask.shape[3]*(1-pf))
      mask[:, :, :, :pf_line] = 0

  if type == 'poisson':
      Nacc = float(mask.shape[0]*mask.shape[1] / np.sum(abs(mask.cpu().numpy())))
      mask = mask[None,None,:,:]
  else:
      Nacc = float(mask.shape[3] / np.sum(abs(mask[0, 0, 0, :].cpu().numpy())))
  mask1d = abs(mask[0,0,0,:].cpu().numpy()).astype(np.int32)
  return mask,Nacc,mask1d

def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x

def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)
    return x


def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x

def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)
    return x

def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            if b.ndim == 4:
                b = ifft2c_2d(b)
            else:
                b = ifft2c(b)
            x = c2r(b)
        else:
            b = r2c(b)
            if b.ndim == 4:
                b = fft2c_2d(b) * mask
            else:
                b = fft2c(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b) * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)

            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b
            x = c2r(x)
    return x


def Emat_xyt_complex(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)

        else:
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b

    return x


def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x


def sos(x):
    xr, xi = torch.chunk(x, 2, 1)
    x = torch.pow(torch.abs(xr), 2)+torch.pow(torch.abs(xi), 2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x, 0.5)
    x = torch.unsqueeze(x, 1)
    return x


def Abs(x):
    x = r2c(x)
    return torch.abs(x)


def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))

    return result


def TV(x, norm='L1'):
    nb, nc, nx, ny = x.size()
    Dx = torch.cat([x[:, :, 1:nx, :], x[:, :, 0:1, :]], 2)
    Dy = torch.cat([x[:, :, :, 1:ny], x[:, :, :, 0:1]], 3)
    Dx = Dx - x
    Dy = Dy - x
    tv = 0
    if norm == 'L1':
        tv = torch.mean(torch.abs(Dx)) + torch.mean(torch.abs(Dy))
    elif norm == 'L2':
        Dx = Dx * Dx
        Dy = Dy * Dy
        tv = torch.mean(Dx) + torch.mean(Dy)
    return tv


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


import torch
import numpy as np
import os

sqrt = np.sqrt
import torch.nn.functional as F
from torch.fft import ifft, fft
import torchvision.transforms as T


def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def ifft2c(x):
    device = x.device
    ny = torch.Tensor([x.shape[-1]])
    ny = ny.to(device)
    nx = torch.Tensor([x.shape[-2]])
    nx = nx.to(device)
    x = ifftshift(x, axes=-2)
    x = torch.transpose(x, -2, -1)
    x = ifft(x)
    x = torch.transpose(x, -2, -1)
    x = torch.mul(fftshift(x, axes=-2), torch.sqrt(nx))
    x = ifftshift(x, axes=-1)
    x = ifft(x)
    x = torch.mul(fftshift(x, axes=-1), torch.sqrt(ny))
    return x


def fft2c(x):
    device = x.device
    ny = torch.Tensor([x.shape[-1]]).to(device)
    nx = torch.Tensor([x.shape[-2]]).to(device)
    x = ifftshift(x, axes=-2)
    x = torch.transpose(x, -2, -1)
    x = fft(x)
    x = torch.transpose(x, -2, -1)
    x = torch.div(fftshift(x, axes=-2), torch.sqrt(nx))
    x = ifftshift(x, axes=-1)
    x = fft(x)
    x = torch.div(fftshift(x, axes=-1), torch.sqrt(ny))
    return x


def fft1c(x, dim):
    device = x.device
    nt = torch.Tensor([x.shape[dim]]).to(device)
    x = ifftshift(x, axes=dim)
    x = torch.transpose(x, dim, -1)
    x = fft(x)
    x = torch.transpose(x, dim, -1)
    x = torch.div(fftshift(x, axes=dim), torch.sqrt(nt))
    return x


def ifft1c(x, dim):
    device = x.device
    nt = torch.Tensor([x.shape[dim]]).to(device)
    x = ifftshift(x, axes=dim)
    x = torch.transpose(x, dim, -1)
    x = ifft(x)
    x = torch.transpose(x, dim, -1)
    x = torch.mul(fftshift(x, axes=dim), torch.sqrt(nt))
    return x

def ssos(x):
    xr, xi = torch.chunk(x, 2, 1)
    x = torch.pow(torch.abs(xr), 2) + torch.pow(torch.abs(xi), 2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x, 0.5)
    return x


def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            x = ifft2c(b)
            x = c2r(x)
        else:
            b = r2c(b)
            b = fft2c(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b)
            x = x * mask
            x = ifft2c(x)
            x = torch.mul(x, torch.conj(csm))
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = torch.mul(b, csm)
            b = fft2c(b)
            x = mask * b
            x = c2r(x)

    return x


def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))
    return result


def sos(x):
    x = torch.pow(torch.abs(x), 2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x, 0.5)
    return x


def complex_kernel_forward(filter, i):
    filter = torch.squeeze(filter[i])
    filter_real = torch.real(filter)
    filter_img = torch.imag(filter)
    kernel_real = torch.cat([filter_real, -filter_img], 1)
    kernel_imag = torch.cat([filter_img, filter_real], 1)
    kernel_complex = torch.cat([kernel_real, kernel_imag], 0)
    return kernel_complex


def conv2(x1, x2):
    return F.conv2d(x1, x2, padding='same')


def ksp2float(ksp, i):
    kdata = torch.squeeze(ksp[i]) 
    if len(kdata.shape) == 3:
        kdata = torch.unsqueeze(kdata, 0)

    kdata_float = torch.cat([torch.real(kdata), torch.imag(kdata)], 1)
    return kdata_float


def spirit(kernel, ksp):
    nb = ksp.shape[0]
    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
    re, im = torch.chunk(res_i, 2, 1)
    res = torch.complex(re, im) - ksp
    return res

def adjspirit(kernel, ksp):
    nb = kernel.shape[0]
    filter = torch.permute(kernel, (0, 2, 1, 3, 4))
    filter = torch.conj(filter.flip(dims=[-2, -1]))

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
    re, im = torch.chunk(res_i, 2, 1)
    res = torch.complex(re, im) - ksp

    return res


def dot_batch(x1, x2):
    batch = x1.shape[0]
    res = torch.reshape(x1 * x2, (batch, -1))
    return torch.sum(res, 1)


class ConjGrad:
    def __init__(self, A, rhs, max_iter=5, eps=1e-10):
        self.A = A
        self.b = rhs
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, x):
        x = CG(x, self.b, self.A, max_iter=self.max_iter, eps=self.eps)
        return x


def CG(x, b, A, max_iter, eps):
    r = b
    p = r
    rTr = dot_batch(torch.conj(r), r)
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    num_iter = 0
    for iter in range(max_iter):
        if rTr.abs().max() < eps:
            break
        Ap = A.A(p)
        alpha = rTr / dot_batch(torch.conj(p), Ap)
        alpha = torch.reshape(alpha, reshape)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = dot_batch(torch.conj(r), r)
        beta = rTrNew / rTr
        beta = torch.reshape(beta, reshape)
        p = r + beta * p
        rTr = rTrNew

        num_iter += 1
    return x


class Aclass_spirit:
    def __init__(self, kernel, mask, lam):
        self.kernel = kernel
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        ksp = spirit(self.kernel, ksp)
        ksp = adjspirit(self.kernel, ksp)
        return ksp

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res


def sense(csm, ksp):
    m = torch.sum(ifft2c(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c(m * csm)
    return res - ksp


def adjsense(csm, ksp):
    m = torch.sum(ifft2c(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c(m * csm)
    return res - ksp


class Aclass_sense:
    def __init__(self, csm, mask, lam):
        self.s = csm
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        Ax = sense(self.s, ksp)
        AHAx = adjsense(self.s, Ax)
        return AHAx

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res


def cgSPIRiT(x0, ksp, kernel, mask, niter, lam):
    Aobj = Aclass_spirit(kernel, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    return x * (1 - mask) + ksp

def cgSPIRiT_heat(x0, ksp, csm, kernel, mask, niter, lam):
    Aobj = Aclass_spirit(kernel, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm),1).unsqueeze(1)
    return x, res

def cgSENSE(ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=torch.zeros_like(y))
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm.unsqueeze(2)), 1)
    return x, res

def cgSENSE_heat(x0, ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm), 1, keepdim=True)
    return x, res

def SPIRiT_Aobj(kernel,ksp):
    ksp = spirit(kernel, ksp)
    ksp = adjspirit(kernel, ksp)
    return ksp


def add_noise(x, snr):
    x_ = x.view(x.shape[0], -1)
    x_power = torch.sum(torch.pow(torch.abs(x_), 2), dim=1, keepdim=True) / x_.shape[1]
    snr = 10 ** (snr / 10)
    noise_power = x_power / snr
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    noise_power = torch.reshape(noise_power, reshape)
    if x.dtype == torch.float32:
        noise = torch.sqrt(noise_power) * torch.randn(x.size(), device=x.device)
    else:
        noise = torch.sqrt(0.5 * noise_power) * (torch.complex(torch.randn(x.size(), device=x.device),
                                                               torch.randn(x.size(), device=x.device)))
    return x + noise


def blur_and_noise(x, kernel_size=7, sig=0.1, snr=10):
    x_org = x
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sig)
    if x.dtype == torch.float32:
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
    else:
        x = c2r(x)
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))

    x_blur = transform(x_)
    x_blur = torch.reshape(x_blur, x.shape)
    x_blur_noise = add_noise(x_blur, snr=snr)
    if x_org.dtype == torch.float32:
        return x_blur_noise
    else:
        return r2c(x_blur_noise)


def ISTA(x0, ksp, csm, mask, niter, lam):
    x_dc = torch.sum(
        ifft2c(fft2c(x0.unsqueeze(1) * csm.unsqueeze(2)) * (1 - mask) + ksp) * torch.conj(csm.unsqueeze(2)), 1)
    f = fft2c(x0.unsqueeze(1) * csm.unsqueeze(2)) * mask - ksp
    x = torch.zeros_like(x0)
    for iter in range(niter):
        Ax = fft2c(x.unsqueeze(1) * csm.unsqueeze(2)) * mask
        r = x - 1 * torch.sum(ifft2c((Ax - f)) * torch.conj(csm.unsqueeze(2)), 1)
        x = torch.sgn(r) * torch.nn.ReLU()(torch.abs(r) - lam)

    return x + x0, x_dc


def GD_SENSE(ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = Aobj.ATA(ksp) * (1 - mask)
    x = ksp

    for iter in range(niter):
        gd = Aobj.A(x * (1 - mask)) * (1 - mask) + lam * y
        x = x - gd
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm.unsqueeze(2)), 1)
    return x, res


def matmul_cplx(x1, x2):
    return torch.view_as_complex(
        torch.stack((x1.real @ x2.real - x1.imag @ x2.imag, x1.real @ x2.imag + x1.imag @ x2.real), dim=-1))


def LSrec(ksp, mask, csm, lambda_L=0.005, lambda_S=0.01, max_iter=50, tol=2e-3):
    M = torch.sum(ifft2c(ksp) * torch.conj(csm.unsqueeze(2)), 1)
    nb, nt, nx, ny = M.shape
    M = torch.reshape(M, (nb, nt, nx * ny))
    L = M
    S = M - L
    for iter in range(max_iter):
        M0 = M
        U, St, Vh = torch.linalg.svd(M, full_matrices=False)

        thres = lambda_L * St[:, 0]
        St = torch.diag_embed(torch.nn.ReLU()(St - thres.unsqueeze(1)) * (St / St.abs()))

        US = matmul_cplx(U, St.type(torch.complex64))
        L = matmul_cplx(US, Vh)

        S_tmp = fft1c(M - L, 1)
        S = ifft1c(torch.nn.ReLU()(S_tmp.abs() - lambda_S) * (S_tmp / S_tmp.abs()), 1)

        m = torch.reshape(L + S, (nb, nt, nx, ny))
        resk = fft2c(m.unsqueeze(1) * csm.unsqueeze(2)) * mask - ksp
        M = L + S - torch.reshape(torch.sum(ifft2c(resk) * torch.conj(csm.unsqueeze(2)), 1), (nb, nt, nx * ny))
        rel_tol = (torch.norm(M - M0, dim=[1, 2]) / torch.norm(M0, dim=[1, 2]))
        if rel_tol.min() < tol:
            break
    L = torch.reshape(L, (nb, nt, nx, ny))
    S = torch.reshape(S, (nb, nt, nx, ny))
    return L + S


class Aclass:
    def __init__(self, csm, mask, lam):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.).float())
        self.lam = lam

    def myAtA(self, img):
        x = Emat_xyt(img, False, self.csm, self.mask)
        x = Emat_xyt(x, True, self.csm, self.mask)
        return x + self.lam * img


def myCG(A, Rhs, x0, it):
    Rhs = r2c(Rhs) + A.lam * r2c(x0)
    x = r2c(x0)
    i = 0
    r = Rhs - r2c(A.myAtA(x0))
    p = r
    rTr = torch.sum(torch.conj(r)*r).float()

    while i < it:

        Ap = r2c(A.myAtA(c2r(p)))
        alpha = rTr / torch.sum(torch.conj(p)*Ap).float()
        alpha = torch.complex(alpha, torch.tensor(0.).float().cuda())
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r)*r).float()
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.).float().cuda())
        p = r + beta * p
        i = i + 1
        rTr = rTrNew

    return c2r(x)


def check_and_print_nan_positions(tensor_name, tensor):
    print(f"{tensor_name}", tensor.view(-1)[0]) 
    contains_nan = torch.isnan(tensor).any()
    print(f"{tensor_name}", contains_nan)

    nan_indices = torch.isnan(tensor).nonzero(as_tuple=True)
    if torch.tensor((nan_indices[0].shape)) > 0:
        print(nan_indices[0].shape)
        for idx in zip(*nan_indices):
            print(f"Position: {idx}")

def compute_sigma(sigma, t):
    sigma = torch.cat([torch.zeros(1).to(sigma.device), sigma], dim=0) # 1001
    g = sigma.index_select(dim=0, index = t+1).view(-1, 1, 1, 1)
    return g
