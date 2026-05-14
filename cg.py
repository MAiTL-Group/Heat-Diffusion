import torch
import numpy as np
import os
sqrt = np.sqrt
import torch.nn.functional as F
from torch.fft import ifft, fft
import torchvision.transforms as T
import sigpy as sp
import sigpy.mri.app as MR
from torch.utils.dlpack import to_dlpack, from_dlpack
from tqdm import tqdm


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

def r2c(x):
    re, im = torch.chunk(x,2,1)
    x = torch.complex(re,im)
    return x.squeeze(1)

def c2r(x):
    x = torch.cat([torch.real(x),torch.imag(x)],1)
    return x

def ssos(x):
    xr, xi = torch.chunk(x,2,1)
    x = torch.pow(torch.abs(xr),2)+torch.pow(torch.abs(xi),2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x,0.5)
    return x

def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            x = ifft2c(b)
            x = c2r(x)
        else:
            b = r2c(b)
            b = fft2c(b)*mask
            x = c2r(b)
    else:
        if inv:
            x = r2c(b)  
            x = x*mask
            x = ifft2c(x)
            x = torch.mul(x,torch.conj(csm))
            x = torch.sum(x,1)
            x = torch.unsqueeze(x,1)
            x = c2r(x)
            
        else:
            b = r2c(b)
            b = torch.mul(b, csm)
            b = fft2c(b)
            x = mask*b
            x = c2r(x)
            
    return x

def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))
    return result

def sos(x):
    x = torch.pow(torch.abs(x),2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x,0.5)
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
    """
    :param kernel: nb, nc, nc_s, kx, ky 
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc, nx, ny
    """
    nb = ksp.shape[0]

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp,(0,2,1,3,4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    else:        
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    
    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i,(0,2,1,3,4))
        ksp = torch.permute(ksp,(0,2,1,3,4))
    
    re, im = torch.chunk(res_i,2,1)
    res = torch.complex(re,im) - ksp
    return res

def adjspirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky 
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc_s, nx, ny
    """

    nb = kernel.shape[0]
    filter = torch.permute(kernel, (0, 2, 1, 3, 4)) 
    filter = torch.conj(filter.flip(dims=[-2,-1]))

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp,(0,2,1,3,4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    else:        
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    
    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i,(0,2,1,3,4))
        ksp = torch.permute(ksp,(0,2,1,3,4))

    re, im = torch.chunk(res_i,2,1)
    
    res = torch.complex(re,im) - ksp

    return res


def dot_batch(x1, x2):
    batch = x1.shape[0]
    res = torch.reshape(x1 * x2, (batch, -1))
    # res = torch.reshape(x1 * x2, (-1, 1))
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
    """
    :param csm: nb, nc, nx, ny 
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c(ksp) * torch.conj(csm.unsqueeze(2)),1)    
    res  = fft2c(m.unsqueeze(1) * csm.unsqueeze(2))
    return res - ksp

def adjsense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny 
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c(ksp) * torch.conj(csm.unsqueeze(2)),1)    
    res  = fft2c(m.unsqueeze(1) * csm.unsqueeze(2))
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

def cgSPIRiT(ksp, kernel, mask, niter, lam):
    Aobj = Aclass_spirit(kernel, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=torch.zeros_like(y))
    return x * (1 - mask) + ksp

def cgSENSE(ksp, csm, mask, x0, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x0 = Emat_xyt(x0, False, csm, 1)
    x = cg_iter.forward(x=r2c(x0))
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm.unsqueeze(2)),1)
    return res

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
    if snr != 0:
        x_blur = add_noise(x_blur, snr=snr)

    if x_org.dtype == torch.float32:
        return x_blur
    else:
        return r2c(x_blur)


def ISTA(x0, ksp, csm, mask, niter, lam):
    x_dc = torch.sum(ifft2c(fft2c(x0.unsqueeze(1) * csm.unsqueeze(2))*(1-mask) + ksp) * torch.conj(csm.unsqueeze(2)),1)      
    f = fft2c(x0.unsqueeze(1) * csm.unsqueeze(2))*mask - ksp
    x = torch.zeros_like(x0)
    for iter in range(niter):        
        Ax = fft2c(x.unsqueeze(1) * csm.unsqueeze(2))*mask
        r = x - 1 * torch.sum(ifft2c((Ax - f)) * torch.conj(csm.unsqueeze(2)),1)                
        x = torch.sgn(r) * torch.nn.ReLU()(torch.abs(r)-lam)
        
    return x + x0, x_dc

def GD_SENSE(ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = Aobj.ATA(ksp) * (1 - mask)
    x = ksp

    for iter in range(niter):
        gd = Aobj.A(x * (1 - mask)) * (1 - mask) + lam * y
        x = x - gd
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c(x) * torch.conj(csm.unsqueeze(2)),1)
    return x, res

def matmul_cplx(x1, x2):
    return torch.view_as_complex(torch.stack((x1.real@x2.real-x1.imag@x2.imag, x1.real@x2.imag + x1.imag@x2.real),dim=-1))

def LSrec(ksp, mask, csm, lambda_L=0.005, lambda_S=0.01, max_iter=50, tol=2e-3):
    M = torch.sum(ifft2c(ksp) * torch.conj(csm.unsqueeze(2)),1)
    nb, nt, nx, ny = M.shape
    M = torch.reshape(M, (nb, nt, nx*ny))
    L = M
    S = M - L
    for iter in range(max_iter):
        M0 = M
        U, St, Vh = torch.linalg.svd(M, full_matrices=False)
        
        thres = lambda_L * St[:,0]
        St = torch.diag_embed(torch.nn.ReLU()(St - thres.unsqueeze(1))*torch.sgn(St))
        
        US = matmul_cplx(U, St.type(torch.complex64))
        L = matmul_cplx(US, Vh)

        S_tmp = fft1c(M-L, 1)
        S = ifft1c(torch.nn.ReLU()(S_tmp.abs() - lambda_S)*torch.sgn(S_tmp), 1)

        m = torch.reshape(L+S, (nb, nt, nx, ny))
        resk = fft2c(m.unsqueeze(1) * csm.unsqueeze(2)) * mask - ksp
        M = L+S-torch.reshape(torch.sum(ifft2c(resk) * torch.conj(csm.unsqueeze(2)),1), (nb, nt, nx*ny))

        rel_tol = (torch.norm(M - M0, dim=[1,2])/torch.norm(M0, dim=[1,2]))
        if  rel_tol.min() < tol:
            break
    L = torch.reshape(L, (nb, nt, nx, ny))
    S = torch.reshape(S, (nb, nt, nx, ny))
    return L+S


def ESPIRiT_calib(ksp, i, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp[i]) 
    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm

def ESPIRiT_calib_prescan(ksp_prescan, ksp, i, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp_prescan[i]) 
    calib = kdata.shape[-1]
    zpad = T.CenterCrop((int(ksp.shape[-2]),int(ksp.shape[-1])))
    kdata = zpad(kdata)

    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')
    return np.dot(A.T.conj(), A)

def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))
            count += 1
    return res

def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))
    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1

    idxY = np.where(dummyK)
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty) # fast 1s

    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)

def spirit_calibrate(acs, kSize, lamda=0.01):
    nCoil = acs.shape[-1]
    AtA = dat2AtA(acs,kSize)
            
    spirit_kernel = np.zeros((nCoil,nCoil,*kSize),dtype='complex128')
    for c in range(nCoil):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)
        spirit_kernel[c] = np.transpose(tmp,[2,0,1])
    spirit_kernel = np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!    
    return spirit_kernel

def L1SENSE(ksp, csm, lamda, gpu_id):
    ksp_gpu = cu_from_dlpack(to_dlpack(ksp.squeeze()))
    csm_gpu = cu_from_dlpack(to_dlpack(csm.squeeze()))
    rec = MR.L1WaveletRecon(ksp_gpu, csm_gpu, lamda,device=sp.Device(gpu_id), show_pbar=False).run()
    rec = from_dlpack(rec.toDlpack())
    return rec

def grad_3D(input):
    nb,nc,nt,nx,ny=input.shape
    x_t = torch.cat([input[:,:,1:nt,:,:], input[:,:,0:1,:,:]],dim=2)
    x_v = torch.cat([input[:,:,:,2:nx,:], input[:,:,:,0:2,:]],dim=3)
    x_h = torch.cat([input[:,:,:,:,2:ny], input[:,:,:,:,0:2]],dim=4)
    Dt = x_t - input
    Dv = x_v - input
    Dh = x_h - input
    grad = torch.sqrt(torch.pow(Dt, 2) + torch.pow(Dv, 2) + torch.pow(Dh, 2) + 1e-6)
    return grad
