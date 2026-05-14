from curses import KEY_ENTER
from operator import index
import os
import logging
import time
import glob
import sigpy as sp
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch import nn
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
from utils import *
from AdaptiveCombine import *
import mat73
from utils2 import get_mask_basic
import math

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "exp_log":
        betas = np.exp(np.linspace(np.log(beta_start), np.log(beta_end), num_diffusion_timesteps, dtype=np.float64))
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_sigma_schedule(sigma_schedule, *, sigma_start, sigma_end, num_diffusion_timesteps):
    if sigma_schedule == "exp_log":
        sigma = np.exp(np.linspace(np.log(sigma_start), np.log(sigma_end), num_diffusion_timesteps, dtype=np.float64))
        psigma = sigma_start *(sigma_end / sigma_start)** np.linspace(0, 1, num_diffusion_timesteps, dtype=np.float64)
    return sigma, psigma

def get_gaussian_mask(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps, Fourier=False, image_size=320):
    gmask = []
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "exp_log":
        betas = np.exp(np.linspace(np.log(beta_start), np.log(beta_end), num_diffusion_timesteps, dtype=np.float64))

    for i in range(num_diffusion_timesteps):
        gmask.append(Gaussian_mask(image_size,image_size,1,betas[i],True))
    return gmask

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        gmask = get_gaussian_mask(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start, # TODO: 0.5
            beta_end=config.diffusion.beta_end,     # TODO; 24
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps, # TODO: 1000
            Fourier=config.diffusion.Fourier
        )
        self.gmask = gmask 

        self.sigma = get_sigma_schedule(
            sigma_schedule=config.diffusion.sigma_schedule,
            sigma_start=config.diffusion.sigma_start,  # 0.01
            sigma_end=config.diffusion.sigma_end,  # 1
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        sigmas, psigmas = self.sigma
        self.sigmas = torch.from_numpy(sigmas).float().to(self.device)
        self.psigmas = torch.from_numpy(psigmas).float().to(self.device)
        self.type = config.diffusion.type

        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        train_loader = get_dataset(config, "train")
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        training_all_resolution = [256, 320, 320, 320, 384]
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            loss_sum = 0
            for i, batch in enumerate(train_loader):
                i_level = i % len(training_all_resolution)
                batch_im_size = training_all_resolution[i_level]
                data_time += time.time() - data_start
                t0 = time.time()
                label = batch
                label = sp.resize(label.numpy(), (label.shape[0], label.shape[1], batch_im_size, batch_im_size)) # TODO: lcc commit
                label = c2r(torch.from_numpy(label)).type(torch.FloatTensor).to(config.device) # 6*2*256*256 # TODO: lcc commit
                coeff = torch.max(Abs(label))
                label = label/coeff
                n = label.size(0)
                model.train()
                step += 1

                e = torch.randn_like(label)
                b = self.betas
                g = self.psigmas
                gmask = get_gaussian_mask(
                    beta_schedule=config.diffusion.beta_schedule,
                    beta_start=config.diffusion.beta_start, # TODO: 0.5
                    beta_end=config.diffusion.beta_end,     # TODO; 24
                    num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps, # TODO: 1000
                    Fourier=config.diffusion.Fourier,
                    image_size = batch_im_size
                )

                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, label, t, e, b, g, config.diffusion.type, gmask, None, None)
                loss_sum += loss
                tb_logger.add_scalar("loss", loss, global_step=step)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)
                
                param_num = sum(param.numel() for param in model.parameters())
                if step % 10 == 0:
                    print('Epoch', epoch + 1, '/', config.training.n_epochs, 'Step', step,
                            'loss = ', loss.cpu().data.numpy(),
                            'loss mean =', loss_sum.cpu().data.numpy() / (i + 1),
                            'time', time.time() - t0, 'param_num', param_num)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.config.workdir, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.config.workdir, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained: 
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.config.sampling.weight, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.config.sampling.weight, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            for name, param in model.named_parameters():
               print(f"parameter dtype: {param.dtype}")

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        test_loader = get_dataset(self.config, "sample") # TODO:
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        mask_type = ['gaussian2d','uniformrandom2d','gaussian1d','uniform1d','regular1d','poisson',
                    'poisson1d','regularlinear']    
        config.sampling.mask_type = mask_type[-1]

        with torch.no_grad():
            for index, batch in enumerate(test_loader):
                k0, calib, savefile = batch
                savefile_path = savefile[0]
                relative_path = os.path.relpath(savefile_path, config.data.sample_kspace_dir)

                gmask = get_gaussian_mask(
                beta_schedule=config.diffusion.beta_schedule,
                beta_start=config.diffusion.beta_start, # TODO: 0.5
                beta_end=config.diffusion.beta_end,     # TODO; 24
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps, # TODO: 1000
                Fourier=config.diffusion.Fourier,
                image_size = config.data.image_size
                )

                k0 = c2r(k0).type(torch.FloatTensor).to(config.device)
                calib = r2c(c2r(calib).type(torch.FloatTensor).to(config.device))

                mask = torch.where(r2c(k0) != 0, torch.ones_like(r2c(k0)), torch.zeros_like(r2c(k0)))
     
                atb = r2c(k0) * mask
                     
                csm = torch.ones_like(atb)

                _, cg_sense = cgSENSE(atb, atb, csm, mask, 30, 1e-5)
                
                coeff = torch.max(Abs(c2r(cg_sense)))
                cg_sense = cg_sense/coeff
                atb = atb/coeff
                recon = cg_sense
                label = r2c(Emat_xyt(k0, True, c2r(csm), 1))

                snr, lr = estimate_snr_from_regions(label)
                print(lr)
       
                x = self.sample_image(c2r(cg_sense), atb, csm, mask, model, gmask, lr)

                full_path = os.path.join(self.args.image_folder, relative_path)

                dir_path = os.path.dirname(full_path)
     
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
      
                matfile = {'cg_sense': recon.squeeze().cpu().numpy(),
                           'label': label.squeeze().cpu().numpy(),
                    'diff': r2c(x).squeeze().cpu().numpy()}
                sio.savemat(full_path, matfile)

    def sample_image(self, x, atb, csm, mask, model, gmask, lr, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps 
                seq = range(0, self.num_timesteps, skip) 
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq) and s]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps, generalized_ddpm_steps

            xs = generalized_steps(x, atb, csm, mask, seq, model, gmask, lr, self.betas, self.sigmas, self.psigmas, self.type, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[-1][-1]
        return x

    def test(self):
        pass
