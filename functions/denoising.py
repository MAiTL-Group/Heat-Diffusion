import torch
from utils import *
import torchvision.utils as tvu
import pytorch_wavelets as wavelets

wave_name = 'haar'  # wavelet family (e.g. haar, db1)
mode = 'zero'       # boundary padding mode
dwt = wavelets.DWTForward(J=3, mode=mode, wave=wave_name).cuda()
idwt = wavelets.DWTInverse( mode=mode, wave=wave_name).cuda()


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0) # 1001
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_beta(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)

    a = beta[t.item()].view(-1, 1, 1, 1)
    # a = beta.cumprod(dim=0).index_select(t.item(), t + 1).view(-1, 1, 1, 1)
    return a

def compute_sigma(sigma, t):
    sigma = torch.cat([torch.zeros(1).to(sigma.device), sigma], dim=0) # 1001
    g = sigma.index_select(dim=0, index = t+1).view(-1, 1, 1, 1)
    return g


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """
    def __init__(self, csm, mask, lam):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.).float())
        self.lam = lam

    def myAtA(self, img):
        
        x = Emat_xyt(c2r(img), False, c2r(self.csm), self.mask)
        x = r2c(Emat_xyt(x, True, c2r(self.csm), self.mask))
        
        return x + self.lam * img


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """
    def __init__(self, csm, mask, lam):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.).float())
        self.lam = lam

    def myAtA(self, img):
        
        x = Emat_xyt(c2r(img), False, c2r(self.csm), self.mask)
        x = r2c(Emat_xyt(x, True, c2r(self.csm), self.mask))
        
        return x + self.lam * img


def myCG(A, Rhs, x0, it):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    #print('Rhs1', Rhs.shape, Rhs.dtype) #Rhs1.shape torch.Size([2, 256, 232])

    Rhs = r2c(Rhs) + A.lam * r2c(x0)
    
    # x = torch.zeros_like(Rhs)
    x = r2c(x0)
    i = 0
    r = Rhs - A.myAtA(r2c(x0))
    p = r
    rTr = torch.sum(torch.conj(r)*r).float()

    while i < it:
        Ap = A.myAtA(p)
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


def generalized_steps(x, atb, csm, mask, seq, model, gmask, lr, b, g, pg, type, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1]) 
        seq_next = seq_next[:50] 
        seq = seq[:50] 
        x0_preds = []

        i = seq[-1]
        t0 = (torch.ones(n) * i).to(x.device)
        gmask_i = gmask[i]
        gmask_i = torch.from_numpy(gmask_i).float().to(x.device)
        
        x = c2r(ifft2c_2d(gmask_i*fft2c_2d(r2c(x)))) + torch.randn_like(x)*compute_sigma(g, t0.long())
        xs = [x]

        
        iteri = 0
        for i, j in zip(reversed(seq), reversed(seq_next)):
          
            t = (torch.ones(n) * i).to(x.device)
            print('t: ', i)
            next_t = (torch.ones(n) * j).to(x.device)
            
            if type == "Heat":
                xt = xs[-1].to(x.device)
                gt = compute_sigma(g, t.long()) # b: 1000
                gt_next = compute_sigma(g, next_t.long())
                gmask_i = gmask[i]
                gmask_i = torch.from_numpy(gmask_i).float().to(x.device)
                gmask_next = gmask[j]
                gmask_next = torch.from_numpy(gmask_next).float().to(x.device)
                with torch.enable_grad():
                    xt = xt.clone()
                    xt = torch.autograd.Variable(xt, requires_grad=True)
                    et = model(xt, t) 
                    x0_t =  xt - et
                    grad = Emat_xyt(x0_t.detach(),False,c2r(csm),mask) - c2r(atb)
                    res = Emat_xyt(grad,True,c2r(csm),mask)
                    res_new = torch.autograd.grad(x0_t, xt, grad_outputs=res, create_graph=False)[0]
                x0_t = x0_t - lr*res_new

                x0_preds.append(x0_t.to('cpu'))
                y_hat_t = c2r(ifft2c_2d(gmask_i*fft2c_2d(r2c(x0_t))))
                eps_t = (y_hat_t - xt)/gt**2
                noise = torch.randn_like(x0_t)
                z_t = xt + (gt**2 -gt_next**2)*eps_t + (gt**2 -gt_next**2).sqrt()*noise
                y_hat_t_1 = c2r(ifft2c_2d(gmask_next*fft2c_2d(r2c(x0_t))))
                xt_next = z_t+(y_hat_t_1-y_hat_t)   
                xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def generalized_ddpm_steps(x, atb, csm, mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
