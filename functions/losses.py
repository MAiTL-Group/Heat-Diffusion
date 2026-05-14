import torch
import numpy as np
from utils import *


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          g: torch.Tensor,
                          type="VP", gmask=None, mask=None, csm=None, keepdim=False):
    if type == "VP":
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        output = model(x, t.float())
        res = e - output
        
    elif type == "Heat": # TODO: 
        device = x0.device
        z = e
        indices = t.cpu().detach().numpy()
        gmaskt = np.array([gmask[i] for i in indices])
        gmaskt = np.expand_dims(gmaskt, axis=1)
        gmaskt = torch.from_numpy(gmaskt).to(device)
        std = g.index_select(dim=0, index=t).view(-1, 1, 1, 1)
        x = c2r(ifft2c_2d(fft2c_2d(r2c(x0))*gmaskt)) + std * z 
        output = model(x, t.float()) 
        e = (x - x0) 
        res = e - output
        res = c2r(ifft2c_2d(fft2c_2d(r2c(res))*gmaskt))
            

    elif type == "VE":
        std = g.index_select(dim=0, index=t).view(-1, 1, 1, 1)
        x = x0 + std * e 
        output = model(x, t.float())
        res = e - output


    elif type == "VE-Null":
        std = g.index_select(dim=0, index=t).view(-1, 1, 1, 1)
        e = to_null_space(e,mask,csm)
        x = x0 + std * e 
        output = model(x, t.float())
        res = e - output
    elif type == "HFS":
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        output = model(x, t.float())
        e = c2r(ifft2c_2d((1-mask)*fft2c_2d(r2c(e))))
        output = c2r(ifft2c_2d((1-mask)*fft2c_2d(r2c(output))))
        res = e - output

    if keepdim:
        return (res).square().sum(dim=(1, 2, 3))
    else:
        return (res).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
