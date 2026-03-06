import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from typing import *
# from tab_transformer_pytorch import TabTransformer

#@title ⏳ Summary: please run this cell which contains the ```VPDiffusionFlowMatching``` class

class VPDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5

    def T(self, s: torch.Tensor) -> torch.Tensor:

        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:

        return torch.exp(-0.5 * self.T(t))

    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        return self.alpha(1. - t) * x_1

    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(1. - self.alpha(1. - t) ** 2)

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        num = torch.exp(-self.T(1. - t)) * x - torch.exp(-0.5 * self.T(1.-t))* x_1
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)


    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """ 
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = self.mu_t(t, x_1) + self.sigma_t(t, x_1) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0], x) - self.u_t(t, x, x_1)) ** 2)

#@title ⏳ Summary: please run this cell which contains the ```VEDiffusionFlowMatching``` class
class VEDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 1.
        self.eps = 1e-5


    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:

        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:

        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max/self.sigma_min))

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        return -(self.dsigma_dt(1. - t) / self.sigma_t(1. - t)) * (x - x_1)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """ 
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = x_1 + self.sigma_t(1. - t) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0], x) - self.u_t(t, x, x_1)) ** 2)

### The basic: optimal transport flow matching
class OTFlowMatching:

    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Conditional Flow
        """
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x_0)
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:,0], self.psi_t(x_0, x_1, t))
        d_psi = x_1 - (1 - self.sig_min) * x_0
        return torch.mean((v_psi - d_psi) ** 2)

#@title ⏳ Summary: please run this cell which contains the ```VEDiffusionFlowMatching``` class
class LogitFlowMatching:

    def __init__(self, sig_min: float = 1e-5) -> None:
        super().__init__()

        self.sig_min = sig_min
        self.alpha = 3.0 ## exp(mu)
        self.lamb = 1.0
        self.eps = 1e-5

    def dtprime_dt(self, t):
        """
        Compute derivative dt'/dt.
        """
        u = (1/(0.999 * t)) - 1. ## (1/ct - 1)^lambda
        b = u**self.lamb

        dB_dt = - self.lamb * (u ** (self.lamb - 1)) / (0.999 * (t**2))

        return -self.alpha / ((self.alpha + b) ** 2) * dB_dt

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        denum_add = (1 / (0.999 * t) - 1)**self.lamb
        t_prime = self.alpha / (self.alpha + denum_add)
        dtprime = self.dtprime_dt(t)
        ## a = t_prime, b = 1-t_prime, da = dtprime, db = -dtprime
        # return (dtprime - (-dtprime/(1-t_prime)) * t_prime) * x_1 + (-dtprime/(1-t_prime)) * x
        return (x_1 - x) * dtprime / (1-t_prime)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """ 
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        
        denum_add = (1 / (0.999 * t) - 1)**self.lamb
        t_prime = self.alpha / (self.alpha + denum_add)    

        x = t_prime * x_1 + (1. - t_prime) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0], x) - self.u_t(t, x, x_1)) ** 2)

class CosineFlowMatching:

    def __init__(self, sig_min: float = 1e-5) -> None:
        super().__init__()

        self.sig_min = sig_min
        self.alpha = 3.0
        self.sigma = 1.0
        self.eps = 1e-5

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        alpha, beta = torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)
        dalpha = torch.cos(0.5 * math.pi * t) * 0.5 * math.pi
        dbeta = -torch.sin(0.5 * math.pi * t) * 0.5 * math.pi
        return (dalpha - (dbeta/beta) * alpha) * x_1 + (dbeta/beta) * x

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """ 
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        
        t = t.clamp(max = 1 - 1e-5)
        x = torch.sin(0.5 * math.pi * t) * x_1 + torch.cos(0.5 * math.pi * t) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0], x) - self.u_t(t, x, x_1)) ** 2)


class SiLU(nn.Module):
        def forward(self, x):
                return x * torch.sigmoid(x)

class Net(nn.Module): ## followed from tabsyn paper
    def __init__(self, in_dim: int, n_frequencies:int) -> None:
        super().__init__()

        dim_t = 2 * n_frequencies
        ins = [dim_t, dim_t*2, dim_t*2]
        outs = [dim_t*2, dim_t*2, dim_t]
        
        self.n_frequencies = n_frequencies

        self.proj = nn.Linear(in_dim, dim_t)

        self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(in_d, out_d), nn.SiLU()) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(dim_t, in_dim))

        self.time_embed = nn.Sequential(
                nn.Linear(2 * n_frequencies, 2 * n_frequencies),
                nn.SiLU(),
                nn.Linear(2 * n_frequencies, 2 * n_frequencies)
        )

    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t[..., None]
        return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.time_encoder(t)
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        for l in self.layers:
            x = l(x)
        return self.top(x)

### the conditional vector fields baseline
class CondVF(nn.Module):
    def __init__(self, net: nn.Module, n_steps: int = 100, sig_min: float = 0.001) -> None:
        super().__init__()
        self.net = net
        self.atol=1e-06
        self.rtol=1e-05
        self.useodeint = False
        self.n_steps = n_steps
        self.sig_min = sig_min
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # print(t.size(), x.size())
        if self.useodeint: 
            t = t.expand(x.size(0))
            return self.net(t, x)
        else:     
            return self.net(t, x)
        
    def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t = t * torch.ones(len(x), device=x.device)
            return self(t, x)
    
    # uses from torchdiffeq import odeint_adjoint as odeint
    def decode_t0_t1(self, x_0, t0, t1, method = 'euler'):
        self.useodeint = True
        # x_1 = odeint(self, y0=x_0, t=torch.tensor([t0, t1],device=x_0.device), 
        #                             method='euler', options={'step_size': (1/self.n_steps)})[-1]
        # x_1 = odeint(self, y0=x_0, t=torch.tensor([t0, t1],device=x_0.device), 
        #                             method='dopri5', rtol=1e-5, atol=1e-5)[-1]
        
        # Default: step_size for fixed-step, rtol/atol for adaptive
        if method in ['euler', 'rk4', 'midpoint']:
            options = {'step_size': 1 / self.n_steps}
        # Untuk solver adaptive (dopri5, adams, dll) bisa set rtol/atol kalau perlu:
        elif method in ['dopri5', 'adaptive_heun', 'explicit_adams', 'implicit_adams']:
            options = {}
            # 'rtol': 1e-5, 'atol': 1e-6

        x_1 = odeint(
            self, 
            y0=x_0, 
            t=torch.tensor([t0, t1], device=x_0.device),
            method=method, 
            options=options,
            rtol=1e-5, atol=1e-6
        )[-1]

        self.useodeint = False
        return x_1

    
    def encode(self, x_1: torch.Tensor) -> torch.Tensor:
        self.useodeint = True
        x_0 = odeint(self, y0=x_1, t=torch.tensor([1., 0.],device=x_1.device), 
                                    method='euler', options={'step_size': (1/self.n_steps)})[-1]
        self.useodeint = False
        return x_0

    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        self.useodeint = True
        x_1 = odeint(self, y0=x_0, t=torch.tensor([0., 1.],device=x_0.device), 
                                    method='euler', options={'step_size': (1/self.n_steps)})[-1]
        
        self.useodeint = False
        return x_1
