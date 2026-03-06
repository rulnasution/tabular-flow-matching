import torch
import torch.nn as nn
from torch import Tensor
from typing import *
import math

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
    ]) # nn.LeakyReLU()
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

class Residual(nn.Module):
    """Residual layer"""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o) 
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)  


class ResNet(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, h_dims: List[int], n_frequencies:int) -> None:
    super().__init__()

    self.n_frequencies = n_frequencies
    dim = in_dim + 2 * n_frequencies
    seq = []
    for item in h_dims:
        seq += [Residual(dim, item)]
        dim += item
    
    self.layers = nn.ModuleList(seq)
    self.top = nn.Sequential(nn.Linear(dim, out_dim))

  def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
    freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
    t = freq * t[..., None]
    return torch.cat((t.cos(), t.sin()), dim=-1)

  def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = self.time_encoder(t)
    x = torch.cat((x, t), dim=-1)

    for l in self.layers:
      x = l(x)
    return self.top(x)

#### network for time function ####

class OT_t(nn.Module):
    '''
    just regular t to ensure it works in gaussian setting
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()

    def atx(self, t: torch.Tensor) -> torch.Tensor:
        # return torch.ones_like(t.view(-1,1))
        # return 1 / (1. - (0.999 * t))
        return (1 / (1. - (0.999 * t.view(-1,1))))

        # return t.view(-1,1)
        
    def forward(self, t : Tensor) -> Tensor:
        return t.view(-1,1), 1. - (0.999 * t.view(-1,1))

class VPDiffusion_t(nn.Module):
    '''
    Variance preserving diffusion field
    '''
    def __init__(
        self
    ) -> None:
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
    
    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - self.alpha(1. - t) ** 2)

    def atx(self, t: torch.Tensor) -> torch.Tensor:
        num = - torch.exp(-0.5 * self.T(1.-t)) ## alpha (t)
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num/denum)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(1. - t), torch.sqrt(1. - self.alpha(1. - t) ** 2)

class VEDiffusion_t(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 2.
        self.eps = 1e-5


    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
    
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
    
        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max/self.sigma_min))
    
    def atx(self, t: torch.Tensor) -> torch.Tensor:
        return (self.dsigma_dt(1. - t) / self.sigma_t(1. - t))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t), self.sigma_t(1. - t)
    
class LogitNormal_t(nn.Module):
    '''
    Logit normal distribution-based time function
    See FLUX.1 paper for details: https://arxiv.org/pdf/2506.15742
    I put 0.999 for stability purposes.
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()
        self.alpha = 3.0
        self.sigma = 1.0

    def atx(self, t):
        t = t.clamp(min=1e-7)
        
        denum_add = (1 / (0.999 * t.view(-1,1)) - 1)**self.sigma
        t_prime = self.alpha / (self.alpha + denum_add) 

        num = self.sigma * self.alpha * (1/(0.999 * t.view(-1,1)) - 1)**(self.sigma - 1)
        den = 0.999 * t**2 * (self.alpha + (1/(0.999 * t.view(-1,1)) - 1)**self.sigma)**2
        # num = self.sigma * self.alpha * (1/(0.999 * t.view(-1,1)) - 1)**(self.sigma - 1)
        t_prime_dot = num/den

        return t_prime_dot / (1 - t_prime)
        
    def forward(self, t : Tensor) -> Tensor:
        t = t.clamp(min=1e-7)
        denum_add = (1 / (0.999 * t.view(-1,1)) - 1)**self.sigma
        # denum_add = (1 / t.view(-1,1) - 1)**self.sigma
        t_prime = self.alpha / (self.alpha + denum_add)  
        return t_prime, 1. - t_prime

class Cosine_t(nn.Module):
    '''
    cosine time function
    See stochastic interpolant paper for details: https://arxiv.org/pdf/2209.15571
    '''
    def __init__(
        self 
    ) -> None:
        super().__init__()

    def atx(self, t) -> Tensor:
        t = t.clamp(max=1 - 1e-5)
        alpha, beta = torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)
        dalpha = torch.cos(0.5 * math.pi * t) * 0.5 * math.pi
        dbeta = -torch.sin(0.5 * math.pi * t) * 0.5 * math.pi
        # (alpha, beta), (dalpha, dbeta) = t_dir(self, t)
        return dalpha - (dbeta/beta) * alpha
        
    def forward(self, t : Tensor) -> Tensor:
        
        return torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)