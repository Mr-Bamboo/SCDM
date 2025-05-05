import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract

from einops import rearrange, repeat



# use_icmg = True

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
            self,
            model,
            img_size,
            img_channels,
            num_classes,
            betas,
            loss_type="l2",
            ema_decay=0.9999,
            ema_start=5000,
            ema_update_rate=1,
            use_icmg=True,
            icmg_scale=2.5,
            pdt_lower_bound=0.96,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)
        self.use_icmg = use_icmg
        self.icmg_scale = icmg_scale
        self.pdt_lower_bound = pdt_lower_bound

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, batch_size, x, rgb, t, y, use_ema=True):
        if use_ema:
            dynamic_array = np.linspace(self.pdt_lower_bound, 1.00, 1000)[::-1]
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                dynamic_array[t.data.item()],
                dim=-1
            )
            s.clamp_(min=1.)
            s = right_pad_dims_to(x, s)
            for i in range(batch_size):
                # print(s[i])
                # x[i] = x[i].clamp((-s[i]).data.item(), s[i].data.item()) / s[i].data.item()
                x[i] = x[i].clamp((-s[i]).data.item(), s[i].data.item())
            if self.use_icmg:
                no_cond = self.ema_model(x, rgb*0, t, y)
                cond = self.ema_model(x, rgb, t, y)
                mu = (x - extract(self.remove_noise_coeff, t, x.shape) * (no_cond + (self.icmg_scale + 1.0) * (cond - no_cond))) * extract(
                    self.reciprocal_sqrt_alphas, t, x.shape)
            else:
                mu = (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, rgb, t, y)) * extract(
                    self.reciprocal_sqrt_alphas, t, x.shape)
            
            return mu
        else:
            dynamic_array = np.linspace(self.pdt_lower_bound, 1.00, 1000)[::-1]
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                dynamic_array[t.data.item()],
                dim=-1
            )
            s.clamp_(min=1.)
            s = right_pad_dims_to(x, s)
            for i in range(batch_size):
                x[i] = x[i].clamp((-s[i]).data.item(), s[i].data.item())
            if self.use_icmg:
                no_cond = self.model(x, rgb * 0, t, y)
                cond = self.model(x, rgb, t, y)
                mu = (x - extract(self.remove_noise_coeff, t, x.shape) * (no_cond + (self.icmg_scale + 1.0) * (cond - no_cond))) * extract(self.reciprocal_sqrt_alphas, t, x.shape)
            else:
                mu = (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, rgb, t, y)) * extract(self.reciprocal_sqrt_alphas, t, x.shape)
            return mu

    @torch.no_grad()
    def sample(self, batch_size, device, rgb, y=None, use_ema=True):
        # print(use_ema)
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(batch_size, x, rgb, t_batch, y, use_ema)


            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, rgb, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, rgb, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        # x_0 = 1/sqrt(at-)*x_t-sqrt(1/at- - 1)*epsilon


    def get_losses(self, x, rgb, t, y):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, rgb, t, y)
        
        estimated_x = self._predict_xstart_from_eps(perturbed_x, t, estimated_noise)

        # sam_loss = F.cosine_similarity(estimated_noise, noise, dim=1)
        sam_loss = F.cosine_similarity(estimated_x, x, dim=1)
        sam_loss = 1 - torch.mean(sam_loss)
        

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise) / 1.0
        else:
            loss = F.mse_loss(estimated_noise, noise)

        return loss, sam_loss

    def forward(self, x, rgb, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[1]:
            raise ValueError("image width does not match diffusion parameters")


        t = torch.randint(0, self.num_timesteps, (b,), device=device)   # 随机生成一个t

        if self.use_icmg == True:
          drop_rate = 0.1
          if drop_rate > 0.0:
              mask = (torch.rand([b, 1, 1, 1]) > drop_rate).float()
              mask = mask.to(device)
              rgb = rgb * mask
        
        return self.get_losses(x, rgb, t, y)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
