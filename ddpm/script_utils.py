import argparse

import torch
import torchvision
import torch.nn.functional as F


from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)
    

def rgb_copy(rgb):
    r = torch.cat((rgb[:, [0], :, :], rgb[:, [0], :, :]), 1)
    r = torch.cat((r, r), 1)
    g = torch.cat((rgb[:, [1], :, :], rgb[:, [1], :, :]), 1)
    g = torch.cat((g, g), 1)
    b = torch.cat((rgb[:, [2], :, :], rgb[:, [2], :, :]), 1)
    b = torch.cat((b, b), 1)
    bg = torch.cat((b, g), 1)
    bgr = torch.cat((bg, r), 1)
    # print(bgr.shape)
    return bgr

def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l1",
        use_labels=False,
        base_channels=128,
        channel_mults=(1, 2, 2, 2, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(3,),
        ema_decay=0.9999,
        ema_update_rate=1,
        input_channels=3,
        output_channels=12,
        input_size=512,
        use_icmg=True,
        icmg_scale=2.5, # stage 2 needs lower scale
        pdt_lower_bound=0.96
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        # "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=args.output_channels,
        input_channels=args.input_channels,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else None,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (args.input_size, args.input_size), args.output_channels, None,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
        use_icmg=args.use_icmg,
        pdt_lower_bound=args.pdt_lower_bound

    )

    return diffusion


