import torch.nn as nn
import torch
import logging
from modules.FastDiff.module.modules import DiffusionDBlock, TimeAware_LVCBlock
from modules.FastDiff.module.util import calc_diffusion_step_embedding

def swish(x):
    return x * torch.sigmoid(x)

class FastDiff(nn.Module):
    """FastDiff module."""

    def __init__(self,
                 audio_channels=1,
                 inner_channels=32,
                 cond_channels=80,
                 upsample_ratios=[8, 8, 4],
                 lvc_layers_each_block=4,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 dropout=0.0,
                 diffusion_step_embed_dim_in=128,
                 diffusion_step_embed_dim_mid=512,
                 diffusion_step_embed_dim_out=512,
                 use_weight_norm=True):
        super().__init__()

        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.audio_channels = audio_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = len(upsample_ratios)
        self.first_audio_conv = nn.Conv1d(1, inner_channels,
                                    kernel_size=7, padding=(7 - 1) // 2,
                                    dilation=1, bias=True)

        # define residual blocks
        self.lvc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        # the layer-specific fc for noise scale embedding
        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_ratios[n]
            lvcb = TimeAware_LVCBlock(
                in_channels=inner_channels,
                cond_channels=cond_channels,
                upsample_ratio=upsample_ratios[n],
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
                noise_scale_embed_dim_out=diffusion_step_embed_dim_out
            )
            self.lvc_blocks += [lvcb]
            self.downsample.append(DiffusionDBlock(inner_channels, inner_channels, upsample_ratios[self.lvc_block_nums-n-1]))


        # define output layers
        self.final_conv = nn.Sequential(nn.Conv1d(inner_channels, audio_channels, kernel_size=7, padding=(7 - 1) // 2,
                                        dilation=1, bias=True))

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, data):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        audio, c, diffusion_steps = data

        # embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        audio = self.first_audio_conv(audio)
        downsample = []
        for down_layer in self.downsample:
            downsample.append(audio)
            audio = down_layer(audio)

        x = audio
        for n, audio_down in enumerate(reversed(downsample)):
            x = self.lvc_blocks[n]((x, audio_down, c, diffusion_step_embed))

        # apply final layers
        x = self.final_conv(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

