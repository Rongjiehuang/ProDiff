import math
import random
from functools import partial
from usr.diff.shallow_diffusion_tts import *
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange

from modules.fastspeech.fs2 import FastSpeech2
from utils.hparams import hparams


class GaussianDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn, teacher_steps=4,
                 timesteps=4, time_scale=1, loss_type='l1', betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.fs2 = FastSpeech2(phone_encoder, out_dims)
        self.fs2.decoder = None
        self.mel_bins = out_dims

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = get_noise_schedule_list(
                schedule_mode=hparams['schedule_type'],
                timesteps=teacher_steps + 1,
                min_beta=0.1,
                max_beta=40,
                s=0.008,
            )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.time_scale = time_scale
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))      # beta
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # alphacum_t
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev)) # alphacum_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, spk_emb=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    def sample_q(self, x_0, ts, epsilon=None):
        """
        Sample from q(x_t | x_0) for a batch of x_0.
        """
        alpha, sigma = self.get_schedule(x_0, ts)
        return alpha * x_0 + sigma * epsilon

    @torch.no_grad()
    def p_sample_ddim(self, x_t, t, cond):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond)
        alpha, sigma = self.get_schedule(x_t, t)
        eps = (x_t - x_0_pred * alpha) / sigma
        return self.sample_q(x_0_pred, t-self.time_scale, eps)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_schedule(self, x_t, t):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape), extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def diffuse_fn(self, x_start, t, noise=None):
        x_start = self.norm_spec(x_start)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        zero_idx = t < 0 # for items where t is -1
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out

    def forward(self, txt_tokens, teacher_fn=None, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=True, infer=infer)
        cond = ret['decoder_inp'].transpose(1, 2)
        if not infer:
            with torch.no_grad():
                t = self.time_scale * torch.randint(1, self.num_timesteps+1, (b,), device=device).long() # [2, 4]
                nonpadding = (mel2ph != 0).float().unsqueeze(1).unsqueeze(1)
                noise = default(None, lambda: torch.randn_like(ref_mels.transpose(1, 2)[:, None, :, :]))

                # Diffusion
                x_t = self.diffuse_fn(ref_mels, t, noise) * nonpadding

                # 2 steps of DDIM
                x0_pred = teacher_fn.denoise_fn(x_t, t, cond) * nonpadding  # p(x_0|x_t,t) correct
                alpha, sigma = self.get_schedule(x_t, t)
                alpha_pre, sigma_pre = self.get_schedule(x_t, t - self.time_scale // 2)
                alpha_pre_pre, sigma_pre_pre = self.get_schedule(x_t, t - self.time_scale)
                x_t_pre = alpha_pre * x0_pred + sigma_pre / sigma * (x_t - alpha * x0_pred)  # correct
                x0_pred1 = teacher_fn.denoise_fn(x_t_pre, t - self.time_scale // 2, cond) * nonpadding  # correct
                x_t_pre_pre = alpha_pre_pre * x0_pred1 + sigma_pre_pre / sigma_pre * (
                            x_t_pre - alpha_pre * x0_pred1)  # correct
                x_target = (x_t_pre_pre - (sigma_pre_pre / sigma) * x_t) / (alpha_pre_pre - sigma_pre_pre / sigma * alpha) * nonpadding

            x_pred = self.denoise_fn(x_t, t - self.time_scale, cond) * nonpadding  # student [0, 1]: 8 steps correct
            x_t_prev = self.diffuse_fn(ref_mels, t - self.time_scale - 1, noise) * nonpadding  # teacher [-1, 1]
            x_t_prev_pred = self.q_posterior_sample(x_pred, x_t, t - self.time_scale) * nonpadding # [-1, 1] p(x_t-1|x_t,x_0,t)

            if self.loss_type == 'l1':
                if nonpadding is not None:  # [B, T]
                    loss = ((x_pred - x_target).abs() * nonpadding).mean()  # [B, B, M, T].mean()
                else:
                    # print('are you sure w/o nonpadding?')
                    loss = (x_pred - x_target).abs().mean()

            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_pred, x_target)
            else:
                raise NotImplementedError()

            ret['mel_out'] = loss # [B, T, 80]
            ret['x_t'] = x_t[:, 0].transpose(1, 2)
            ret['x_t_prev'] = x_t_prev[:, 0].transpose(1, 2)
            ret['x_t_prev_pred'] = x_t_prev_pred[:, 0].transpose(1, 2)
            ret['t'] = t
        else:
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device)  # noise
            sample_steps = [self.time_scale * i for i in range(0, self.num_timesteps)]
            for i in tqdm(reversed(sample_steps), desc='ProDiff sample time step', total=len(sample_steps)):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)
            x = x[:, 0].transpose(1, 2)
            # p_sample: 0.1805
            ret['mel_out'] = self.denorm_spec(x)  # 去除norm
        return ret


    def norm_spec(self, x):
        return x

    def denorm_spec(self, x):
        return x

    def out2mel(self, x):
        return x