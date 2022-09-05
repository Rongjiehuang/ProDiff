import os

import torch
import utils
from modules.FastDiff.module.FastDiff_model import FastDiff
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils import audio
from utils.hparams import hparams
from modules.FastDiff.module.util import theta_timestep_loss, compute_hyperparams_given_schedule, sampling_given_noise_schedule


class FastDiffTask(VocoderBaseTask):
    def __init__(self):
        super(FastDiffTask, self).__init__()

    def build_model(self):
        self.model = FastDiff(audio_channels=hparams['audio_channels'],
                 inner_channels=hparams['inner_channels'],
                 cond_channels=hparams['cond_channels'],
                 upsample_ratios=hparams['upsample_ratios'],
                 lvc_layers_each_block=hparams['lvc_layers_each_block'],
                 lvc_kernel_size=hparams['lvc_kernel_size'],
                 kpnet_hidden_channels=hparams['kpnet_hidden_channels'],
                 kpnet_conv_size=hparams['kpnet_conv_size'],
                 dropout=hparams['dropout'],
                 diffusion_step_embed_dim_in=hparams['diffusion_step_embed_dim_in'],
                 diffusion_step_embed_dim_mid=hparams['diffusion_step_embed_dim_mid'],
                 diffusion_step_embed_dim_out=hparams['diffusion_step_embed_dim_out'],
                 use_weight_norm=hparams['use_weight_norm'])
        utils.print_arch(self.model)

        # Init hyperparameters by linear schedule
        noise_schedule = torch.linspace(float(hparams["beta_0"]), float(hparams["beta_T"]), int(hparams["T"])).cuda()
        diffusion_hyperparams = compute_hyperparams_given_schedule(noise_schedule)

        # map diffusion hyperparameters to gpu
        for key in diffusion_hyperparams:
            if key in ["beta", "alpha", "sigma"]:
                diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
        self.diffusion_hyperparams = diffusion_hyperparams

        return self.model

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mels = sample['mels']
        y = sample['wavs']
        X = (mels, y)
        loss = theta_timestep_loss(self.model, X, self.diffusion_hyperparams)
        return loss, {'loss': loss}


    def validation_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        X = (mels, y)
        loss = theta_timestep_loss(self.model, X, self.diffusion_hyperparams)
        return loss, {'loss': loss}


    def test_step(self, sample, batch_idx):
        mels = sample['mels']
        y = sample['wavs']
        loss_output = {}

        if hparams['noise_schedule'] != '':
            noise_schedule = hparams['noise_schedule']
            if isinstance(noise_schedule, list):
                noise_schedule = torch.FloatTensor(noise_schedule).cuda()
        else:
            # Select Schedule
            try:
                reverse_step = int(hparams.get('N'))
            except:
                print('Please specify $N (the number of revere iterations) in config file. Now denoise with 4 iterations.')
                reverse_step = 4
            if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000).cuda()
            elif reverse_step == 200:
                noise_schedule = torch.linspace(0.0001, 0.02, 200).cuda()

            # Below are schedules derived by Noise Predictor.
            # We will release codes of noise predictor training process & noise scheduling process soon. Please Stay Tuned!
            elif reverse_step == 8:
                noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                 0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5]
            elif reverse_step == 6:
                noise_schedule = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                  0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
            elif reverse_step == 4:
                noise_schedule = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
            elif reverse_step == 3:
                noise_schedule = [9.0000e-05, 9.0000e-03, 6.0000e-01]
            else:
                raise NotImplementedError

        if isinstance(noise_schedule, list):
            noise_schedule = torch.FloatTensor(noise_schedule).cuda()

        audio_length = mels.shape[-1] * hparams["hop_size"]
        # generate using DDPM reverse process

        y_ = sampling_given_noise_schedule(
            self.model, (1, 1, audio_length), self.diffusion_hyperparams, noise_schedule,
            condition=mels, ddim=False, return_sequence=False)
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)

        if len(y) == 0:
            # Inference from mel
            for idx, (wav_pred, item_name) in enumerate(zip(y_, sample["item_name"])):
                wav_pred = wav_pred / wav_pred.abs().max()
                audio.save_wav(wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav',
                               hparams['audio_sample_rate'])
        else:
            for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
                wav_gt = wav_gt / wav_gt.abs().max()
                wav_pred = wav_pred / wav_pred.abs().max()
                audio.save_wav(wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_gt.wav', hparams['audio_sample_rate'])
                audio.save_wav(wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav', hparams['audio_sample_rate'])
        return loss_output
        
    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(hparams['lr']), weight_decay=float(hparams['weight_decay']))
        return optimizer

    def compute_rtf(self, sample, generation_time, sample_rate=22050):
        """
        Computes RTF for a given sample.
        """
        total_length = sample.shape[-1]
        return float(generation_time * sample_rate / total_length)