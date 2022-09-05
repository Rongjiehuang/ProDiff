import glob
import re
import librosa
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from modules.FastDiff.module.FastDiff_model import FastDiff as FastDiff_model
from utils.hparams import hparams
from modules.parallel_wavegan.utils import read_hdf5
from vocoders.base_vocoder import BaseVocoder, register_vocoder
import numpy as np
from modules.FastDiff.module.util import theta_timestep_loss, compute_hyperparams_given_schedule, sampling_given_noise_schedule

def load_fastdiff_model(config_path, checkpoint_path):
    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = FastDiff_model(audio_channels=config['audio_channels'],
                 inner_channels=config['inner_channels'],
                 cond_channels=config['cond_channels'],
                 upsample_ratios=config['upsample_ratios'],
                 lvc_layers_each_block=config['lvc_layers_each_block'],
                 lvc_kernel_size=config['lvc_kernel_size'],
                 kpnet_hidden_channels=config['kpnet_hidden_channels'],
                 kpnet_conv_size=config['kpnet_conv_size'],
                 dropout=config['dropout'],
                 diffusion_step_embed_dim_in=config['diffusion_step_embed_dim_in'],
                 diffusion_step_embed_dim_mid=config['diffusion_step_embed_dim_mid'],
                 diffusion_step_embed_dim_out=config['diffusion_step_embed_dim_out'],
                 use_weight_norm=config['use_weight_norm'])

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"]["model"], strict=True)

    # Init hyperparameters by linear schedule
    noise_schedule = torch.linspace(float(config["beta_0"]), float(config["beta_T"]), int(config["T"])).cuda()
    diffusion_hyperparams = compute_hyperparams_given_schedule(noise_schedule)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key in ["beta", "alpha", "sigma"]:
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
    diffusion_hyperparams = diffusion_hyperparams


    if config['noise_schedule'] != '':
        noise_schedule = config['noise_schedule']
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

        # Below are schedules derived by Noise Predictor
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

    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| FastDiff device: {device}.")
    return model, diffusion_hyperparams, noise_schedule, config, device


@register_vocoder
class FastDiff(BaseVocoder):
    def __init__(self):
        if hparams['vocoder_ckpt'] == '':  # load LJSpeech FastDiff pretrained model
            base_dir = 'checkpoint/FastDiff'
            config_path = f'{base_dir}/config.yaml'
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            print('| load FastDiff: ', ckpt)
            self.scaler = None
            self.model, self.dh, self.noise_schedule, self.config, self.device = load_fastdiff_model(
                config_path=config_path,
                checkpoint_path=ckpt,
            )
        else:
            base_dir = hparams['vocoder_ckpt']
            print(base_dir)
            config_path = f'{base_dir}/config.yaml'
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            print('| load FastDiff: ', ckpt)
            self.scaler = None
            self.model, self.dh, self.noise_schedule, self.config, self.device = load_fastdiff_model(
                config_path=config_path,
                checkpoint_path=ckpt,
            )

    def spec2wav(self, mel, **kwargs):
        # start generation
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            audio_length = c.shape[-1] * hparams["hop_size"]
            y = sampling_given_noise_schedule(
                self.model, (1, 1, audio_length), self.dh, self.noise_schedule, condition=c, ddim=False, return_sequence=False)
        wav_out = y.cpu().numpy()
        return wav_out

    @staticmethod
    def wav2spec(wav_fn, return_linear=False):
        from data_gen.tts.data_gen_utils import process_utterance
        res = process_utterance(
            wav_fn, fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=return_linear, vocoder='fastdiff', eps=float(hparams.get('wav2spec_eps', 1e-10)))
        if return_linear:
            return res[0], res[1].T, res[2].T  # [T, 80], [T, n_fft]
        else:
            return res[0], res[1].T

    @staticmethod
    def wav2mfcc(wav_fn):
        fft_size = hparams['fft_size']
        hop_size = hparams['hop_size']
        win_length = hparams['win_size']
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                    n_fft=fft_size, hop_length=hop_size,
                                    win_length=win_length, pad_mode="constant", power=1.0)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta]).T
        return mfcc
