import os

import torch

from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
import numpy as np
from modules.FastDiff.module.util import compute_hyperparams_given_schedule, sampling_given_noise_schedule

import os

import torch

from modules.FastDiff.module.FastDiff_model import FastDiff
from utils.ckpt_utils import load_ckpt
from utils.hparams import set_hparams


class BaseTTSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.ph_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder, self.diffusion_hyperparams, self.noise_schedule = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = FastDiff(audio_channels=config['audio_channels'],
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
        load_ckpt(vocoder, base_dir, 'model')

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
                reverse_step = int(self.hparams.get('N'))
            except:
                print(
                    'Please specify $N (the number of revere iterations) in config file. Now denoise with 4 iterations.')
                reverse_step = 4
            if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000).cuda()
            elif reverse_step == 200:
                noise_schedule = torch.linspace(0.0001, 0.02, 200).cuda()

            # Below are schedules derived by Noise Predictor.
            # We will release codes of noise predictor training process & noise scheduling process soon. Please Stay Tuned!
            elif reverse_step == 8:
                noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                  0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593,
                                  0.5]
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

        return vocoder, diffusion_hyperparams, noise_schedule

    def run_vocoder(self, c):
        c = c.transpose(2, 1)
        audio_length = c.shape[-1] * self.hparams["hop_size"]
        y = sampling_given_noise_schedule(
            self.vocoder, (1, 1, audio_length), self.diffusion_hyperparams, self.noise_schedule, condition=c, ddim=False, return_sequence=False)
        return y

    def preprocess_input(self, inp):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw, preprocess_args)
        ph_token = self.ph_encoder.encode(ph)
        spk_id = self.spk_map[spk_name]
        item = {'item_name': item_name, 'text': txt, 'ph': ph, 'spk_id': spk_id, 'ph_token': ph_token}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls):
        from utils.hparams import set_hparams
        from utils.hparams import hparams as hp
        from utils.audio import save_wav

        set_hparams()
        inp = {
            'text': hp['text']
        }
        infer_ins = cls(hp)
        out = infer_ins.infer_once(inp)
        os.makedirs('infer_out', exist_ok=True)
        save_wav(out, f'infer_out/{hp["text"]}.wav', hp['audio_sample_rate'])
