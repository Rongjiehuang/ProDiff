import torch
from torch import nn
import utils
from functools import partial
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff import GaussianDiffusion
from usr.diff.net import DiffNet
from tasks.tts.fs2 import FastSpeech2Task
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from utils.pitch_utils import denorm_f0
from tasks.tts.fs2_utils import FastSpeechDataset
DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class ProDiff_Task(FastSpeech2Task):
    def __init__(self):
        super(ProDiff_Task, self).__init__()
        self.dataset_cls = FastSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.num_params(self.model, print_out=True, model_name="Generator: student")
        utils.num_params(self.teacher, print_out=True, model_name="Generator: teacher")
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
        return self.model

    def build_tts_model(self):
        mel_bins = hparams['audio_num_mel_bins']
        checkpoint = torch.load(hparams['teacher_ckpt'], map_location='cpu')["state_dict"]['model']
        teacher_timesteps = int(checkpoint['timesteps'].item())
        teacher_timescales = int(checkpoint['timescale'].item())
        student_timesteps = teacher_timesteps // 2
        student_timescales = teacher_timescales * 2

        self.teacher = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            loss_type=hparams['diff_loss_type'],
            timesteps=teacher_timesteps, time_scale=teacher_timescales,
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=student_timesteps, time_scale=student_timescales,
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        utils.load_ckpt(self.teacher, hparams['teacher_ckpt'], 'model', strict=False)
        utils.load_ckpt(self.model, hparams['teacher_ckpt'], 'model', strict=False)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.model.num_timesteps = student_timesteps
        self.model.time_scale = student_timescales
        self.model.register_buffer('timesteps', to_torch(student_timesteps))      # beta
        self.model.register_buffer('timescale', to_torch(student_timescales))      # beta

        for k, v in self.model.fs2.named_parameters():
            if not 'denoise_fn' in k:
                v.requires_grad = False

        for param in self.teacher.parameters():
            param.requires_grad = False


    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        # mel2ph = sample['mel2ph'] if hparams['use_gt_dur'] else None # [B, T_s]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(txt_tokens, self.teacher, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        losses['l1'] = output['mel_out']
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            # model_out = self.model(
            #     txt_tokens, spk_embed=spk_embed, mel2ph=None, f0=None, uv=None, energy=None, ref_mels=None, inference=True)
            # self.plot_mel(batch_idx, model_out['mel_out'], model_out['fs2_mel'], name=f'diffspeech_vs_fs2_{batch_idx}')
            model_out = self.model(
                txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=None, infer=True)
            gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=model_out.get('f0_denorm'))
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
        return outputs


    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

