# ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech

#### Rongjie Huang, Zhou Zhao, Huadai Liu, Jinglin Liu, Chenye Cui, Yi Ren

PyTorch Implementation of [ProDiff (ACM Multimedia'22)](https://arxiv.org/abs/2207.06389): a conditional diffusion probabilistic model capable of generating high fidelity speech efficiently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2207.06389)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/ProDiff?style=social)](https://github.com/Rongjiehuang/ProDiff)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/ProDiff)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/Rongjiehuang/ProDiff)

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://prodiff.github.io/) for audio samples.

## News
- April, 2022: Our previous work **[FastDiff](https://arxiv.org/abs/2204.09934) (IJCAI 2022)** released in [Github](https://github.com/Rongjiehuang/FastDiff). 
- September, 2022: **[ProDiff](https://arxiv.org/abs/2207.06389) (ACM Multimedia 2022)** released in Github.

## Key Features
- **Extremely-Fast** diffusion text-to-speech synthesis pipeline for potential **industrial deployment**.
- **Tutorial and code base** for speech diffusion models.
- More **supported diffusion mechanism** (e.g., guided diffusion) will be available.

## Quick Started
We provide an example of how you can generate high-fidelity samples using ProDiff.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

### Support Datasets and Pretrained Models

Simply run following command to download the weights
```python
  from huggingface_hub import snapshot_download 
  downloaded_path = snapshot_download(repo_id="Rongjiehuang/ProDiff")
```

and move the downloaded checkpoints to `checkpoints/$Model/model_ckpt_steps_*.ckpt`
```bash
   mv ${downloaded_path}/checkpoints/  checkpoints/
```

Details of each folder are as in follows:

| Model             | Dataset     | Config                                          | 
|-------------------|-------------|-------------------------------------------------|
| ProDiff Teacher   | LJSpeech    | `modules/ProDiff/config/prodiff_teacher.yaml`   | 
| ProDiff           | LJSpeech    | `modules/ProDiff/config/prodiff.yaml`           | 


More supported datasets are coming soon.



### Dependencies
See requirements in `requirement.txt`:
- [pytorch](https://github.com/pytorch/pytorch)
- [librosa](https://github.com/librosa/librosa)
- [NATSpeech](https://github.com/NATSpeech/NATSpeech)

### Multi-GPU
By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Extremely-Fast Text-to-Speech with diffusion probabilistic models 

Here we provide a speech synthesis pipeline using diffusion probabilistic models: ProDiff (acoustic model) + FastDiff (neural vocoder). [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/Rongjiehuang/ProDiff)

1. Prepare acoustic model (ProDiff or ProDiff Teacher): Download LJSpeech checkpoint and put it in `checkpoints/ProDiff` or `checkpoints/ProDiff_Teacher`
2. Prepare neural vocoder (FastDiff): Download LJSpeech checkpoint and put it in `checkpoints/FastDiff`

3. Specify the input `$text`, and set `N` for reverse sampling in neural vocoder, which is a trade off between quality and speed. 
4. Run the following command for extreme fast speed `(2-iter ProDiff + 4-iter FastDiff)`:
```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/ProDiff.py --config modules/ProDiff/config/prodiff.yaml --exp_name ProDiff --hparams="N=4,text='$txt'" --reset
```
Generated wav files are saved in `infer_out` by default.<br>
Note: For better quality, it's recommended to finetune the FastDiff neural vocoder [here](https://github.com/Rongjiehuang/FastDiff).

5. Enjoy speed-quality trade-off:  `(4-iter ProDiff Teacher + 6-iter FastDiff)`:
```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/ProDiff_teacher.py --config modules/ProDiff/config/prodiff_teacher.yaml --exp_name ProDiff_Teacher --hparams="N=6,text='$txt'" --reset
```

# Train your own model

### Data Preparation and Configuraion ##
1. Set `raw_data_dir`, `processed_data_dir`, `binary_data_dir` in the config file
2. Download dataset to `raw_data_dir`. Note: the dataset structure needs to follow `egs/datasets/audio/*/pre_align.py`, or you could rewrite `pre_align.py` according to your dataset.
3. Preprocess Dataset 
```bash
# Preprocess step: unify the file structure.
python data_gen/tts/bin/pre_align.py --config $path/to/config
# Align step: MFA alignment.
python data_gen/tts/runs/train_mfa_align.py --config $CONFIG_NAME
# Binarization step: Binarize data for fast IO.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config $path/to/config
```

You could also build a dataset via [NATSpeech](https://github.com/NATSpeech/NATSpeech), which shares a common MFA data-processing procedure.
We also provide our processed LJSpeech dataset [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/Eo7r83WZPK1GmlwvFhhIKeQBABZpYW3ec9c8WZoUV5HhbA?e=9QoWnf).


### Training Teacher of ProDiff 
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/ProDiff/config/prodiff_teacher.yaml  --exp_name ProDiff_Teacher --reset
```

### Training ProDiff
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/ProDiff/config/prodiff.yaml  --exp_name ProDiff --reset
```

### Inference using ProDiff Teacher

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/ProDiff/config/prodiff_teacher.yaml  --exp_name ProDiff_Teacher --infer
```

### Inference using ProDiff

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/ProDiff/config/prodiff.yaml  --exp_name ProDiff --infer
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[FastDiff](https://github.com/Rongjiehuang/FastDiff),
[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger),
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@inproceedings{huang2022prodiff,
  title={ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech},
  author={Huang, Rongjie and Zhao, Zhou and Liu, Huadai and Liu, Jinglin and Cui, Chenye and Ren, Yi},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}

@article{huang2022fastdiff,
  title={FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis},
  author={Huang, Rongjie and Lam, Max WY and Wang, Jun and Su, Dan and Yu, Dong and Ren, Yi and Zhao, Zhou},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year={2022}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
