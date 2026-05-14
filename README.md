# Physics-Informed DeepMRI: k-Space Interpolation Meets Heat Diffusion

This repository contains the official PyTorch implementation of:

> **Physics-Informed DeepMRI: k-Space Interpolation Meets Heat Diffusion**
> Zhuo-Xu Cui, Congcong Liu, Xiaohong Fan, Chentao Cao, Jing Cheng, Qingyong Zhu, Yuanyuan Liu, Sen Jia, Yihang Zhou, Haifeng Wang, Yanjie Zhu, Jianping Zhang, Qiegen Liu, Dong Liang.
> *IEEE Transactions on Medical Imaging*, 2024.
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10683732/) &nbsp;&nbsp; [[arXiv:2308.15918]](https://arxiv.org/abs/2308.15918) &nbsp;&nbsp; [[Papers with Code]](https://paperswithcode.com/paper/physics-informed-deepmri-bridging-the-gap)

This is the maintained release at the MAiTL-Group. The original first-author release is available at [`ZhuoxuCui/Heat-Diffusion`](https://github.com/ZhuoxuCui/Heat-Diffusion).

---

## Overview

We model the attenuation of high-frequency information in k-space as a forward **heat diffusion** process, and formulate accelerated MRI reconstruction as the corresponding **reverse heat diffusion**. To make the reverse process tractable, we modify the heat equation to be consistent with magnetic-resonance **parallel-imaging physics**, and solve it with a **score-based generative model**. Experiments on public datasets show improvements over both traditional and deep-learning k-space interpolation methods, especially in high-frequency regions.

## Dependencies

Tested on Ubuntu 22.04.5 LTS with CUDA 12.6 and PyTorch 2.6.

Main packages:

* `torch`, `torchvision`, `torchaudio`
* `numpy`, `scipy`, `h5py`, `mat73`, `dill`
* `sigpy`, `pytorch-wavelets`, `PyWavelets`, `opencv-python`
* `tensorboard`, `tqdm`, `PyYAML`, `icecream`, `matplotlib`, `pillow`
* `cupy-cuda12x` (for ESPIRiT calibration via SigPy)

A pinned list is provided in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Structure

```
HD_code/
├── main.py                  # Entry point (train / sample)
├── configs/
│   └── fastMRI.yml          # Configuration (data, model, diffusion, training, sampling, optim)
├── runners/
│   └── diffusion.py         # Train / sample loops
├── models/
│   ├── diffusion.py         # U-Net backbone with sinusoidal time embedding
│   └── ema.py               # EMA weight averaging
├── functions/
│   ├── denoising.py         # Reverse-diffusion samplers (Heat / DDPM variants)
│   ├── losses.py            # Score-matching / noise-estimation losses
│   └── ckpt_util.py
├── datasets/
│   └── __init__.py          # FastMRIv2 dataset wrapper
├── utils.py, utils2.py      # MRI operators (FFT, SENSE, SPIRiT, ESPIRiT, CG, masks)
├── AdaptiveCombine.py       # Multi-coil adaptive combination
├── cg.py
├── optimal_thresh.py
├── train.sh / test.sh       # Convenience launchers
├── LICENSE                  # MIT
└── requirements.txt
```

`main.py` is the common gateway. Run `python main.py --help` for the full CLI. The config file path is relative to `configs/`, so `--config=fastMRI.yml` resolves to `configs/fastMRI.yml`.

## Data Preparation

The default pipeline (`config.data.dataset = "fastMRIv2"`) expects multi-coil k-space samples stored as `.mat` files. Set the two dataset paths in `configs/fastMRI.yml`:

```yaml
data:
    train_kspace_dir: "/path/to/your/training/data"
    sample_kspace_dir: "/path/to/your/test/data"
```

* Training files should contain an image volume keyed `img`.
* Sampling files should contain a k-space volume keyed `ksp`.

Files may live in nested subdirectories — the dataset walks `*.mat` recursively.

## Training

```bash
sh train.sh
# or equivalently:
python main.py --config=fastMRI.yml --exp=./exp --doc=heat
```

Outputs:

* Checkpoints: `./exp/logs/heat/<timestamp>/ckpt_<step>.pth`
* TensorBoard: `./exp/tensorboard/heat/`

Snapshots are written every `training.snapshot_freq` steps (default 1000).

## Sampling

1. Set the checkpoint directory in `configs/fastMRI.yml`:

   ```yaml
   sampling:
       weight: "./exp/logs/heat/<timestamp>"
       ckpt_id: 671000
   ```

2. Run:

   ```bash
   sh test.sh
   # or equivalently:
   python main.py --config=fastMRI.yml --exp=./exp --doc=heat \
                  --sample --fid --timesteps=50 --eta=1 --image_folder=results
   ```

Reconstructions are saved as `.mat` files under `./exp/image_samples/results/`, preserving the directory structure relative to `data.sample_kspace_dir`. Each output contains:

* `cg_sense` — CG-SENSE zero-filled baseline (warm start)
* `label` — fully-sampled reference (from the inverse FFT of `ksp`)
* `diff` — reconstruction from the reverse heat diffusion

## Pretrained Checkpoints

Pretrained weights will be released here. *(TODO: add download link)*

## Citation

If you find this work useful, please cite:

```bibtex
@article{cui2024physics,
  title   = {Physics-Informed {DeepMRI}: k-Space Interpolation Meets Heat Diffusion},
  author  = {Cui, Zhuo-Xu and Liu, Congcong and Fan, Xiaohong and Cao, Chentao and
             Cheng, Jing and Zhu, Qingyong and Liu, Yuanyuan and Jia, Sen and
             Zhou, Yihang and Wang, Haifeng and Zhu, Yanjie and Zhang, Jianping and
             Liu, Qiegen and Liang, Dong},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2024}
}
```

Preprint:

```bibtex
@article{cui2023physics,
  title   = {Physics-Informed {DeepMRI}: Bridging the Gap from Heat Diffusion to k-Space Interpolation},
  author  = {Cui, Zhuo-Xu and Liu, Congcong and Fan, Xiaohong and Cao, Chentao and
             Cheng, Jing and Zhu, Qingyong and Liu, Yuanyuan and Jia, Sen and
             Zhou, Yihang and Wang, Haifeng and Zhu, Yanjie and Zhang, Jianping and
             Liu, Qiegen and Liang, Dong},
  journal = {arXiv preprint arXiv:2308.15918},
  year    = {2023}
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2024 MAiTL-Group, SIAT, CAS.

## Acknowledgments

Parts of the codebase are derived from the official DDPM and NCSN PyTorch implementations.

## Known Issues

The following pre-existing issues are tracked but not addressed in this release; PRs welcome.

* `utils.py` and `utils2.py` contain duplicated MRI operator definitions.
* `functions/denoising.py` defines the `Aclass` data-consistency helper twice.
* `runners/diffusion.py` includes a stale `from curses import KEY_ENTER` import.
* `datasets/celeba.py`, `datasets/ffhq.py`, `datasets/lsun.py` are unused by the FastMRI pipeline.
* `runners/diffusion.py` truncates the sampling schedule to the first 50 steps internally, which can override the value passed via `--timesteps`.
* `runners/diffusion.py` overwrites `config.sampling.mask_type` at runtime; the value in `configs/fastMRI.yml` is currently ignored.
