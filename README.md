# Recursive Noise Diffusion

![Recursive Noise Diffusion diagram](recursive_noise_diffusion_diagram.svg?raw=true "Recursive Noise Diffusion")

This repo is the official implementation of 
[Multi-Class Segmentation from Aerial Views using Recursive Noise Diffusion](https://arxiv.org/abs/2212.00787)

The core idea of Recursive Noise Diffusion is the _recursive denoising_ process, as shown in the figure above. 
Training with _recursive denoising_ involves progressing through each time step t from T to 1, recursively (as the name suggests), which allows a portion of the predicted error to propagate.
This process is initialised with pure noise. The noise function diffuses the previous predicted segmentation, then the model denoises this diffused segmentation given a conditioning RGB image. The denoised predicted segmentation is compared to the ground truth. Notably, the ground truth segmentation is never used as part of the input to the model. This process is agnostic to the choice of noise function, diffusion model and loss.

## Getting Started

### Setup

- Clone this repo:

```bash
git clone https://github.com/benediktkol/recursive-noise-diffusion.git
cd recursive-noise-diffusion
```

- Install requirements:

```bash
pip install -r requirements.txt
```

- [Optional] Create a conda environment:

```bash
conda env create -f rnd.yml
conda activate rnd
```

- Download data
  - [Vaihingen Buildings](https://drive.google.com/open?id=1nenpWH4BdplSiHdfXs0oYfiA5qL42plB)
  - [UAVid](https://uavid.nl)

- File structure

```bash
  data
  ├── UAVid
  │   ├── uavid_test
  │   │   ├── seq21
  │   │   │   └── Images
  │   │   │       ├── 000000.png
  │   │   │       └── ...
  │   │   └── ...
  │   ├── uavid_train
  │   │   ├── seq1
  │   │   │   ├── Images
  │   │   │   │   ├── 000000.png
  │   │   │   │   └── ...
  │   │   │   └── Labels
  │   │   │       ├── 000000.png
  │   │   │       └── ...
  │   │   └── ...
  │   └── uavid_val
  │       ├── seq16
  │       │   ├── Images
  │       │   │   ├── 000000.png
  │       │   │   └── ...
  │       │   └── Labels
  │       │       ├── 000000.png
  │       │       └── ...
  │       └── ...
  └── Vaihingen_buildings
      ├── all_buildings_mask_001.png
      ├── ...
      ├── building_001.png
      ├── ...
      ├── building_gt_001.png
      ├── ...
      ├── building_mask_001.png
      └── ...
 recursive-noise-diffusion (this repo)
 ├── test.py
 ├── train.py
 └── ...
```


### Train

To train a model use ```train.py```, for example:

```bash
python train.py --dataset vaihingen --scale_procedure loop --n_scales 3 --n_timesteps 25
```

### Evaluation

To evaluate a model use ```test.py```, for example:

```bash
python test.py --load_checkpoint /path/to/checkpoint.pt --dataset vaihingen --scale_procedure loop --n_scales 3 --n_timesteps 25
```

## Cite

```
@InProceedings{Kolbeinsson_2024_WACV,
    author    = {Kolbeinsson, Benedikt and Mikolajczyk, Krystian},
    title     = {Multi-Class Segmentation From Aerial Views Using Recursive Noise Diffusion},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {8439-8449}
}
```
