<!-- PROJECT LOGO -->

# A Noising-Denoising Framework for Point Cloud Upsampling via Normalizing Flows

## Introduction

This repository is for our Pattern Recognition (PR) 2023 paper
'A Noising-Denoising Framework for Point Cloud Upsampling via Normalizing Flows'.
In this paper, we present a novel noising-denoising framework for 3D point cloud upsampling (3DPU),
which aims to generate dense points from a sparse input point cloud.



## Installation
Install the common dependencies from the `requirements.txt` file
```bash
pip install -r requirements.txt
```


## Data Preparation

We provide the **pre-processed supervised and self-supervised data** for the following datasets:
* [PU-GAN (Extracting code: 2k2c)](https://pan.baidu.com/s/1CiNF8VZUhqOXhxh-UhbOig)
* [PU1K (Extracting code: cbhy)](https://pan.baidu.com/s/19RUGuo2bZMgqiPXtadypuw)
* [Sketchfab (Extracting code: 393s)](https://pan.baidu.com/s/1qC3y1-f8UmT_CbRaRC4zCw)

Please put the datasets in `./data`. You can put the datasets elsewhere if you modify the corresponding paths in the args.py.

The directory structure of our project looks like this:
```
│
├── data                   <- Project data
│   └── PU-GAN 
│   │   └── pointclouds
│   │   │   └── train
│   │   │   └── test
│   │   └── meshes 
│   │   │   └── train
│   │   │   └── test            
│   └── PU1K       
│   │   └── pointclouds
│   │   │   └── train
│   │   │   └── test
│   │   └── meshes 
│   │   │   └── test  
│   └── Sketchfab  
│   │   └── pointclouds
│   │   │   └── train
│   │   │   └── test
│   │   └── meshes 
│   │   │   └── train
│   │   │   └── test      
```



## Running
> Current settings in `args.py` are tested on one NVIDIA GeForce RTX 3090. To reduce memory consumption, you can set `batch_size`, or `patch_size` to a smaller number.

Train model on PU-GAN or Sketchfab dataset:
```bash
python train.py
```

Train model on PU1K dataset:
```bash
python train_pu1k.py
```

Test model on PU-GAN, PU1K or Sketchfab dataset:
```bash
python test.py
```
  

## Citation
If you find our code or paper useful, please cite
```bibtex
@article{HU2023109569,
  title     = {A Noising-Denoising Framework for Point Cloud Upsampling via Normalizing Flows},
  author    = {Xin Hu, Xin Wei and Jian Sun},
  journal = {Pattern Recognition},
  volume  = {140},
  pages   = {109569},
  issn    = {0031-3203},
  year      = {2023}
```
