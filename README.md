# Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation
This repository accompanies the submission titled "Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation".

## 🚀 Getting Started
> ### Conda environment for **HTDemucs** and **Mel-RoFo. (S)**:
> 1. ```$ conda env create -f ./env_info/mss_baseline_env_conda.yml```
> 2. ```$ pip install -r ./env_info/mss_baseline_env_requirements.txt```

> ### Conda environment for **SGMSVS** model:
> 1. set ```CUDA_HOME```environment variable in ```env_info/sgmsvs_env_conda.yml``` to path where conda environment will be located
> 2. ```$ conda env create -f ./env_info/sgmsvs_env_conda.yml```
> 3. ```$ pip install -r ./env_info/sgmsvs_env.txt```

> ### Conda environment for finetuning **BigVGAN** (**Mel-RoFo. (S)+BigVGAN**):
> 1. set ```CUDA_HOME```environment variable in ```env_info/sgmsvs_env_conda.yml``` to path where conda environment will be located
> 2. ```$ conda env create -f ./env_info/bigvgan_env.yml```
> 3. ```$ pip install -r ./env_info/bigvgan_env_requirements.txt```

## 🏋🏽‍♀️🏃🏽‍♀️‍➡️ Training and Inference
The folder 00_training_and_inference contains all code required to carry out training and inference for all the models mentioned in the paper.

In order to find out on how to start the training of each model you can the bash-files ```*.sh```

In order to resume training or use 
The trained models (HTDemucs, Mel-Rofo. (S), SGMSVS and the finetuned BigVGAN) can be found here: <url>https://drive.google.com/drive/folders/13D_0ciDODkNv9q5W9l2s2WsmpHlZJvnS?usp=sharing</url>
>
The model checkpoint for Mel-Rofo. (L) was taken from [3].

### Training
#### ./00_training_and_inference/train_baseline.py
This python-script was adapted from [1] and can be used to train HTDemucs and Mel-RoFo. (S)
The baseline model classes were taken from [2] and [3].

#### ./00_training_and_inference/train_sgmsvs.py
This python-script was adapted from [1] and can be used to train the SGMSVS model (train_sgmsvs.py).

#### ./00_training_and_inference/train_finetuned_bigvgan.py
This python-script was adapted from [4] and can be used to task-specifically finetune the vocoder-based BigVGAN for singing voice separation.
### Inference

## 🧮 Evaluation and Correlation Analysis
01_evaluation_and_correlation contains the code to compute all objective evaluation metrics and all *.csv files of the evaluated objective audio quality metrics.

Note: to compute ViSQOL metrics the ViSQOL API from <url>https://github.com/google/visqol</url> has to be installed locally. 

This folder contains the DMOS data collected with a degradation category rating. \n
It also contains code to analyze the listening test results. 

It also allows to reproduce the correlation analysis results of the paper.


## 🎼 Audio Examples
In the folder ```03_audio_examples``` 12 audio examples of each model can be found. The audio examples are loudness normalized to -18 dBFS according to EBU R128.

## trained models
The trained models used in this paper can be downloaded from <add-url-here>
12 audio examples of each model can be found in this folder.

## env_info
Contains the *.yml files to create the conda environments and *.txt requirements to install all packages using pip.
All environments were tested using a Debian machine. 
To create one of the environments carry out the following steps:
1.) conda create 
2.) pip install ...

## trained models
The trained models used in this paper can be downloaded from <add-url-here>

## Third party code
### Third party folders
All third-party code is contained in separate folders, each of which is specifically listed in this README.md file, and the corresponding LICENSE files for each folder are located within their respective directories.

The used third party folders are the following:
- Path: 00_training_and_inference/sgmsvs/sgmse
    - Reference: [1]

## References:
These references pertain only to the code reused to set up this codebase. For more information on the origins of the models, please refer to the references within the paper.

>[1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023, Git Repository: <url>https://github.com/sp-uhh/sgmse</url>
>  
>[2] Roman Solovyev, Alexander Stempkovskiy, Tatiana Habruseva. "Benchmarks and leaderboards for sound demixing tasks", 2023, Git Repository: <url>https://github.com/ZFTurbo/Music-Source-Separation-Training</url>
>
>[3] Kimberley Jensen, "Mel-Band-Roformer-Vocal-Model", 2024, Git Repository:<url>https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model</url>
>
>[4] Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon, "Big{VGAN}: A Universal Neural Vocoder with Large-Scale Training", in Proc. ICLR, 2023, Git Repository: <url>https://github.com/NVIDIA/BigVGAN</url>

```bib
@article{richter2023speech,
  title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351-2364},
  year={2023},
  doi={10.1109/TASLP.2023.3285241}
}
```
```bib
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      howpublished={\url{https://github.com/ZFTurbo/Music-Source-Separation-Training}},
      primaryClass={cs.SD},
      url={https://github.com/ZFTurbo/Music-Source-Separation-Training}
      }
```
```bib
@misc{jensen2024melbandroformer,
  author       = {Kimberley Jensen},
  title        = {Mel-Band-Roformer-Vocal-Model},
  year         = {2024},
  howpublished = {\url{https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}},
  note         = {GitHub repository},
  url          = {https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}
}
```
```bib
@inproceedings{
lee2023bigvgan,
title={Big{VGAN}: A Universal Neural Vocoder with Large-Scale Training},
author={Sang-gil Lee and Wei Ping and Boris Ginsburg and Bryan Catanzaro and Sungroh Yoon},
booktitle={in Proc. ICLR, 2023},
year={2023},
url={https://openreview.net/forum?id=iTtGCMDEzS_}
}
```

## Cite This Work:
