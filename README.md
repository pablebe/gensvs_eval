# Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation
This repository accompanies the submission titled "Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation".

## 🚀 Getting Started
To run all of the code in this repository we recommend setting up the following conda environments, neccesary to infer or train the models mentioned in the paper.

### Conda environment for training and inference of **HTDemucs** and **Mel-RoFo. (S)** (```gensvs_eval_baseline_env```):
> 1. Create the conda environment:  
```$ conda env create -f ./env_info/mss_baseline_env_conda.yml```  
> 2. Install additional python dependencies:  
```$ pip install -r ./env_info/mss_baseline_env_requirements.txt```

### Conda environment for training and inference of **SGMSVS** model (```gensvs_eval_sgmsvs_env```):
> 1. Set the ```CUDA_HOME```environment variable in ```env_info/sgmsvs_env_conda.yml``` to path where the CUDA toolkit is installed. This can be the path where conda environment will be located
> 2. Create the conda environment:   
```$ conda env create -f ./env_info/sgmsvs_env_conda.yml```
> 3. Install additional python dependencies:  
```$ pip install -r ./env_info/sgmsvs_env.txt```

### Conda environment for training/finetuning and inference of **Mel-RoFo. (S)+BigVGAN** (```gensvs_eval_bigvgan_env```):
> 1. Set ```CUDA_HOME```environment variable in ```env_info/sgmsvs_env_conda.yml``` to path where conda environment will be located
> 2. Create the conda environment:  
```$ conda env create -f ./env_info/bigvgan_env.yml```
> 3. Install additional python dependencies:  
```$ pip install -r ./env_info/bigvgan_env_requirements.txt```

### Conda environment for evaluation of FAD and MSE metrics (```gensvs_fad_mse_eval_env```):
For evaluating the model's performance using the proposed FAD and MSE evaluation metrics an additional conda environment has to be set up. This conda envirnonment builds on Microsoft's Frechet Audio Distance Toolkit [5].
> 1. Create the conda environment:
```$ conda env create -f ./env_info/svs_fad_mse_eval_env.yml``` 
> 2. Install additional python dependencies:
```$ pip install -r ./env_info/svs_fad_mse_eval_env.txt```  
> 3. Test fadtk installation with: ```python -m fadtk.test```

### Conda environment for evaluation of other metrics and correlation analysis (```gensvs_eval_env```)
For evaluating the model's performance using the objective evaluation metrics and reproduce the correlation analysis of the paper we recommend setting up an additional conda environment with:
> 1. Create the conda environment:
```$ ``` 

> 2. Install additional python dependencies:
```$ ```

In addition to the dependencies the ViSQOL Python API and a running Matlab installation is required.
## 🏋🏽‍♀️🏃🏽‍♀️‍➡️ Training and Inference
The folder ```00_training_and_inference``` contains all code required to carry out training and inference for all the models mentioned in the paper. 

### Trained models
 - HTDemucs, Mel-Rofo. (S), SGMSVS and the finetuned BigVGAN checkpoints (generator and discriminator) can be downloaded here: https://drive.google.com/drive/folders/13D_0ciDODkNv9q5W9l2s2WsmpHlZJvnS?usp=sharing  
 - The model checkpoint for Mel-Rofo. (L) can be downloaded from [3].

### Inference
To run all models mentioned in the paper on a folder of musical mixtures you can either use the python inference scripts or use the provided bash scripts which show how to call the python scripts:
#### Discriminative baseline models: HTDemucs, Mel-RoFo. (S) & Mel-RoFo. (L) 
To run the mask-based baseline models on a folder of musical mixtures the following scripts can be used:  
- Python:
  - HTDemucs & Mel-RoFo. (S): ```00_training_and_inference/inference_baseline.py```  
  - Mel-RoFo. (L): ```00_training_and_inference/inference_melroformer_large.py```  
- Bash:
  - HTDemucs: ```$ 00_training_and_inference/infer_htdemucs.sh```
  - Mel-RoFo. (S):```$ 00_training_and_inference/infer_melroformer_small.sh``` 
  - Mel-RoFo. (L): ```$ 00_training_and_inference/infer_melroformer_large.sh```

#### Generative models: SGMSVS & Mel-RoFo. (S) + BigVGAN
To run the generative models on a folder of musical mixtures the following scripts can be used:  
- Python: 
  - SGMSVS: ```00_training_and_inference/inference_sgmsvs.py```
  - Mel-RoFo. (S) + BigVGAN: ```00_training_and_inference/inference_melroformer_small_bigvgan.py```
- Bash:
  - SGMSVS: ```$ 00_training_and_inference/infer_sgmsvs.sh```
  - Mel-RoFo. (S) + BigVGAN: ```00_training_and_inference/infer_melroformer_small_bigvgan.sh```
### Training
Below the python training scripts and example bash scripts are listed for all models to reproduce all trainings and the set parameters mentioned in the paper:
#### Discriminative Baselines: HTDemucs & Mel-RoFo (S)
To train the dicriminative mask-based baselines use:
  - Python: ```00_training_and_inference/train_baseline.py```
  - Bash: 
    - ```$ 00_training_and_inference/train_htdemucs.sh```
    - ```$ 00_training_and_inference/train_melroformer.sh```
#### Score-based generative model: SGMSVS
To train the SGMSVS model, use the scripts:
- Python: ```00_training_and_inference/train_sgmsvs.py```
- Bash: ```$ 00_training_and_inference/train_sgmsvs.sh```

#### GAN-based model: Mel-RoFo. (S) + BigVGAN
To task-specifically finetune BigVGAN for singing voice separation with Mel-RoFo. (S) you can use:
- Python: ```00_training_and_inference/train_finetune_bigvgan.py```
- Bash: ```$ 00_training_and_inference/train_bigvgan.sh```

## 🧮 Evaluation and Correlation Analysis
01_evaluation_and_correlation contains the code to compute all objective evaluation metrics and all *.csv files of the evaluated objective audio quality metrics.

ADD Evaluation plot here!

Note: to compute ViSQOL metrics the ViSQOL API from <url>https://github.com/google/visqol</url> has to be installed locally. 

This folder contains the DMOS data collected with a degradation category rating. \n
It also contains code to analyze the listening test results. 

It also allows to reproduce the correlation analysis results of the paper.


## 🎼 Audio Examples
In the folder ```03_audio_examples``` 12 audio examples of each model can be found.  
The audio examples are mono and loudness normalized to -18 dBFS according to EBU R128.



## Third party code
### Third party folders
All third-party code is contained in separate folders, each of which is specifically listed in this README.md file, and the corresponding LICENSE files for each folder are located within their respective directories.

<LIST-THIRD-PARTY-CODE-HERE!>

## References:
These references pertain only to the code reused to set up this codebase. For more information on the origins of the models, please refer to the references within the paper.

>[1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023, Git Repository: <url>https://github.com/sp-uhh/sgmse</url>
>  
>[2] Roman Solovyev, Alexander Stempkovskiy, Tatiana Habruseva. "Benchmarks and leaderboards for sound demixing tasks", 2023, Git Repository: <url>https://github.com/ZFTurbo/Music-Source-Separation-Training</url>
>
>[3] Kimberley Jensen, "Mel-Band-Roformer-Vocal-Model", 2024, Git Repository:<url>https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model</url>
>
>[4] Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon, "Big{VGAN}: A Universal Neural Vocoder with Large-Scale Training", in Proc. ICLR, 2023, Git Repository: <url>https://github.com/NVIDIA/BigVGAN</url>
>
>[5] Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou, "Adapting Frechet Audio Distance for Generative Music Evaluation", in Proc. ICASSP, 2024, Git Repository: <url>https://github.com/microsoft/fadtk</url>
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
@inproceedings{lee2023bigvgan,
               title={Big{VGAN}: A Universal Neural Vocoder with Large-Scale Training},
               author={Sang-gil Lee and Wei Ping and Boris Ginsburg and Bryan Catanzaro and Sungroh Yoon},
               booktitle={in Proc. ICLR, 2023},
               year={2023},
               url={https://openreview.net/forum?id=iTtGCMDEzS_}
              }
```
```bib
@inproceedings{fadtk,
               title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
               author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
               booktitle = {Proc. IEEE ICASSP 2024},
               year = {2024},
               url = {https://arxiv.org/abs/2311.01616},
              }
```
## Cite This Work:
Citation if paper is accepted will be added here!